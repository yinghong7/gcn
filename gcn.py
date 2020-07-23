import pandas as pd
import numpy as np
from itertools import chain
from numpy import load, save
import h5py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import FastText
from gensim.utils import tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from nltk import PorterStemmer, word_tokenize
from keras.wrappers.scikit_learn import KerasClassifier

project = pd.read_csv ('C:/Users/yh448/OneDrive - University of Cambridge/Name_text/paper/full_short_.csv')
project = project.rename(columns={"Logic_y": "Succ_logic", "Lag_y": "Succ_lag", "Logic_x": "Pred_logic", "Lag_x": "Pred_lag"})
p_id = project.drop_duplicates (subset = 'Project_ID')
p_id['Project_ID'].astype('int32')
id_list = p_id['Project_ID'].tolist()

def chainer(s):
    return list(chain.from_iterable(s.str.split(', ')))


class graph:
    def __init__ (self, df, pred, logic, lag, weight, location, label):
        self.df = df
        self.pred = pred
        self.logic = logic
        self.lag = lag
        self.weight = weight
        self.location = location
        self.label = label
        
        
    def order_actv (self):
        id_list = (self.df.drop_duplicates (subset = 'Project_ID'))['Project_ID'].tolist()
        ordered_actv = self.df.head(0)
        for i in id_list:
            a = self.df.loc[self.df['Project_ID'] == i,]
            a['Graph_x'] = np.arange(a.shape[0])
            ordered_actv = ordered_actv.append (a)
        return ordered_actv
    
    
#This is just a function to clean my dataset
    def expand_predecessor (self):
        df0 = self.order_actv()
        df0[self.pred] = df0[self.pred].astype(str)
        df0[self.logic] = df0[self.logic].astype(str)
        df0[self.lag] = df0[self.lag].astype(str)
        predecessor_list = chainer(df0[self.pred])
        logic_list = chainer(df0[self.logic])
        lag_list = chainer(df0[self.lag])
        lens = df0[self.pred].str.split(', ').map(len)
        expand = pd.DataFrame({self.pred: predecessor_list, 'Clean_task_code': np.repeat(df0['Clean_task_code'], lens), 
                      self.logic: logic_list, self.lag: lag_list})
        x2 = self.df.loc[:,['Clean_task_code','Duration', 'Project_ID','Task_name', 'Clean_task_name','WBS', 'Graph_x','r_length', 'r_pos', 'Foundation_detail']]
        expand_ = expand.merge(x2)
        return expand_
    
    
    def logic_weight (self):
        df1 = self.expand_predecessor()
        ls_l = df1[self.logic].tolist()
        logic_percent = []
        summarise_logic = df1.groupby(self.logic).count()[['Clean_task_code']]
        for i in ls_l:
            if i == 'PR_FF':
                logic_percent.append ((summarise_logic.loc['PR_FF', 'Clean_task_code'])/len(df1))
            if i == 'PR_SS':
                logic_percent.append ((summarise_logic.loc['PR_SS', 'Clean_task_code'])/len(df1))
            if i == 'PR_FS':
                logic_percent.append ((summarise_logic.loc['PR_FS', 'Clean_task_code'])/len(df1))
            if i == 'PR_SF':
                logic_percent.append ((summarise_logic.loc['PR_SF', 'Clean_task_code'])/len(df1))
            if i == 'None':
                logic_percent.append ((summarise_logic.loc['None', 'Clean_task_code'])/len(df1))
            if i == 'nan':
                logic_percent.append ((summarise_logic.loc['nan', 'Clean_task_code'])/len(df1))
        df1[self.weight] = logic_percent
        return df1
    
    
    def predecessor_location(self):
        df2 = self.logic_weight()
        predecessor = df2[self.pred].tolist()
        ls1 = []
        for i in predecessor:
            if i != 'nan':
                location = df2.loc[df2['Clean_task_code']==i, ['Graph_x']].median()['Graph_x']
            else: location = int(10000)
            ls1.append (location)
        df2[self.location] = ls1
        df2[self.location] = df2[self.location].fillna(10000)
        return df2
    
    
    def group_label(self):
        df3 = self.predecessor_location()
        name_ls = df3[self.pred].tolist()
        ls1 = []
        for i in name_ls:
            g = df3.loc[df3['Clean_task_code']==i,]
            if g.shape[0] != 0:
                num = g.Foundation_detail.any()
            else: num = 'None'
            ls1.append (num)
        df3[self.label] = ls1
        return df3

    
#pred is row, succ is column, loc = row*dim+column
def adjacency_matrix (df, pred, succ, location, pred_logic_weight, succ_logic_weight):
    predecessor = df[pred].tolist()
    successor = df[succ].tolist()
    location = df[location].tolist()
    predecessor_logic_weight = df[pred_logic_weight].tolist()
    successor_logic_weight = df[succ_logic_weight].tolist()
    
    x = df.drop_duplicates(subset = 'Clean_task_code')
    length = x.shape[0]
    
    pred_location, pred_weight, succ_location, succ_weight = [],[],[],[]
    for i, j, m in zip(predecessor, location, predecessor_logic_weight):
        if i == 10000:
            pred_location.append (0)
            pred_weight.append (0)
        else: 
            pred_location.append(i*length+j)
            pred_weight.append(m)
    for i, j, k in zip(successor, location, successor_logic_weight):
        if i == 10000:
            succ_location.append (0)
            succ_weight.append (0)
        else: 
            succ_location.append(i+j*length)
            succ_weight.append(k)         
            
    adjacency_m = np.zeros((length, length))
    np.put(adjacency_m, pred_location, pred_weight)
    np.put(adjacency_m, succ_location, succ_weight)
    np.fill_diagonal(adjacency_m, 1)
    return adjacency_m


def degree_matrix(adjacency_matrix):
    ls1 = []
    length = len(adjacency_matrix)
    degree_m = np.zeros((length, length))
    for i in range (length):
        ls1.append (np.sum(adjacency_matrix[i]))
    np.fill_diagonal(degree_m, ls1)
    return degree_m


def normal_matrix (adjacency_matrix, degree_matrix):
    x = np.power(degree_matrix.diagonal(), -0.5)
    y = np.zeros((len(degree_matrix), len(degree_matrix)))
    np.fill_diagonal(y, x)
    mid = np.dot(y, adjacency_matrix)
    a_delta = np.dot (mid, y)
    return a_delta


def feature_input (df):
    x = df.drop_duplicates(subset = 'Clean_task_code')
    empty = np.zeros(60).reshape(1,60)
    ls = x['Clean_task_name'].tolist()
    ls_token = token(prep(ls))
    length_curr = x['r_length'].replace('None',0).tolist()
    pos_curr  = x['r_pos'].replace('None',0).tolist()
    x['Duration'] = x['Duration'].astype(int)
    a = x['Duration'].to_numpy()
    dur_curr = (np.log(a+1)).reshape(len(ls),1)
    for i in range (len(ls)):
        n = ' '.join(ls_token[i])
        m = trained_model.wv[n].reshape(1,60)
        empty = np.append (empty, m, axis = 0)
    empty_ = np.delete(empty, 0, axis = 0)
    len_curr = np.asarray(length_curr).reshape(len(ls),1)
    po_curr = np.asarray(pos_curr).reshape(len(ls),1)
    empty_ = np.append (empty_, len_curr, axis = 1)
    empty_ = np.append (empty_, po_curr, axis = 1)
    empty_ = np.append (empty_, dur_curr, axis = 1)
    empty_ = empty_.astype(float)
    return empty_ 


lemmatizer = WordNetLemmatizer()
def prep (ls):
    test_token, test_lemma, test_return = [], [], []
    for i in range (len (ls)):
        token = tokenize(str(ls[i]))
        test_token.append (list(token))  
    for i in range (len (test_token)):
        x = lemmatizer.lemmatize(str(test_token[i]))
        x1 = eval('' + x + '')
        test_lemma.append (x1)    
    for i in range (len (test_lemma)):
        test_return.append (' '.join(test_lemma[i]))       
    return test_return


def token (ls):
    ps = PorterStemmer()
    token_word = []
    for i in range (len(ls)):
        ls1 = word_tokenize(str(ls[i]))
        ls2 = []
        for word in ls1:
            word = word.lower()
            word = ps.stem(word)   
            ls2.append (word)
        token_word.append (ls2)
    return token_word
    
    
test_ = pd.read_csv ('D:/OneDrive - University of Cambridge/Name_text/paper/test_head.csv')
for i in id_list:
    x2 = project.loc[project['Project_ID']== i,]
    pred = graph(x2, 'Predecessor', 'Pred_logic', 'Pred_lag', 'Weight_pred', 'Pred_graph', 'Pred_found')
    succ = graph(x2, 'Successor', 'Succ_logic', 'Succ_lag', 'Weight_succ', 'Succ_graph', 'Succ_found')
    pred_label = pred.group_label()
    succ_label = succ.group_label()
    label2 = pred_label.merge (succ_label, on = ('Clean_task_code','Duration', 'Project_ID','Task_name', 'Clean_task_name','WBS', 'Graph_x','r_length', 'r_pos', 'Foundation_detail'))
    label2['Pred_graph'] = np.where(label2.Pred_found.any() == label2.Foundation_detail.any() != 'None' , label2['Graph_x'], label2['Pred_graph'])
    label2['Succ_graph'] = np.where(label2.Succ_found.any() == label2.Foundation_detail.any() != 'None' , label2['Graph_x'], label2['Succ_graph'])
    test_ = test_.append (label2)
    #amatrix = adjacency_matrix ('Pred_graph', 'Succ_graph', 'Graph_x', 'Weight_pred', 'Weight_succ', label2)
    #print (amatrix.shape)
    
train_name = project['Clean_task_name'].tolist()
train_name_ = token(prep(train_name))
trained_model = FastText(train_name_, size = 60, min_count = 1, negative = 7, workers = 5)

init = np.zeros((1,1))
f_init = np.zeros((1,63))
for i in id_list:
    x2 = test_.loc[test_['Project_ID']==i,]
    x2_ = x2.drop_duplicates(subset = 'Clean_task_code')
    amatrix = adjacency_matrix (x2_, 'Pred_graph', 'Succ_graph', 'Graph_x', 'Weight_pred', 'Weight_succ')
    A = init.shape[0]
    B = amatrix.shape[0]
    init = np.block([[init,np.zeros((A,B))],[np.zeros((B,A)), amatrix]])
    feature = feature_input(x2_)
    f_init = np.append (f_init, feature, axis=0)
    print(x2_.shape, amatrix.shape, init.shape, f_init.shape, feature.dtype, i)
    
    
init_ = np.delete(init, 0, 0)
init_f = np.delete(init_, 0, 1)
degree_m = degree_matrix (init_f)
normal_m = normal_matrix (degree_m, init_f)
feature_m = np.delete(f_init, 0, 0)
train_in = np.dot (normal_m, feature_m)


batch_size = [20, 40, 60, 70, 80]
epochs = [60, 70, 80, 90, 100]
optimizer = ['SGD']
neurons = [50,70,100,120,150]

def create_model(optimizer='sgd', neurons = 20): 
    gcn_model = models.Sequential()
    gcn_model.add(layers.Dense(neurons, activation='relu', input_dim = 63))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='relu'))
    gcn_model.add(layers.Dense(1))
    #optimizer = tf.keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    gcn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return gcn_model

gcn_model = KerasClassifier(build_fn = create_model, verbose = 0)
param_grid_1 = dict(batch_size = batch_size, epochs = epochs, optimizer=optimizer, 
                    neurons = neurons)#
grid_1 = GridSearchCV(estimator = gcn_model, param_grid=param_grid_1, 
                      n_jobs=-1, cv=5, scoring='f1_macro')
label_x = test_nodup['Foundation_detail'].to_numpy()
grid_result = grid_1.fit(train_in, label_x)
best = grid_result.best_estimator_
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print (classification_report(label_x, best.predict(train_in)))
print (confusion_matrix(label_x, best.predict(train_in)))
