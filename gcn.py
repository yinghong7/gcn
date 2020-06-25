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

project = pd.read_csv ('C:/Users/yh448/OneDrive - University of Cambridge/Name_text/paper/full_short_.csv')
project = project.rename(columns={"Logic_y": "Succ_logic", "Lag_y": "Succ_lag", "Logic_x": "Pred_logic", "Lag_x": "Pred_lag"})
p_id = project.drop_duplicates (subset = 'Project_ID')
p_id['Project_ID'].astype('int32')
id_list = p_id['Project_ID'].tolist()

class graph ():
    def __init__ (self, df, pred, logic, lag, weight, order, label):
        self.df = df
        self.pred = pred
        self.logic = logic
        self.lag = lag
        self.weight = weight
        self.order = order
        self.label = label
    def chainer(s):
        return list(chain.from_iterable(s.str.split(', ')))

    def full_graph (self):
        self.df[pred] = self.df[self.pred].astype(str)
        self.df[logic] = self.df[self.logic].astype(str)
        self.df[lag] = dself.f[self.lag].astype(str)
        a = self.chainer(self.df[self.pred])
        b = self.chainer(self.df[self.logic])
        c = self.chainer(self.df[self.lag])
        lens = self.df[self.pred].str.split(', ').map(len)
        graph = pd.DataFrame({self.pred: a, 'Clean_task_code': np.repeat(self.df['Clean_task_code'], lens), 
                      self.logic: b, self.lag: c,})
        x2 = self.df.loc[:,['Clean_task_code','Duration', 'Project_ID','Task_name', 'Clean_task_name','WBS', 'Graph_x','r_length', 'r_pos', 'Foundation_detail']]
        graph_ = graph.merge(x2)
        return graph_

    def weight (self):
        df = self.full_graph()
        ls_l = df[self.logic].tolist()
        ls = []
        mid = df.groupby(self.logic).count()[['Clean_task_code']]
        for i in ls_l:
            if i == 'PR_FF':
                ls.append ((mid.loc['PR_FF', 'Clean_task_code'])/len(full_))
            if i == 'PR_SS':
                ls.append ((mid.loc['PR_SS', 'Clean_task_code'])/len(full_))
            if i == 'PR_FS':
                ls.append ((mid.loc['PR_FS', 'Clean_task_code'])/len(full_))
            if i == 'PR_SF':
                ls.append ((mid.loc['PR_SF', 'Clean_task_code'])/len(full_))
            if i == 'None':
                ls.append ((mid.loc['None', 'Clean_task_code'])/len(full_))
            if i == 'nan':
                ls.append ((mid.loc['nan', 'Clean_task_code'])/len(full_))
        df[self.weight] = ls
        return df

    def pend_node(self):
        df = self.wegiht()
        name_ls = df[self.pred].tolist()
        ls1 = []
        for i in name_ls:
            if i != 'nan':
                num = df.loc[df['Clean_task_code']==i, ['Graph_x']].median()['Graph_x']
            else: num = int(10000)
            ls1.append (num)
        df[self.order] = ls1
        df[self.order] = df[self.order].fillna(10000)
        return df

    def group_label(self):
        df = self.pend_node()
        name_ls = df[self.pred].tolist()
        ls1 = []
        for i in name_ls:
            g = df.loc[df['Clean_task_code']==i,]
            if g.shape[0] != 0:
                num = g.Foundation_detail.any()
            else: num = 'None'
            ls1.append (num)
        df[self.label] = ls1
        return df
#pred is row, succ is column, loc = row*dim+column
class matrix ():
    def __init__ (self, df, pred, succ, order, weight_1, weight_2):
        self.df = df
        self.pred = pred
        self.succ = succ
        self.order = order
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def adjacency_matrix (self):
        name1_ls = self.df[self.pred].tolist()
        graph1 = self.df[self.order].tolist()
        weight1 = self.df[self.weight_1].tolist()
        name2_ls = self.df[self.succ].tolist()
        weight2 = self.df[self.weight_2].tolist()
        x = self.df.drop_duplicates(subset = 'Clean_task_code')
        length = x.shape[0]
    
        ls1,ls2,ls3,ls4 = [],[],[],[]
        for i, j, m in zip(name1_ls, graph1, weight1):
            if i == 10000:
                ls1.append (0)
                ls2.append (0)
            else: 
                ls1.append(i*length+j)
                ls2.append(m)
        for i, j, k in zip(name2_ls, graph1, weight2):
            if i == 10000:
                ls3.append (0)
                ls4.append (0)
            else: 
                ls3.append(i+j*length)
                ls4.append(k)         
            
        x = np.zeros((length, length))
        np.put(x, ls1, ls2)
        np.put(x, ls3, ls4)
        np.fill_diagonal(x, 1)
        return x

    def degree_matrix(self):
        ls1 = []
        amatrix = self.adjacency_matrix ()
        length = len(amatrix)
        x = np.zeros((length, length))
        for i in range (length):
            ls1.append (np.sum(amatrix[i]))
        np.fill_diagonal(x, ls1)
        return x

    def normal_matrix (self):
        degree_m = self.degree_matrix()
        ad_m = self.adjacency_matrix()
        x = np.power(degree_m.diagonal(), -0.5)
        y = np.zeros((len(degree_m), len(degree_m)))
        np.fill_diagonal(y, x)
        mid = np.dot(y, ad_m)
        a_delta = np.dot (mid, y)
        return a_delta


def feature_input ():
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
    
    
test_ = pd.read_csv ('C:/Users/yh448/OneDrive - University of Cambridge/Name_text/paper/test_head.csv')
for i in id_list:
    x2 = project.loc[project['Project_ID']== i,]
    pred_label = graph.group_label(x2, 'Predecessor', 'Pred_logic', 'Pred_lag', 'Weight_pred', 'Pred_graph')
    succ_label = graph.group_label(x2, 'Successor', 'Succ_logic', 'Succ_lag', 'Weight_succ', 'Succ_graph')
    label2 = pred_label.merge (succ_label, on = ('Clean_task_code','Duration', 'Project_ID','Task_name', 'Clean_task_name','WBS', 'Graph_x','r_length', 'r_pos', 'Foundation_detail']))
    label2['Pred_graph'] = np.where(label2.Pred_found.any() == label2.Foundation_detail.any() != 'None' , label2['Graph_x'], label2['Pred_graph'])
    label2['Succ_graph'] = np.where(label2.Succ_found.any() == label2.Foundation_detail.any() != 'None' , label2['Graph_x'], label2['Succ_graph'])
    test_ = test_.append (label2)
    #amatrix = adjacency_matrix ('Pred_graph', 'Succ_graph', 'Graph_x', 'Weight_pred', 'Weight_succ', label2)
    #print (amatrix.shape)
    
    
init = np.zeros((1,1))
f_init = np.zeros((1,63))
for i in id_list:
    x2 = test_.loc[test_['Project_ID']==i,]
    amatrix = matrix.adjacency_matrix ('Pred_graph', 'Succ_graph', 'Graph_x', 'Weight_pred', 'Weight_succ', x2)
    A = init.shape[0]
    B = amatrix.shape[0]
    init = np.block([[init,np.zeros((A,B))],[np.zeros((B,A)), amatrix]])
    feature = feature_input(x2)
    f_init = np.append (f_init, feature, axis=0)
    print(x2.shape, amatrix.shape, init.shape, f_init.shape, feature.dtype, i)
init_ = np.delete(init, 0, 0)
init_f = np.delete(init_, 0, 1)

degree_m = matrix.degree_matrix (init_f)
normal_m = matrix.normal_matrix (degree_m, init_f)
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
    gcn_model.add(layers.Dense(neurons, activation='relu'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='relu'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='relu'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(neurons, activation='relu'))
    gcn_model.add(layers.Dense(neurons, activation='softmax'))
    gcn_model.add(layers.Dense(1))
    #optimizer = tf.keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    gcn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return gcn_model

gcn_model = KerasClassifier(build_fn = create_model, verbose = 0)
param_grid_1 = dict(batch_size = batch_size, epochs = epochs, optimizer=optimizer, 
                    neurons = neurons)#
grid_1 = GridSearchCV(estimator = gcn_model, param_grid=param_grid_1, 
                      n_jobs=-1, cv=5, scoring='f1_macro')
label_x = test_nodup['Ont_label'].to_numpy()
grid_result = grid_1.fit(train_in, label_x)
best = grid_result.best_estimator_
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print (classification_report(label_x, best.predict(train_in)))
print (confusion_matrix(label_x, best.predict(train_in)))