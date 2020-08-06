import pandas as pd
import numpy as np
import itertools
import collections
import enum
import heapq
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
from sklearn import preprocessing


def childern (df, successor_name, current_name):
    sub_child = []
    df[successor_name] = df[successor_name].astype(str)
    for i in range (len(df)):
        successor_id = iter ((df[successor_name].tolist())[i].split(', '))
        one = iter(np.ones(len(df[successor_name])))
        sub_child.append (dict(zip(successor_id, one)))
    succ_list = list(itertools.chain.from_iterable(df[successor_name].str.split(', ')))
    diff_set = set(succ_list) - set(df[current_name].tolist()) - set(['None'])
    G = {**dict(zip(df[current_name].tolist(), sub_child)), **dict([(diff, {}) for diff in diff_set])}
    return G


def cleanNullTerms(d):
    for i, j in d.items():
        for k, v in j.items():
            if k == 'None': d[i] = {}
    return d


def topological_order(sources, children):
    sources = list(sources)
    order = []
    visited = set()
    while sources:
        root = sources.pop()
        if root in visited:
            continue
        for data, event in Traverse.depth_first(
                root, children, preorder=False, cycles=True
        ):
            if event == Traverse.CYCLE:
                raise CycleException(list(data))
            if event == Traverse.EXIT and data not in visited:
                order.append(data)
                visited.add(data)
    return reversed(order)


def topological(order, children, flow=None, *, choose=min):
    flow = flow or {}
    for node_id in order:
        if node_id not in flow:
            flow[node_id] = (0, set())
            node_distance = 0
        else:
            node_distance, _ = flow[node_id]
        for child_id, step_distance in children(node_id):
            child_distance = step_distance + node_distance
            if child_id not in flow:
                flow[child_id] = (child_distance, {node_id})
                continue
            candidate_distance, candidate_steps = flow[child_id]
            if child_distance == candidate_distance:
                candidate_steps.add(node_id)
                continue
            flow[child_id] = choose(
                flow[child_id], (child_distance, {node_id}), key=lambda x: x[0]
            )
    return flow


def sub_graph (id_, project_df, foundation_df, successor):
    succ_topo = {}
    sub_project = project_df.loc[project_df['Project_ID']== id_,]
    sub_foundation = foundation_df.loc[foundation_df['Project_ID']== id_,]
    if sub_foundation.shape[0] > 3:
        succ_sub_G = cleanNullTerms (childern (sub_project, successor, 'Clean_task_code'))
        foundation_list = sub_foundation['Clean_task_code'].tolist()
        for j in foundation_list:
            topo1 = {j: topological(topological_order([j], lambda n: succ_sub_G[n].keys()), lambda n: succ_sub_G[n].items())}
            succ_topo.update(topo1)
    curr, succ, loc = [], [], []
    for i, j in succ_topo.items():
        for m, n in j.items():
            curr.append (i), succ.append (m), loc.append (n[0])
    succ_df = pd.DataFrame()
    succ_df['Current'], succ_df[successor], succ_df['Location'] = curr, succ, loc
    succ_df = succ_df.loc[succ_df['Location'] != 0.0, ['Current', successor]]
    return succ_topo, succ_df


class Traverse(enum.Enum):
    ENTRY = 1
    EXIT = 2
    CYCLE = 3
    LEAF = 4
    def depth_first(root, children, *,
                    preorder=True,
                    postorder=True,
                    cycles=False,
                    leaves=False,
                    repeat=False
                    ):
        ancestors = []
        visited = {}
        to_visit = [(root, Traverse.ENTRY)]
        while to_visit:
            (visiting, event) = to_visit.pop()
            if event == Traverse.EXIT:
                ancestors.pop()
                visited[visiting] = False
                if postorder: 
                    yield visiting, event
                continue
            is_ancestor = visited.get(visiting)
            if not is_ancestor is None:
                if is_ancestor and cycles:
                    cycle_start = len(ancestors)
                    for i, ancestor in enumerate(reversed(ancestors)):
                        if ancestor == visiting:
                            cycle_start = len(ancestors) - (i + 1)
                            break
                    yield ancestors[cycle_start:], Traverse.CYCLE
                if not repeat:
                    continue
            if event == Traverse.ENTRY:
                ancestors.append(visiting)
                visited[visiting] = True
                if preorder: 
                    yield visiting, event
                en_route = list(zip(children(visiting), itertools.repeat(Traverse.ENTRY)))
                if leaves and not en_route:
                    yield iter(ancestors), Traverse.LEAF
                to_visit.append((visiting, Traverse.EXIT))
                to_visit.extend(en_route)



def adjacency_matrix (id_, project_df, foundation_df):
    succ_topo, succ_df = sub_graph (id_, project_df, foundation_df, 'Successor')
    pred_topo, pred_df = sub_graph (id_, project_df, foundation_df, 'Predecessor')
    pred_df = pred_df.rename(columns={"Current": "Successor", "Predecessor": "Current"})
    total_df = pred_df.append (succ_df)
    diff = list (set(total_df['Successor'].tolist()) - set(total_df['Current'].tolist()))
    total_ = total_df.append (pd.DataFrame (diff, columns =['Current']))
    task = total_.drop_duplicates (subset = ['Current'])['Current'].tolist()
    order = (np.arange(0, len(task))).tolist()
    curr_loc, succ_loc = [], []
    for i, j in zip(task, order):
        for m in total_df['Current']:
            if i == m: curr_loc.append (j)
        for n in total_df['Successor']:
            if i ==n: succ_loc.append (j)
    adjacency_m = np.zeros((len(task), len(task)))
    for k, v in zip(succ_loc, curr_loc):
        np.put(adjacency_m, k*len(task)+v, 1)
    np.fill_diagonal(adjacency_m, 1)   
    return task, adjacency_m


def check_matrix (matrix):
    unique, counts = np.unique(matrix, return_counts=True)
    dict1 = dict(zip(unique, counts))
    if dict1[1.0] > matrix.shape[0]:
        return 'Clean'


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


def feature_input (sub_project, id_, trained_model):
    empty = np.zeros(60).reshape(1,60)
    ls = sub_project['Clean_task_name'].tolist()
    ls_token = token(prep(ls))
    length_curr = sub_project['r_length'].replace('None',0).tolist()
    pos_curr  = sub_project['r_pos'].replace('None',0).tolist()
    sub_project['Duration'] = sub_project['Duration'].astype(int)
    a = sub_project['Duration'].to_numpy()
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


id_list_ = [140297, 121395, 131325, 142884, 96831, 3148, 6621, 4019, 4459, 5046, 41214, 75020, 101597, 119044, 118994, 118944, 100785, 93867, 93917, 34143, 143451, 99673]
project = pd.read_csv ('/Users/ying/Name_text/paper/full_short_.csv')
foundation = project.head(0)
for i in ['Mat', 'Pile', 'Foundation', 'Shaft', 'Lift pit', 'Pad']:
    foundation = foundation.append(project.loc[project['Foundation']==i, ])


train_name = project['Clean_task_name'].tolist()
train_name_ = token(prep(train_name))
trained_model = FastText(train_name_, size = 60, min_count = 1, negative = 7, workers = 5)


init = np.zeros((1,1))
f_init = np.zeros((1,63))
for i in id_list_:
    actv_ls, a_matrix = adjacency_matrix (i, project, foundation)
    sub_project = project.head(0)
    for j in actv_ls:
        sub_project = sub_project.append(project.loc[(project['Project_ID']==i)&(project['Clean_task_code']==j), ])
        A = init.shape[0]
        B = a_matrix.shape[0]
        init = np.block([[init,np.zeros((A,B))],[np.zeros((B,A)), a_matrix]])
        feature = feature_input(sub_project, i, trained_model)
        f_init = np.append (f_init, feature, axis=0)

        
init_ = np.delete(init, 0, 0)
init_f = np.delete(init_, 0, 1)
degree_m = degree_matrix (init_f)
normal_m = normal_matrix (degree_m, init_f)
feature_m = np.delete(f_init, 0, 0)
train_in = np.dot (normal_m, feature_m)


batch_size = [20, 40, 60, 70, 80]
epochs = [60, 70, 80, 90]
optimizer = ['SGD']
neurons = [70,100,120]


def create_model(optimizer='sgd', neurons = 20): 
    gcn_model = models.Sequential()
    gcn_model.add(layers.Dense(neurons, activation='relu', input_dim = 63))
    gcn_model.add(layers.Dense(neurons, activation='relu'))
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

test_nodup = test_.drop_duplicates(subset = ('Clean_task_code', 'Project_ID'))
le = preprocessing.LabelEncoder()
label = le.fit_transform(test_nodup['Foundation_detail']).reshape(len(test_nodup),1)
grid_result = grid_1.fit(train_in, label)
best = grid_result.best_estimator_
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print (classification_report(label_x, best.predict(train_in)))
print (confusion_matrix(label_x, best.predict(train_in)))