import pandas as pd
import numpy as np
import itertools
import collections
import enum
import heapq
import json
from itertools import chain
from numpy import load, save
import h5py
import re
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from gensim import corpora
import pickle
import scipy



####Tagging
taggers = pd.read_csv ('data_yh/ontology.csv')
project = pd.read_csv ('data_yh/full_dataset.csv')

level3 = taggers['Element_level3'].tolist()
level4 = taggers['Element_level4'].tolist()
level4_token = token(prep(level4))
level3_token = token(prep(level3))

sim_tag = []
for i, j in zip (level3_token, level4_token):
    sim_tag.append (list(zip(j, i)))

tag_ = UnigramTagger(sim_tag)
test = token(prep(project['Task_name'].tolist()))

tagged_actv = []
for i in test:
    tagged_actv.append (tag_.tag(i))

tagged_ls = []
for i in range (len(tagged_actv)):
    a = []
    for j in range (len(tagged_actv[i])):
        a.append(str(tagged_actv[i][j][1]))
    tagged_ls.append(' '.join(a))
project['All_tagged'] = tagged_ls
project["All_tagged"] = project["All_tagged"].str.replace ('None', '').str.strip(' ')
project.to_csv ('data_yh/full_dataset.csv')

####GCN input
def chainer(s):
    return list(chain.from_iterable(s.str.split(', ')))

def create_graph_edge (project, project_id_):
    lens = project['Successor'].str.split(', ').map(len)
    expanded_df = pd.DataFrame({'Current': np.repeat(project['Current'], lens), 
                                'Project_ID': np.repeat(project['Project_ID'], lens),
                                'Successor': chainer (project['Successor']), 
                                'Logic': chainer (project['Succ_logic']), 'Lag': chainer (project['Succ_lag'])})
    expanded_df['Successor_ID'] = expanded_df['Successor'].astype(str) +"_"+ expanded_df['Project_ID'].astype(str)

    total_list = expanded_df['Current'].tolist()
    total_list.extend(expanded_df['Successor_ID'].tolist())
    actv_list = list(dict.fromkeys(total_list))
    actv_id = list(range (0, len(actv_list)))

    expanded_df['Current_num'], expanded_df['Successor_num'] = "", ""
    
    for i, j in zip(actv_list, actv_id):
        expanded_df.loc[expanded_df.Current == i, 'Current_num'] = j
        expanded_df.loc[expanded_df.Successor_ID == i, 'Successor_num'] = j
        
    expanded_df['Current_num'] = expanded_df['Current_num'].astype(str)
    expanded_df['Successor_num'] = expanded_df['Successor_num'].astype(str)   
    expanded_df['Logic'] = (expanded_df['Logic'].astype(str)).str.strip()
    expanded_df['Current']= expanded_df['Current'].str.strip()
    expanded_df['Lag'] = expanded_df['Lag'].str.strip().replace ('None', 0)    
    expanded_df = expanded_df.assign (Weight_lag = lambda x: np.nan_to_num(np.log(x['Lag'].astype(float)/8+1)))   
    m = expanded_df.head(0)
    
    for i in project_id_:
        sub = expanded_df.loc[expanded_df['Project_ID']==int(i),]
        x = lambda name: sub['Logic'].value_counts()[name]/len(sub)
        sub['Weight_logic'] = (x(sub['Logic'].tolist())).to_frame()['Logic'].tolist()
        m = m.append(sub)
    m['Combined_edge'] = 0.5*m['Weight_lag'] + 0.5*m['Weight_logic'] 
    return m, actv_list

def cbow (ls, model):
    x, cbow = np.zeros((1,60)), np.zeros((1,60))
    ls1 = []
    for i in range(len(ls)):
        m = ls[i]
        for word in m:
            vector = model.wv[word].reshape(1, 60)
            x = np.append (x, vector, axis = 0)
        cbow_ = (np.sum(x, axis = 0)).reshape(1,60)
        cbow = np.append (cbow, cbow_, axis = 0)
    cbow_m = np.delete(cbow, 0, axis = 0)   
    return cbow_m

def token (ls):
    ps = PorterStemmer()
    token_word = []      
    for i in range (len(ls)):
        ls1 = word_tokenize(str(ls[i]))
        ls2 = []
        for word in ls1:
            b = re.sub(r"\b[a-zA-Z]\b", "", ps.stem (word.lower()))
            ls2.append (re.sub(r'[^\w]', '', b))
        token_word.append ([x for x in ls2 if x])
    return token_word

def feature (full, actv_list, trained_model):
    empty = np.zeros(123).reshape(1,123)
    label_ = []
    for i in range (len(actv_list)):
        y = lambda name: np.asarray(df_[name].fillna(0).tolist())
        df_ = full.loc[(full['Current'] == actv_list[i]), ]
        if df_.shape[0] != 0:
            length = y('r_length')
            pos = y('r_pos')
            duration = np.log(y('Duration')/8 + 1)
            task_token = token(df_['Task_name'].tolist())
            wbs_token = token(df_['WBS'].tolist())
            m = cbow (task_token, trained_model)
            n = cbow (wbs_token, trained_model)
            other_feature_2 = np.append (duration.reshape(1,1), np.append (length.reshape(1,1), pos.reshape(1,1), axis = 1), axis = 1)
            empty = np.append (empty, np.append (np.append(m, n, axis = 1), other_feature_2, axis = 1), axis = 0)
            label_.extend (df_['2021_label'].tolist())
        if df_.shape[0] == 0:
            empty = np.append (empty, np.zeros(123).reshape(1,123), axis = 0)
            label_.extend ([np.nan])
        empty_ = np.delete(empty, 0, axis = 0)        
    return empty_, label_


trained_model60 = FastText.load ('pretrained_model/60_model')

train_id_ = [4019, 4459, 5046, 41214, 75020, 140297, 99673, 121395, 3148]
test_id_ = [101597, 96831, 118994, 6621]
id_ = [4019, 4459, 5046, 41214, 75020, 140297, 99673, 121395, 3148, 101597, 96831, 118994, 6621]
train_project = project[project['Project_ID'].isin(train_id_)].drop_duplicates (subset = ['Current'])
train_df, train_actv = expand (train_project, train_id_)
train_matrix, train_label = feature(project, train_actv, trained_model60)

test_project = project[project['Project_ID'].isin(test_id_)].drop_duplicates (subset = ['Current'])
test_df, test_actv = expand (test_project, test_id_)
test_matrix, test_label = feature(project, test_actv, trained_model60)


le = LabelEncoder()
enc = OneHotEncoder()
train_label.extend (test_label)
label_enc = enc.fit_transform(le.fit_transform(np.asarray(train_label)).reshape(len(train_label),1)).toarray()

train_label_onehot = label_enc[0:8677]
test_label_onehot = label_enc[8677:12980]
pickle.dump(train_label_onehot, open("data_yh/ind.new_label2d.ally","wb"))
pickle.dump(test_label_onehot, open("data_yh/ind.new_label2d.ty","wb"))


train_df = pd.DataFrame(output_format(train_matrix))
train_df['label'] = label_[0:8677]
train_df = pd.concat ([train_df, pd.DataFrame(train_label)], axis = 1)
transductive_df = train_df.loc[train_df['label']!= 'nan']

x = scipy.sparse.csr_matrix(transductive_df.iloc[:, 0:123].values)
y = transductive_df.iloc[:,124:133].to_numpy()
pickle.dump(y, open("data_yh/ind.new_label.y","wb"))
pickle.dump(x, open("data_yh/ind.new_label.x","wb"))

allx = scipy.sparse.csr_matrix(output_format(train_matrix).astype(float))
tx = scipy.sparse.csr_matrix(output_format(test_matrix).astype(float))
pickle.dump(tx, open("data_yh/ind.new_label.tx","wb"))
pickle.dump(allx, open("data_yh/ind.new_label.allx","wb"))


scaler = MinMaxScaler()
tsne_model_en_2d = TSNE(perplexity=30, n_components=3, n_iter=3500, random_state=32)
x_2d = tsne_model_en_2d.fit_transform(x.toarray())
tx_2d = tsne_model_en_2d.fit_transform(tx.toarray())
allx_2d = tsne_model_en_2d.fit_transform(allx.toarray())
pickle.dump(scipy.sparse.csr_matrix(x_2d), open("data_yh/ind.new_label2d.x","wb"))
pickle.dump(scipy.sparse.csr_matrix(tx_2d), open("data_yh/ind.new_label2d.tx","wb"))
pickle.dump(scipy.sparse.csr_matrix(allx_2d), open("data_yh/ind.new_label2d.allx","wb"))


total = project[project['Project_ID'].isin(id_)].drop_duplicates (subset = ['Current'])
total_df, total_actv = create_graph_edge (total, id_)
a = np.asarray(total_df['Current_num'].to_numpy(), dtype=np.int, order='C')
b = np.asarray(total_df['Successor_num'].to_numpy(), dtype=np.int, order='C')
c = np.asarray(total_df['Combined_edge'].to_numpy(), dtype=np.int, order='C')
elist = zip(a, b, c)
pickle.dump(elist, open("data_yh/ind.new_label.graph","wb"))


######
feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
a = sess.run([model.outputs, model.loss, model.accuracy], feed_dict=feed_dict_val)
y_pred = a[0] #graph embedding

import itertools
import enum

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
        if len(flow.keys()) < 3:del flow[list(flow.keys())[0]]
    return flow


def sub_graph (project, successor):
    succ_topo = {}
    succ_sub_G = cleanNullTerms (childern (project, successor, 'Clean_task_code'))
    foundation_list = project['Clean_task_code'].tolist()
    for j in foundation_list:
        topo1 = {j: topological(topological_order([j], lambda n: succ_sub_G[n].keys()), lambda n: succ_sub_G[n].items())}
        succ_topo.update(topo1)
    return succ_topo

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

def replace_text (df, my_dict):
    for i in range (len(df['Clean_task_code'])):
        for j in list(my_dict.keys()):
            if df['Clean_task_code'][i] == j:
                new_key = df['Task_name'][i]
                my_dict[new_key] = my_dict.pop(j)
    return my_dict
