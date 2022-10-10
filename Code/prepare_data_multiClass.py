from sklearn.utils import shuffle
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#
import tensorflow.keras
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Add, concatenate , Subtract, Activation , average,multiply,add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.dummy import DummyClassifier
import json
import numpy as np
import os
import sys

from keras import backend as K

np.random.seed(44)
#import tensorflow.compat.v1 as tf
#print('update tensorflow')
#tf.disable_v2_behavior()
#from tensorflow.keras.backend.tensorflow_backend import set_session

import math as mth
from tensorflow.keras import backend as K
#from tensorflow.keras.utils.generic_utils import get_custom_objects
import scipy.spatial as sp
import pandas as pd
from pycm import *

print('Imports termines')
list_cat=['COH','HYPER','MERO','RANDOM','SYN']

#On lit les données
link_data='../Data/data_multi_class.csv'

df_all=pd.read_csv(link_data, index_col=[0])




def load_embeddings(path, dimension):
    f = open(path, encoding="utf8").read().splitlines()
    vectors = {}
    for i in f:
        elems = i.split()
        vectors[" ".join(elems[:-dimension])] =  np.array(elems[-dimension:]).astype(float)
    return vectors



embeddings = load_embeddings("../Data/Embeddings/glove.6B.300d.txt", 300)

print('embeddings charges')
words_ = sorted(list(set(df_all.w1.values.tolist() + df_all.w2.values.tolist())))

words_train, words_test =train_test_split(words_, test_size=0.4)
   
df_all["known_words"] = df_all.apply(lambda l: l["w1"] in embeddings and l["w2"] in embeddings, axis =1  )
    
    
    
    
df_all["is_train"] = df_all.apply(lambda l : l["w1"] in words_train and l["w2"] in words_train and l["known_words"] == True, axis=1 )
df_all["is_test"] = df_all.apply(lambda l : l["w1"] in words_test and l["w2"] in words_test and l["known_words"] == True, axis=1)



def get_vector_representation_of_word_pairs(dataframe, embeddings_voca):
    x1 = [embeddings_voca[word] for word in dataframe.w1.values]
    x2 =[embeddings_voca[word] for word in dataframe.w2.values]
    y = dataframe[['CAT_COH','CAT_HYPER','CAT_MERO','CAT_RANDOM','CAT_SYN']].values
    x = np.hstack((x1, x2))
    

    return x, y



xtrain, ytrain = get_vector_representation_of_word_pairs(df_all.loc[df_all.is_train==True], embeddings)
xtest, ytest   = get_vector_representation_of_word_pairs(df_all.loc[df_all.is_test==True], embeddings)


xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,  test_size=0.30, random_state=1234,)


print('data préparee')