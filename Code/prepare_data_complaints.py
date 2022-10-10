import csv
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pycm import *
from tensorflow.keras.layers import Dense, Add, concatenate , Subtract, Activation , average,multiply,add
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import xlrd

import math as mth
from tensorflow.keras import backend as K
#from tensorflow.keras.utils.generic_utils import get_custom_objects
import scipy.spatial as sp


print('Imports termines')
#list_cat=['COH','HYPER','MERO','RANDOM','SYN']

#On lit les données
link_data='../Data/Complaints/Consumer_Complaints.csv'

df_all=pd.read_csv(link_data, index_col=[0])

df_all=df_all.dropna(subset=['Consumer Complaint'])
df_all=df_all[df_all.Product!='Virtual currency']
df_all=df_all[df_all.Product!='Other financial service']

list_cat=list(set(df_all['Product']))
classes=df_all['Product'].values
text=df_all['Consumer Complaint'].values


def transfo_y(y_train):
    y_train_multi=[]
    for i in range(len(y_train)):
        ind=list_cat.index(y_train[i])
        list_o=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        list_o[ind]=1
        y_train_multi.append(list_o)
    return y_train_multi

classes=transfo_y(classes)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print('embedding chargé')

print('on embed le texte')
nb=55
text_embedded=[]
for i in range(nb):
    #print(i)
    text_embedded=text_embedded+list(embed(text[i*5000:(i+1)*5000]))

classes=classes[:nb*5000]

#data split for basic NN, logistic regression
xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded), classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)

