
pd_lib=False

print("...Start Imports")

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
import keras.backend as K
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pycm import *
from tensorflow.keras.layers import Dense, Add, concatenate , Subtract, Activation , average,multiply,add
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import xlrd
print("...End Imports")

def transfo_y(y_train):
    y_train_multi=[]
    for i in range(len(y_train)):
        if y_train[i]==0:
            y_train_multi.append([1,0,0,0])
        elif y_train[i]==1:
            y_train_multi.append([0,1,0,0])
        elif y_train[i]==2:
            y_train_multi.append([0,0,1,0])
        elif y_train[i]==3:
            y_train_multi.append([0,0,0,1])
    return y_train_multi


print("...Start Embedding")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("...Embedding loaded")


config = pd.read_csv("../Data/Linkedin/train.txt",sep='\t')
turn1=[]
turn2=[]
turn3=[]
classes=[]
for row in config.itertuples():
    turn1.append(row.turn1)
    turn2.append(row.turn2)
    turn3.append(row.turn3)
    if row.label=='angry':
        classes.append(0)
    elif row.label=='happy':
        classes.append(1)
    elif row.label=='sad':
        classes.append(2)
    else:
        classes.append(3)
classes=transfo_y(classes)


turn1=list(embed(turn1))
turn2=list(embed(turn2))
turn3=list(embed(turn3))


text_embedded=[]
for i in range(len(turn1)):
    m=[]
    for j in range(len(turn1[i])):
        m.append(float(turn1[i][j]))
    for j in range(len(turn2[i])):
        m.append(float(turn2[i][j]))
    for j in range(len(turn3[i])):
        m.append(float(turn3[i][j]))
    text_embedded.append(m)

list_cat=['angry','happy','sad','others']

#data split for basic NN, logistic regression
xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded), classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)


print("...Data is ready!")