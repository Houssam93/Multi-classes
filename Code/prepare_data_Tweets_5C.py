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
import random


print("...End Imports")
print("...Loading embedding")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print('...Embedding loaded')


file_=open("../Data/Tweets_5C.txt","r")
file=file_.readlines()
file_.close()



classes_train=[]
Tweets_train=[]



for i in range(len(file)):
        liste=file[i].split("\t")

        classes_train.append(int(liste[2]))
        Tweets_train.append(liste[3])


classes=[]
Tweets=[]

for i in range(len(classes_train)):
    if classes_train[i]==-2 :
        Tweets.append(Tweets_train[i])
        classes.append([1,0,0,0,0])
for i in range(len(classes_train)):
    if classes_train[i]==2:
        Tweets.append(Tweets_train[i])
        classes.append([0,0,0,0,1])
for i in range(len(classes_train)):
    if classes_train[i]==0:
        prob=random.randint(1,5)
        if prob==1:
            Tweets.append(Tweets_train[i])
            classes.append([0,0,1,0,0])
    if classes_train[i]==1:
        prob=random.randint(1,5)
        if prob==1:
            Tweets.append(Tweets_train[i])
            classes.append([0,0,0,1,0])
for i in range(len(classes_train)):
    if classes_train[i]==-1:
        prob=random.randint(1,2)
        if prob==1:
            Tweets.append(Tweets_train[i])
            classes.append([0,1,0,0,0])


d={}
for i in range(len(classes)):
    d[Tweets[i]]=classes[i]

keys=[]
for key in d:
    if d[key]==[1,0,0,0,0]:
        keys.append(key)
        keys.append(key)
    keys.append(key)
random.shuffle(keys)
Tweets_=[]
classes_=[]
for i in keys:
    Tweets_.append(i)
    classes_.append(d[i])

classes=classes_
Tweets=Tweets_


text_embedded=list(embed(Tweets))

print("...Spliting the data")

xtrain, xtest, ytrain, ytest = train_test_split(text_embedded, classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)

xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)
xtest=np.array(xtest)
ytest=np.array(ytest)

print("...Data is ready!")

list_cat=['Very_Neg','Neg','Neutral','Pos','Very_Pos']
