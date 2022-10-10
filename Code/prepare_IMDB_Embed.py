
only_phrase=True

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
from sentence_transformers import SentenceTransformer

print("...End Imports")
def transfo_y(y_train):
    y_train_multi=[]
    for i in range(len(y_train)):
        if y_train[i]==0:
            y_train_multi.append([1,0,0,0,0])
        elif y_train[i]==1:
            y_train_multi.append([0,1,0,0,0])
        elif y_train[i]==2:
            y_train_multi.append([0,0,1,0,0])
        elif y_train[i]==3:
            y_train_multi.append([0,0,0,1,0])
        else:
            y_train_multi.append([0,0,0,0,1])
    return y_train_multi

# openning the file

df=pd.read_csv("../Data/IMDB/train.tsv",sep="\t")
def prepare_data(df):
    print("...Start Embedding")
    embed1 = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embed2 = SentenceTransformer('paraphrase-distilroberta-base-v1')
    print("...Embedding loaded")
    text_embedded=[]
    classes=[]
    if only_phrase:
        i=1
        for row in df.itertuples():
            if row.SentenceId==i:
                text_embedded.append(row.Phrase)
                classes.append(row.Sentiment)
                i+=1
    else:
        text_embedded=df["Phrase"]
        classes=df["Sentiment"]
    text_embedded1=list(embed1(text_embedded))
    text_embedded2=list(embed2.encode(text_embedded))
    classes=transfo_y(classes)
    print("...Embedding ended")

    print("len classes ",len(classes))
    print("len text ", len(text_embedded))



    




    print("...Data is ready!")

    #data split for basic NN, logistic regression
    xtrain1, xtest1, ytrain1, ytest1 = train_test_split(list(text_embedded1), classes,stratify=classes, test_size=0.4, random_state=0)
    xtrain1, xvalid1, ytrain1, yvalid1 = train_test_split(xtrain1, ytrain1, stratify=ytrain1,test_size=0.3, random_state=0)
    xtrain2, xtest2, ytrain2, ytest2 = train_test_split(list(text_embedded2), classes,stratify=classes, test_size=0.4, random_state=0)
    xtrain2, xvalid2, ytrain2, yvalid2 = train_test_split(xtrain2, ytrain2, stratify=ytrain2,test_size=0.3, random_state=0)

    xtrain1=np.array(xtrain1)
    xtest1=np.array(xtest1)
    xvalid1=np.array(xvalid1)
    ytrain1=np.array(ytrain1)
    ytest1=np.array(ytest1)
    yvalid1=np.array(yvalid1)


    xtrain2=np.array(xtrain2)
    xtest2=np.array(xtest2)
    xvalid2=np.array(xvalid2)
    ytrain2=np.array(ytrain2)
    ytest2=np.array(ytest2)
    yvalid2=np.array(yvalid2)





    data = {}
    data["Emb1"]={"xtrain":xtrain1,"ytrain":ytrain1,"xtest":xtest1,"ytest":ytest1,"xvalid":xvalid1,"yvalid":yvalid1}
    data["Emb2"]={"xtrain":xtrain2,"ytrain":ytrain2,"xtest":xtest2,"ytest":ytest2,"xvalid":xvalid2,"yvalid":yvalid2}
    print("...Data is ready!")
    return data

list_cat=['negative','somewhat negative','neutral','somewhat positive','positive']
data=prepare_data(df)
