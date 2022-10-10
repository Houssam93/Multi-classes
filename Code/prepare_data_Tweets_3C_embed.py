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
from sentence_transformers import SentenceTransformer




embed1=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed2 = SentenceTransformer('paraphrase-distilroberta-base-v1')
print("embedding chargé")
#data split for basic NN, logistic regression
def split_func(text_embedded_aux,classes):
    xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded_aux), classes,stratify=classes, test_size=0.4, random_state=0)
    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
    xtrain=np.array(xtrain)
    xtest=np.array(xtest)
    xvalid=np.array(xvalid)
    ytrain=np.array(ytrain)
    yvalid=np.array(yvalid)
    ytest=np.array(ytest)
    return xtrain, ytrain, xtest,  ytest , xvalid ,yvalid
#metrics 



def make_data():
    Tweets=pd.read_csv('../Data/Tweets_3C.csv', sep=',', index_col=0)


    text=Tweets['text'].values
    airline_sentiment=Tweets['airline_sentiment'].values
    airline_sentiment_confidence=Tweets['airline_sentiment_confidence'].values
    negativereason=Tweets['negativereason'].values
    negativereason_confidence=Tweets['negativereason_confidence'].values
    airline=Tweets['airline'].values
    airline_sentiment_gold=Tweets['airline_sentiment_gold'].values
    name=Tweets['name'].values
    negativereason_gold=Tweets['negativereason_gold'].values
    retweet_count=Tweets['retweet_count'].values
    tweet_coord=Tweets['tweet_coord'].values
    tweet_created=Tweets['tweet_created'].values
    tweet_location=Tweets['tweet_location'].values
    user_timezone=Tweets['user_timezone'].values



    text_embedded1=embed1(text)
    text_embedded2=embed2.encode(text)



    classes=[]
    # -1 négatif
    #0 neutre
    # 1 positif
    for i in airline_sentiment:
        if i=='neutral':
            classes.append([1,0,0])
        elif i=='positive':
            classes.append([0,1,0])
        else:
            classes.append([0,0,1])

    
    print('on fait le split')


    print('split terminé')

    xtrain1, ytrain1, xtest1,  ytest1 , xvalid1 ,yvalid1=split_func(text_embedded1,classes)
    xtrain2, ytrain2, xtest2,  ytest2 , xvalid2 ,yvalid2=split_func(text_embedded2,classes)
    data = {}

    data["Emb1"]={"xtrain":xtrain1,"ytrain":ytrain1,"xtest":xtest1,"ytest":ytest1,"xvalid":xvalid1,"yvalid":yvalid1}
    data["Emb2"]={"xtrain":xtrain2,"ytrain":ytrain2,"xtest":xtest2,"ytest":ytest2,"xvalid":xvalid2,"yvalid":yvalid2}
    print("...Data is ready!")
    return data

list_cat=['Neg','Neutral','Pos']
data=make_data()

