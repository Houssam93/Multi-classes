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


#metrics 
def accuracy(CM):
    diag=0
    total=0
    for i in range(len(CM)):
        for j in range(len(CM)):
            total+=CM[i][j]
            if i==j:
                diag+=CM[i][j]
    return diag/total
def precision_class(CM,i):
    i=i-1
    sum_ligne=0
    for j in CM[i]:
        sum_ligne+=j
    return CM[i][i]/sum_ligne
def precision_macro(CM):
    summ=0
    for i in range(len(CM)):
        summ+=precision_class(CM,i+1)
    return summ/len(CM)
def recall_class(CM,i):
    i=i-1
    sum_col=0
    for j in range(len(CM)):
        sum_col+=CM[j][i]
    return CM[i][i]/sum_col
def recall_macro(CM):
    summ=0
    for i in range(len(CM)):
        summ+=recall_class(CM,i+1)
    return round(summ/len(CM),4)
def F1_score_class(CM,i):
    return (2*recall_class(CM,i)*precision_class(CM,i))/(recall_class(CM,i)+precision_class(CM,i))    
def F1_score_macro(CM):
    return (2*recall_macro(CM)*precision_macro(CM))/(recall_macro(CM)+precision_macro(CM))



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







embed=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("embedding chargé")



text_embedded=embed(text)




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

list_cat=['Neg','Neutral','Pos']
print('on fait le split')
#data split for basic NN, logistic regression
xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded), classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)

print('split terminé')

