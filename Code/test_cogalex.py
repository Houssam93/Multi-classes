
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
import pandas as pd
import os
import sys

from keras import backend as K

np.random.seed(44)

link1="../Data/CogALexV_train_v1/gold_task2.txt"
link2="../Data/CogALexV_test_v1/gold_task2.txt"
dff1 = pd.read_csv(link1,header=None,sep = '\t')
dff2 = pd.read_csv(link2,header=None,sep = '\t')
dff1.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
dff2.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
dff3=pd.concat([dff1,dff2])


df_all = pd.concat([dff3,pd.get_dummies(dff3['Category'], prefix='Cat')],axis=1)

df_all.drop(['Category'],axis=1, inplace=True)

def three_data_aux(name_data):
    Rumen,Root9,Bless,Cogalex,Weeds=False,False,False,False,False,
    if name_data == 'Rumen':
        Rumen=True
    if name_data == 'Root9':
        Root9=True
    if name_data == 'Bless':
        Bless=True
    if name_data == 'Cogalex':
        Cogalex=True
    if name_data == 'Weeds':
        Weeds=True
    

    if Weeds:
        Rumen=True
        task1Weeds="HYPER"
        task2Weeds="COO"
        link2="../Data/coordpairs2_wiki100.json"
        link1="../Data/entpairs2_wiki100.json"
    if Cogalex :
        task1="HYPER"
        task2="SYN"
        link1="../Data/CogALexV_train_v1/gold_task2.txt"
        link2="../Data/CogALexV_test_v1/gold_task2.txt"
    if Rumen :
        task1="HYPER"
        task2="SYN"
        link="../Data/RUMEN/RumenPairs.txt"
    if Root9 :
        link_hyper="../Data/ROOT9/ROOT9_hyper.txt"
        link_coord="../Data/ROOT9/ROOT9_coord.txt"
        link_random="../Data/ROOT9/ROOT9_random.txt"
        task1= "HYPER"
        task2= "COORD"
    if Bless :
        task1= "HYPER"
        task2= "MERO"
        link_coord="../Data/BLESS/BLESS_mero.txt"
        link_hyper="../Data/BLESS/BLESS_hyper.txt"
        link_random="../Data/BLESS/BLESS_random.txt"

    def get_names(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task1
        if cat == 2: return task2
    def get_names_Weeds1(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task1Weeds
    def get_names_Weeds2(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task2Weeds




    if Rumen :
        dff = pd.read_csv(link)
        dff.rename(columns={"W1":"w1", "W2":"w2","rel":"Category"}, inplace=True)
        dff["Category"] = dff["Category"].apply(get_names)
        df = dff.loc[dff.Category == task2]
        df2 = dff.loc[dff.Category == task1]
        df3 = dff.loc[dff.Category == "RANDOM"]
        #print(len(df),len(df2),len(df3))
    if Root9 or Bless:

        df = pd.read_csv(link_coord,header=None,sep = '\t')
        df.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df=prep_df(df)
        df2 = pd.read_csv(link_hyper,header=None,sep = '\t')
        df2.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df2=prep_df(df2)
        df3 = pd.read_csv(link_random,header=None,sep = '\t')
        df3.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df3=prep_df(df3)
    if Cogalex:

        dff1 = pd.read_csv(link1,header=None,sep = '\t')
        dff2 = pd.read_csv(link2,header=None,sep = '\t')
        dff1.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff2.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff3=pd.concat([dff1,dff2])
        #print(list(set(dff3.Category.values.tolist())))
        df = dff3.loc[dff3.Category == task2]
        df2 = dff3.loc[dff3.Category == task1]
        df3 = dff3.loc[dff3.Category == "RANDOM"]
    if Weeds :
        json_data=open(link1).read()
        data = json.loads(json_data)
        dff=pd.DataFrame(data)
        dff.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff["Category"] = dff["Category"].apply(get_names_Weeds1)
        df2 = dff.loc[dff.Category == task1Weeds]
        #df3 = dff.loc[dff.Category == "RANDOM"]
        #df3=df3[0:len(df2)]
        #print("taille 0,1 pour entpairs",len(df2),len(df3))
        
        json_data2=open(link2).read()
        data2 = json.loads(json_data2)
        dff2=pd.DataFrame(data2)
        dff2.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff2["Category"] = dff2["Category"].apply(get_names_Weeds2)
        df = dff2.loc[dff2.Category == task2Weeds]
        #df=df[0:len(df2)]
        #print("taille 0,1 pour coord",len(df))
        
        
    return df,df2,df3
