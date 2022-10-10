print("...Start Imports")
from joblib import Parallel, delayed
import glob
import csv
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import xlrd
import os
from xlwt import Workbook
from itertools import product 
from pycm import *
from sklearn.metrics import mean_absolute_error
import glob
print("...End Imports")
print("...Loading embedding")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print('...Embedding loaded')
link_business="../Data/business" # business class=-2
link_entertainment="../Data/entertainment" #entertainment class=-1
link_politics="../Data/politics" # politics class=0
link_sport="../Data/sport" # sport class=1
link_tech="../Data/tech" # tech class=2
text_to_embed=[]
classes=[]

 
path = link_business
files = os.listdir(path)
for i in files:
    f=open(link_business+"/"+i,'r', encoding='unicode_escape')
    t=f.read()
    text_to_embed.append(t)
    f.close()
    classes.append([1,0,0,0,0])
    
path = link_entertainment
files = os.listdir(path)
for i in files:
    f=open(link_entertainment+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,1,0,0,0])
    
path = link_politics
files = os.listdir(path)
for i in files:
    f=open(link_politics+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,1,0,0])
    
path = link_sport
files = os.listdir(path)
for i in files:
    f=open(link_sport+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,0,1,0])
    
path = link_tech
files = os.listdir(path)
for i in files:
    f=open(link_tech+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,0,0,1])

print("...Start Embedding")

text_embedded=embed(text_to_embed)

print("...Text embedded succesfully")

print("...Spliting the data")
list_cat=['Business','Ent','POL','Sport','TECH']
xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded), classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)

print("...Data is ready!")
