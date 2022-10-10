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
link_athletics="../Data/athletics" # business class=-2
link_cricket="../Data/cricket" #entertainment class=-1
link_football="../Data/football" # politics class=0
link_rugby="../Data/rugby" # sport class=1
link_tennis="../Data/tennis" # tech class=2
text_to_embed=[]
classes=[]

 
path = link_athletics
files = os.listdir(path)
for i in files:
    f=open(link_athletics+"/"+i,'r', encoding='unicode_escape')
    t=f.read()
    text_to_embed.append(t)
    f.close()
    classes.append([1,0,0,0,0])
    
path = link_cricket
files = os.listdir(path)
for i in files:
    f=open(link_cricket+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,1,0,0,0])
    
path = link_football
files = os.listdir(path)
for i in files:
    f=open(link_football+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,1,0,0])
    
path = link_rugby
files = os.listdir(path)
for i in files:
    f=open(link_rugby+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,0,1,0])
    
path = link_tennis
files = os.listdir(path)
for i in files:
    f=open(link_tennis+"/"+i,'r', encoding='unicode_escape')
    text_to_embed.append(f.read())
    f.close()
    classes.append([0,0,0,0,1])

print("...Start Embedding")

text_embedded=embed(text_to_embed)

print("...Text embedded succesfully")

print("...Spliting the data")


list_cat=['ATH','Cric','Foot','Rugby','TENN']
xtrain, xtest, ytrain, ytest = train_test_split(list(text_embedded), classes,stratify=classes, test_size=0.4, random_state=0)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, stratify=ytrain,test_size=0.3, random_state=0)
xtrain=np.array(xtrain)
xvalid=np.array(xvalid)
ytrain=np.array(ytrain)
yvalid=np.array(yvalid)

print("...Data is ready!")
