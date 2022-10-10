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
print('MAJ0')

list_cat=['COH','HYPER','MERO','RANDOM','SYN']

#On lit les données
link_data='../Data/data_multi_class.csv'

df_all=pd.read_csv(link_data, index_col=[0])

print(df_all.head())



def load_embeddings(path, dimension):
    f = open(path, encoding="utf8").read().splitlines()
    vectors = {}
    for i in f:
        elems = i.split()
        vectors[" ".join(elems[:-dimension])] =  np.array(elems[-dimension:]).astype(float)
    return vectors



embeddings = load_embeddings("../Data/Embeddings/glove.6B.300d.txt", 300)


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


def ModelBaseline_MultiClass(nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=600))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model






def ModelBaseline_OneVsRest(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(600,))

    x_coh=Dense(nb_n1, activation='sigmoid')(inputs)
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(600,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared2=Dense(nb_n1, activation='sigmoid')(inputs)
    

    
    ind_coh=concatenate([inputs,x_shared1])
    ind_hyper=concatenate([inputs,x_shared1])
    ind_mero=concatenate([inputs,x_shared1])
    ind_random=concatenate([inputs,x_shared1])
    ind_syn=concatenate([inputs,x_shared1])
    
    x_coh=Dense(nb_n2, activation='sigmoid')(ind_coh)
    x_hyper=Dense(nb_n2, activation='sigmoid')(ind_hyper)
    x_mero=Dense(nb_n2, activation='sigmoid')(ind_mero)
    x_random=Dense(nb_n2, activation='sigmoid')(ind_random)
    x_syn=Dense(nb_n2, activation='sigmoid')(ind_syn)
    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)
    out_random=Dense(1, activation='sigmoid')(x_random)
    out_syn=Dense(1, activation='sigmoid')(x_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model






def ModelOneVsRest_AllShared(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(600,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
   
    x_coh=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_hyper=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_mero=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_random=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_syn=Dense(nb_n2, activation='sigmoid')(x_shared1)
    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)
    out_random=Dense(1, activation='sigmoid')(x_random)
    out_syn=Dense(1, activation='sigmoid')(x_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model







def ModelOneVsRest_SharedPrivate_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(600,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper])
    ind_coh2=concatenate([inputs,x_shared_coh_mero])
    ind_coh3=concatenate([inputs,x_shared_coh_random])
    ind_coh4=concatenate([inputs,x_shared_coh_syn])
    
    
    ind_hyper1=concatenate([inputs,x_shared_hyper_mero])
    ind_hyper2=concatenate([inputs,x_shared_hyper_random])
    ind_hyper3=concatenate([inputs,x_shared_hyper_syn])
    ind_hyper4=concatenate([inputs,x_shared_coh_hyper])
    
    
    ind_mero1=concatenate([inputs,x_shared_mero_random])
    ind_mero2=concatenate([inputs,x_shared_mero_syn])
    ind_mero3=concatenate([inputs,x_shared_hyper_mero])
    ind_mero4=concatenate([inputs,x_shared_coh_mero])
    
    
    ind_random1=concatenate([inputs,x_shared_random_syn])
    ind_random2=concatenate([inputs,x_shared_coh_random])
    ind_random3=concatenate([inputs,x_shared_hyper_random])
    ind_random4=concatenate([inputs,x_shared_mero_random])
    
    ind_syn1=concatenate([inputs,x_shared_coh_syn])
    ind_syn2=concatenate([inputs,x_shared_hyper_syn])
    ind_syn3=concatenate([inputs,x_shared_mero_syn])
    ind_syn4=concatenate([inputs,x_shared_random_syn])
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper3)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def ModelOneVsRest_All_shared_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(600,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    
    ind_coh1=x_shared_coh_hyper
    ind_coh2=x_shared_coh_mero
    ind_coh3=x_shared_coh_random
    ind_coh4=x_shared_coh_syn
    
    
    ind_hyper1=x_shared_hyper_mero
    ind_hyper2=x_shared_hyper_random
    ind_hyper3=x_shared_hyper_syn
    ind_hyper4=x_shared_coh_hyper
    
    
    ind_mero1=x_shared_mero_random
    ind_mero2=x_shared_mero_syn
    ind_mero3=x_shared_hyper_mero
    ind_mero4=x_shared_coh_mero
    
    
    ind_random1=x_shared_random_syn
    ind_random2=x_shared_coh_random
    ind_random3=x_shared_hyper_random
    ind_random4=x_shared_mero_random
    
    ind_syn1=x_shared_coh_syn
    ind_syn2=x_shared_hyper_syn
    ind_syn3=x_shared_mero_syn
    ind_syn4=x_shared_random_syn
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper3)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model




def Model_softmax():
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(5,))


    out=Dense(5, activation='sigmoid')(inputs)
    
    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer='Adam', loss= ['categorical_crossentropy'])
    return model



def train_model_softmax(name_model,epochs):
    model=name_model
    
    model.fit(xtrain, [ytrain.transpose()[0],ytrain.transpose()[1],ytrain.transpose()[2]
                       ,ytrain.transpose()[3],ytrain.transpose()[4]],
                epochs=epochs, validation_data=(xvalid, [yvalid.transpose()[0],yvalid.transpose()[1],yvalid.transpose()[2]
                       ,yvalid.transpose()[3],yvalid.transpose()[4]]),  verbose=False, callbacks=[EarlyStopping(patience=5)])
    model_softmax=Model_softmax()
    pred_train=model.predict(xtrain, verbose=False)
    pred_valid=model.predict(xvalid,verbose=False)
    model_softmax.fit((np.array(list(map(list, zip(*pred_train))))[:,:,0]),ytrain, epochs=50,
                      verbose=False, callbacks=[EarlyStopping(patience=5)]
                     ,validation_data=(((np.array(list(map(list, zip(*pred_valid))))[:,:,0])),yvalid))
    return model,model_softmax


def train_model_multiClass(name_model,epochs):
    model=name_model
    model.fit(xtrain, ytrain,
                epochs=epochs, validation_data=(xvalid, yvalid),  verbose=False, callbacks=[EarlyStopping(patience=5)])
    return model

def func_preds_max_softmax(model,model_softmax,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_pre=[]

    preds_final=model_softmax.predict((np.array(list(map(list, zip(*preds))))[:,:,0]),verbose=False)
    for i in range(len(preds[0])):
        a=np.zeros(5)
        b=[]
        for j in range(5):
            b.append(preds[j][i])
        a[np.argmax(b)]=1
        preds_pre.append(a)
    preds_all_max=[list_cat[np.argmax(l)] for l in preds_pre]
    preds_all_softmax=[list_cat[np.argmax(l)] for l in preds_final]
    return preds_all_max,preds_all_softmax
    

def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all


def create_metrics(y_test,y_pred):
    y_test=[list_cat[np.argmax(l)] for l in y_test]
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred) # Create CM From Data
    dict_F1=cm.class_stat['F1']
    df_metrics=pd.DataFrame.from_dict(dict_F1, orient='index',columns=['F1'])
    df_metrics['ACC']=pd.DataFrame.from_dict(cm.class_stat['ACC'], orient='index' )

    #cm.class_stat
    return df_metrics


def main(name_model,xtest,epochs):
    model=train_model_multiClass(name_model,epochs)
    preds_all=func_preds(model,xtest)
    df_metrics=create_metrics(ytest,preds_all)
    return df_metrics

def main_softmax(name_model,xtest,epochs):
    model,model_softmax=train_model_softmax(name_model,epochs)
    preds_all_max,preds_all_softmax=func_preds_max_softmax(model,model_softmax,xtest)
    df_metrics_max=create_metrics(ytest,preds_all_max)
    df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_metrics_max, df_metrics_softmax


def calculate_mean_std(out_data):
    f1_COH=0
    f1_HYPER=0
    f1_MERO=0
    f1_RANDOM=0
    f1_SYN=0
    f1_COH_list=[]
    f1_HYPER_list=[]
    f1_MERO_list=[]
    f1_RANDOM_list=[]
    f1_SYN_list=[]
    nb_occ = len(out_data)
    for i in range(nb_occ):
        f1_COH_list.append(out_data[i]['F1']['COH'])
        f1_HYPER_list.append(out_data[i]['F1']['HYPER'])
        f1_MERO_list.append(out_data[i]['F1']['MERO'])
        f1_RANDOM_list.append(out_data[i]['F1']['RANDOM'])
        f1_SYN_list.append(out_data[i]['F1']['SYN'])
       
    f1_COH=np.array(f1_COH_list).mean()
    f1_COH_std=np.array(f1_COH_list).std()
    f1_HYPER=np.array(f1_HYPER_list).mean()
    f1_HYPER_std=np.array(f1_HYPER_list).std()
    f1_MERO=np.array(f1_MERO_list).mean()
    f1_MERO_std=np.array(f1_MERO_list).std()
    f1_RANDOM=np.array(f1_RANDOM_list).mean()
    f1_RANDOM_std=np.array(f1_RANDOM_list).std()
    f1_SYN=np.array(f1_SYN_list).mean()
    f1_SYN_std=np.array(f1_SYN_list).std()
 

    matrix_f=[[f1_COH,f1_HYPER,f1_MERO,f1_RANDOM,f1_SYN],
             [f1_COH_std,f1_HYPER_std,f1_MERO_std,f1_RANDOM_std,f1_SYN_std]]
    df_final=pd.DataFrame(matrix_f,columns=['COH','HYPER','MERO','RANDOM','SYN'],
                         index=['F1_mean','STD'])
    return df_final


def multi_main_multi_class(name_model,nb_occ):

    out_data=[]
    
    nb_epochs=50
   
    for i in range(nb_occ):
        
        
        df_oneVsRest=main(name_model,xtest,nb_epochs)
        
        out_data.append(df_oneVsRest)
    df_final=calculate_mean_std(out_data)

    
    return df_final


def multi_main_vote(name_model,nb_n1,nb_n2,nb_occ):

    out_data1=[]
    out_data2=[]
    name_model=name_model(nb_n1,nb_n2)
    nb_epochs=1
   
    for i in range(nb_occ):
        
        
        df_max,df_softmax=main_softmax(name_model,xtest,nb_epochs)
      
        out_data1.append(df_max)
        out_data2.append(df_softmax)
    df_final_max=calculate_mean_std(out_data1)
    df_final_softmax=calculate_mean_std(out_data2)
    
    df_final_max['decision']='max'
   
    df_final_softmax['decision']='sofmax'
    
    df_final_All=df_final_max.append(df_final_softmax)
    df_final_All['nb_n1']=nb_n1
    df_final_All['nb_n2']=nb_n2
    df_final_All['Model']=name_model
    return df_final_All