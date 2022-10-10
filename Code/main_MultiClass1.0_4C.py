

#name_data="Cogalex"
name_data="Linkedin"
print(name_data)
if name_data =='Linkedin' :
    exec(open('Dataprep_Linkedin.py').read())
    exec(open('model_hier_Linkedin.py').read())
    input_dime=1536

exec(open('Models_MultiClass_4C.py').read())







def train_model_softmax(name_model,epochs):
    model=name_model
    
    model.fit(xtrain, [ytrain.transpose()[0],ytrain.transpose()[1],ytrain.transpose()[2]
                       ,ytrain.transpose()[3]],
                epochs=epochs, validation_data=(xvalid, [yvalid.transpose()[0],yvalid.transpose()[1],yvalid.transpose()[2]
                       ,yvalid.transpose()[3]]),  verbose=False, callbacks=[EarlyStopping(patience=5)])
    #model_softmax=Model_softmax()
    pred_train=model.predict(xtrain, verbose=False)
    pred_valid=model.predict(xvalid,verbose=False)
    #model_softmax.fit((np.array(list(map(list, zip(*pred_train))))[:,:,0]),ytrain, epochs=50,
    #                  verbose=False, callbacks=[EarlyStopping(patience=5)]
    #                 ,validation_data=(((np.array(list(map(list, zip(*pred_valid))))[:,:,0])),yvalid))
    #return model,model_softmax
    return model

def train_model_multiClass(name_model,epochs):
    model=name_model
    model.fit(xtrain, ytrain,
                epochs=epochs, validation_data=(xvalid, yvalid),  verbose=False, callbacks=[EarlyStopping(patience=5)])
    return model


from keras import backend as K

def f1_loss(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def Model_softmax():
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(4,))


    out=Dense(4, activation='sigmoid')(inputs)
    
    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer='Adam', loss= [f1_loss])
    return model

def func_preds_max_softmax(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_pre=[]

    #preds_final=model_softmax.predict((np.array(list(map(list, zip(*preds))))[:,:,0]),verbose=False)
    for i in range(len(preds[0])):
        a=np.zeros(4)
        b=[]
        for j in range(4):
            b.append(preds[j][i])
        a[np.argmax(b)]=1
        preds_pre.append(a)
    preds_all_max=[list_cat[np.argmax(l)] for l in preds_pre]
    #preds_all_softmax=[list_cat[np.argmax(l)] for l in preds_final]
    #return preds_all_max,preds_all_softmax
    return preds_all_max

def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all


def create_metrics(y_test,y_pred):
    y_test=[list_cat[np.argmax(l)] for l in y_test]
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred) # Create CM From Data
    overall_stats=[cm.overall_stat['F1 Micro'],cm.overall_stat['F1 Macro'],cm.overall_stat['ACC Macro'],cm.overall_stat['AUNU'],cm.overall_stat['AUNP']]
    dict_F1=cm.class_stat['F1']
    df_metrics=pd.DataFrame.from_dict(dict_F1, orient='index',columns=['F1'])
    df_metrics['ACC']=pd.DataFrame.from_dict(cm.class_stat['ACC'], orient='index' )
    df_metrics['AUC']=pd.DataFrame.from_dict(cm.class_stat['AUC'], orient='index' )
    df_metrics['AUPR']=pd.DataFrame.from_dict(cm.class_stat['AUPR'], orient='index' )
    df_overall=pd.DataFrame([overall_stats],columns=['F1 Micro' ,'F1 Macro','ACC MACRO','AUNU','AUNP'])
    #cm.class_stat
    return df_metrics,df_overall


def main_multi(name_model,xtest,epochs):
    model=train_model_multiClass(name_model,epochs)
    if name_data =='Linkedin':
        preds_all=func_preds(model,np.array(xtest))
    else :
        preds_all=func_preds(model,xtest)
    df_metrics_max,df_overall=create_metrics(ytest,preds_all)
    #df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_metrics_max,df_overall

def main_softmax(name_model,xtest,epochs):
    model=train_model_softmax(name_model,epochs)
    if name_data =='Linkedin':
        preds_all_max=func_preds_max_softmax(model,np.array(xtest))
    else :
        preds_all_max=func_preds_max_softmax(model,xtest)
    df_metrics_max,df_overall=create_metrics(ytest,preds_all_max)
    #df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_metrics_max,df_overall




#name_str=['ModelBaseline_OneVsRest','ModelOneVsRest_AllShared',
#'ModelOneVsRest_SharedPrivate','ModelOneVsRest_All_shared_2Per2',
#'ModelOneVsRest_SharedPrivate_2Per2','ModelOneVsRest_SharedPrivate_2Per2wORandom','ModelOneVsRest_SharedPrivate_3Per3','ModelOneVsRest_SharedPrivate_4Per4']
#print(name_str)
#name_str=['hier','hier_connected','hier_fully_connected']
name_str=['ModelBaseline_MultiClass','ModelBaseline_OneVsRest','ModelOneVsRest_AllShared','ModelOneVsRest_SharedPrivate',
'ModelOneVsRest_All_shared_2Per2','ModelOneVsRest_SharedPrivate_2Per2','ModelOneVsRest_SharedPrivate_3Per3','ModelOneVsRest_AllShared_3Per3','hier']
print(name_str)


def multi_main_vote(name_model,name_str_n,nb_n1,nb_n2,nb_occ,is_multi):

    out_data1=[]
    out_data2=[]
    name_model=name_model(nb_n1,nb_n2)
    nb_epochs=80
    
    if is_multi:
        for i in range(nb_occ):    
            df_max,df_overall=main_softmax(name_model,xtest,nb_epochs)
            out_data1.append(df_max)
            out_data2.append(df_overall)
            df_max.to_csv('../Results_'+name_data+'/Classes_n1'+str(nb_n1)+'_n2_'+str(nb_n2)+name_str_n+'_it_'+str(i)+'.csv')
            df_overall.to_csv('../Results_'+name_data+'/Overall_n1'+str(nb_n1)+'_n2_'+str(nb_n2)+name_str_n+'_it_'+str(i)+'.csv')
    else:
        for i in range(nb_occ):    
            df_max,df_overall=main_multi(name_model,xtest,nb_epochs)
            out_data1.append(df_max)
            out_data2.append(df_overall)
            df_max.to_csv('../Results_'+name_data+'/Classes_n1'+str(nb_n1)+'_n2_'+str(nb_n2)+name_str_n+'_it_'+str(i)+'.csv')
            df_overall.to_csv('../Results_'+name_data+'/Overall_n1'+str(nb_n1)+'_n2_'+str(nb_n2)+name_str_n+'_it_'+str(i)+'.csv')
    df_concat_max=pd.concat(out_data1)
    df_concat_overall=pd.concat(out_data2)

    by_row_index = df_concat_max.groupby(df_concat_max.index)
    df_means_max = by_row_index.mean()

    by_row_index = df_concat_overall.groupby(df_concat_overall.index)
    df_means_overall = by_row_index.mean()

    df_means_max['nb_n1']=nb_n1
    df_means_max['nb_n2']=nb_n2
    df_means_max['Model']=name_str_n

    df_means_overall['nb_n1']=nb_n1
    df_means_overall['nb_n2']=nb_n2
    df_means_overall['Model']=name_str_n

    return df_means_max,df_means_overall

name_models=[ModelBaseline_MultiClass,ModelBaseline_OneVsRest,ModelOneVsRest_AllShared,ModelOneVsRest_SharedPrivate,
ModelOneVsRest_All_shared_2Per2,ModelOneVsRest_SharedPrivate_2Per2,ModelOneVsRest_SharedPrivate_3Per3,ModelOneVsRest_AllShared_3Per3,hier]
is_MultiClass=[False,True,True,True,
True,True,True,
True,True]
#is_MultiClass=[True,True,True,True,True,True,True]

#name_models=[hier]
#is_MultiClass=[True,True,True]
#name_str=['ModelBaseline_OneVsRest','ModelOneVsRest_AllShared','ModelOneVsRest_SharedPrivate','ModelOneVsRest_All_shared_2Per2','ModelOneVsRest_SharedPrivate_2Per2']


def main(inp):
    nb_occ=24
    nb_1=inp[0]
    nb_2=inp[1]
    nb_1=int(nb_1)
    nb_2=int(nb_2)
    #df_restit=pd.DataFrame()
    df_restit_max=pd.DataFrame()
    df_restit_overall=pd.DataFrame()
    for nnn in range(len(name_models)) :
    	print(name_str[nnn])
    	df_means_max,df_means_overall=multi_main_vote(name_models[nnn],name_str[nnn],nb_1,nb_2,nb_occ,is_MultiClass[nnn])
    	df_restit_max=df_restit_max.append(df_means_max)
    	df_restit_overall=df_restit_overall.append(df_means_overall)

   # df_restit=(multi_main_vote(name_model,nb_1,nb_2,nb_occ))  
    #return df_restit
    df_restit_max.to_csv('../Results/'+name_data+'/Classes_n1'+str(nb_1)+'_n2_'+str(nb_2)+'.csv')
    df_restit_overall.to_csv('../Results/'+name_data+'/Overall_n1'+str(nb_1)+'_n2_'+str(nb_2)+'.csv')


import os
import sys
if __name__ == "__main__":
	print(sys.argv[1:])

	main(sys.argv[1:])