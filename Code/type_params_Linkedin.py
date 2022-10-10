#name_data="Cogalex"
#name_data="Cogalex"
name_data="Linkedin"
print(name_data)
if name_data =='Linkedin' :
    exec(open('Dataprep_Linkedin.py').read())
    exec(open('model_hier_Linkedin.py').read())
    input_dime=1536

exec(open('Models_MultiClass_4C.py').read())


import time




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
    return df_overall


def main_multi(name_model,xtest,epochs):
    model=train_model_multiClass(name_model,epochs)
    if name_data =='Linkedin':
        preds_all=func_preds(model,np.array(xtest))
    else :
        preds_all=func_preds(model,xtest)
    df_overall=create_metrics(ytest,preds_all)
    #df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_overall

def main_softmax(name_model,xtest,epochs):
    model=train_model_softmax(name_model,epochs)
    if name_data =='Linkedin':
        preds_all_max=func_preds_max_softmax(model,np.array(xtest))
    else :
        preds_all_max=func_preds_max_softmax(model,xtest)
    df_overall=create_metrics(ytest,preds_all_max)
    #df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_overall



name_str=['MultiClass','OR','AS','ORSP','HIER']
type_params=['Très Faible','Faible','Moyen','Fort','Très Fort']
name_models=[ModelBaseline_MultiClass,ModelBaseline_OneVsRest,ModelOneVsRest_AllShared,ModelOneVsRest_SharedPrivate,hier]
dic_params={'MultiClass':[(10,5),(50,5),(200,50),(300,200),(500,500)],
           'OR': [(10,5),(10,100),(50,5),(100,20),(200,20)],
           'AS': [(10,5),(50,20),(200,50),(300,20),(500,100)],
           'ORSP': [(10,5),(50,5),(100,20),(200,20),(500,50)],
           'HIER': [(10,5),(50,5),(200,5),(300,100),(500,300)]}



df_out=pd.DataFrame()
for i in range(len(name_str)):
    print(name_str[i])
    if name_str[i]=='MultiClass':
        for j in range(len(type_params)):
            start = time.time()
            df_metrics=main_multi(name_models[i](dic_params[name_str[i]][j][0],dic_params[name_str[i]][j][1]),xtest,5)
            end = time.time()
            df_metrics['model']=name_str[i]
            df_metrics['time']=end - start
        
            df_metrics['type_parm']=type_params[j]
            df_out=df_out.append(df_metrics)
    else : 
        
        for j in range(len(type_params)):
            start = time.time()
            df_metrics=main_softmax(name_models[i](dic_params[name_str[i]][j][0],dic_params[name_str[i]][j][1]),xtest,5)
            end = time.time()
            df_metrics['model']=name_str[i]
            df_metrics['time']=end - start
        
            df_metrics['type_parm']=type_params[j]
            df_out=df_out.append(df_metrics)
    print(df_out.values)
df_out.to_csv('output_final'+name_data+'.csv')