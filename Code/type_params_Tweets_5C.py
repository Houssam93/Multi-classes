#name_data="Cogalex"
name_data="Tweets_5C"
print(name_data)
if name_data =='MultiClass' :
    exec(open('prepare_data_multiClass.py').read())
    exec(open('Models_Hier.py').read())
    input_dime=600
if name_data =='Cogalex' :
    exec(open('prepare_data_multiClass_cogalex.py').read())
    exec(open('Models_Hier.py').read())
    input_dime=600
if name_data =='BBC' :
    exec(open('prepare_data_BBC.py').read())
    exec(open('model_hier_BBC.py').read())
    input_dime=512

if name_data =='BBCSport' :
    exec(open('prepare_data_BBCSport.py').read())
    exec(open('model_hier_BBCSport.py').read())
    input_dime=512

if name_data=="Tweets_5C" :
    exec(open('prepare_data_Tweets_5C.py').read())
    exec(open('model_hier_Tweets_5C.py').read())
    input_dime=512

if name_data=="IMDB" :
    exec(open('Dataprep_IMDB.py').read())
    exec(open('model_hier_IMDB.py').read())
    input_dime=512
exec(open('Models_MultiClass.py').read())
import time


def ModelBaseline_MultiClass(nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model






def ModelBaseline_OneVsRest(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))

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
    inputs = Input(shape=(input_dime,))
    
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
    inputs = Input(shape=(input_dime,))
    
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






def train_model_softmax(name_model,epochs):
    model=name_model
    
    model.fit(xtrain, [ytrain.transpose()[0],ytrain.transpose()[1],ytrain.transpose()[2]
                       ,ytrain.transpose()[3],ytrain.transpose()[4]],
                epochs=epochs, validation_data=(xvalid, [yvalid.transpose()[0],yvalid.transpose()[1],yvalid.transpose()[2]
                       ,yvalid.transpose()[3],yvalid.transpose()[4]]),  verbose=False, callbacks=[EarlyStopping(patience=5)])
   
    return model
def func_preds_max_softmax(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_pre=[]

    for i in range(len(preds[0])):
        a=np.zeros(5)
        b=[]
        for j in range(5):
            b.append(preds[j][i])
        a[np.argmax(b)]=1
        preds_pre.append(a)
    preds_all_max=[list_cat[np.argmax(l)] for l in preds_pre]

    return preds_all_max

def create_metrics(y_test,y_pred):
    y_test=[list_cat[np.argmax(l)] for l in y_test]
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred) # Create CM From Data
    overall_stats=[cm.overall_stat['F1 Micro'],cm.overall_stat['F1 Macro'],cm.overall_stat['ACC Macro'],cm.overall_stat['AUNU'],cm.overall_stat['AUNP']]
    df_overall=pd.DataFrame([overall_stats],columns=['F1 Micro' ,'F1 Macro','ACC MACRO','AUNU','AUNP'])
    
    return df_overall
def main_softmax(name_model,xtest,epochs):
    model=train_model_softmax(name_model,epochs)
    preds_all_max=func_preds_max_softmax(model,xtest)
    df_overall=create_metrics(ytest,preds_all_max)
    #df_metrics_softmax=create_metrics(ytest,preds_all_softmax)
    return df_overall


def train_model_multiClass(name_model,epochs):
    model=name_model
    model.fit(xtrain, ytrain,
                epochs=epochs, validation_data=(xvalid, yvalid),  verbose=False, callbacks=[EarlyStopping(patience=5)])
    return model
def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all

def main_multi(name_model,xtest,epochs):
    model=train_model_multiClass(name_model,epochs)
    preds_all=func_preds(model,xtest)
    df_overall=create_metrics(ytest,preds_all)
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
            df_metrics=main_multi(name_models[i](dic_params[name_str[i]][j][0],dic_params[name_str[i]][j][1]),np.array(xtest),5)
            end = time.time()
            df_metrics['model']=name_str[i]
            df_metrics['time']=end - start
        
            df_metrics['type_parm']=type_params[j]
            df_out=df_out.append(df_metrics)
    else : 
        
        for j in range(len(type_params)):
            start = time.time()
            df_metrics=main_softmax(name_models[i](dic_params[name_str[i]][j][0],dic_params[name_str[i]][j][1]),np.array(xtest),5)
            end = time.time()
            df_metrics['model']=name_str[i]
            df_metrics['time']=end - start
        
            df_metrics['type_parm']=type_params[j]
            df_out=df_out.append(df_metrics)
    print(df_out.values)
df_out.to_csv('output_final'+name_data+'.csv')