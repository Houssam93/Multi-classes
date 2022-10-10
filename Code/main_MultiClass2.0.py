exec(open('prepare_data_multiClass.py').read())

exec(open('Models_Hier.py').read())

#name_models=[Model_OneVsRest_Hier1_0,Model_OneVsRest_Hier1_1,Model_OneVsRest_Hier1_2,
#Model_OneVsRest_Hier2_0,Model_OneVsRest_Hier2_1,Model_OneVsRest_Hier2_2,Model_OneVsRest_Hier2_3]
#name_str=['Model_OneVsRest_Hier1_0','Model_OneVsRest_Hier1_1','Model_OneVsRest_Hier1_2',
#'Model_OneVsRest_Hier2_0','Model_OneVsRest_Hier2_1','Model_OneVsRest_Hier2_2','Model_OneVsRest_Hier2_3']
name_models=[Model_OneVsRest_Hier1_1RMSCH]
name_str=['Model_OneVsRest_Hier1_1RMSCH']



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




def main_softmax(name_model,xtest,epochs):
    model=train_model_softmax(name_model,epochs)
    preds_all_max=func_preds_max_softmax(model,xtest)
    df_metrics_max=create_metrics(ytest,preds_all_max)
    return df_metrics_max

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




def multi_main_vote(name_model,name_str_n,nb_n1,nb_n2,nb_occ):

    out_data1=[]
   
    name_model=name_model(nb_n1,nb_n2)
    nb_epochs=50
   
    for i in range(nb_occ):
        
        
        df_max=main_softmax(name_model,xtest,nb_epochs)
      
        out_data1.append(df_max)
        
    df_final_max=calculate_mean_std(out_data1)

    
    df_final_max['decision']='max'
   

    
    df_final_All=df_final_max
    df_final_All['nb_n1']=nb_n1
    df_final_All['nb_n2']=nb_n2
    df_final_All['Model']=name_str_n
    return df_final_All



def main(inp):
    nb_occ=40
    nb_1=inp[0]
    nb_2=inp[1]
    nb_1=int(nb_1)
    nb_2=int(nb_2)
    #df_restit=pd.DataFrame()
    df_restit=pd.DataFrame()
    for nnn in range(len(name_models)) :
    	print(name_str[nnn])
    	df_restit=df_restit.append(multi_main_vote(name_models[nnn],name_str[nnn],nb_1,nb_2,nb_occ))

    df_restit.to_csv('../Results/Hier_Out_Results2Bis_n1'+str(nb_1)+'_n2_'+str(nb_2)+'.csv')
#df=main(1,1)

if __name__ == "__main__":
	print(sys.argv[1:])

	main(sys.argv[1:])