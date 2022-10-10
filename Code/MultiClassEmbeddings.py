
from keras import backend as K


#exec(open('prepare_IMDB_Embed.py').read())
#nb_class=5

exec(open('prepare_data_Tweets_3C_embed.py').read())
nb_class=3

def MultiClass(input_dime,nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(200, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model


def AllShared(input_dime1,input_dime2,nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs1 = Input(shape=(input_dime1,))
    inputs2 = Input(shape=(input_dime2,))

    xpri1=Dense(200, activation='sigmoid')(inputs1)
    xpri2=Dense(200, activation='sigmoid')(inputs2)
    

    xshared=Dense(nb_n1, activation='sigmoid')


    xshared1=xshared(xpri1)
    xshared2=xshared(xpri2)


    xprivate1 = Dense(nb_n2, activation='sigmoid')(xshared1)
    xprivate2=Dense(nb_n2, activation='sigmoid')(xshared2)

    out1 = Dense(nb_class, activation='sigmoid', name='out1')(xprivate1)
    #coord=Dropout(0.05)(coord)
    out2 = Dense(nb_class, activation='sigmoid', name='out2')(xprivate2)
    
    model1 = Model(inputs=[inputs1], outputs=[out1])
    model2 = Model(inputs=[inputs2], outputs=[out2])

    model1.compile(optimizer='Adam', loss='categorical_crossentropy')
    model2.compile(optimizer='Adam', loss='categorical_crossentropy')
    
    
    return model1,model2

def SharedPrivate(input_dime1,input_dime2,nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs1 = Input(shape=(input_dime1,))
    inputs2 = Input(shape=(input_dime2,))

    xpri1=Dense(200, activation='sigmoid')(inputs1)
    xpri2=Dense(200, activation='sigmoid')(inputs2)
    

    xshared=Dense(nb_n1, activation='sigmoid')


    xshared1=xshared(xpri1)
    xshared2=xshared(xpri2)

    in_private1=concatenate([xshared1,inputs1])
    in_private2=concatenate([xshared2,inputs2])
    xprivate1 = Dense(nb_n2, activation='sigmoid')(in_private1)
    xprivate2=Dense(nb_n2, activation='sigmoid')(in_private2)

    out1 = Dense(nb_class, activation='sigmoid', name='out1')(xprivate1)
    #coord=Dropout(0.05)(coord)
    out2 = Dense(nb_class, activation='sigmoid', name='out2')(xprivate2)
    
    model1 = Model(inputs=[inputs1], outputs=[out1])
    model2 = Model(inputs=[inputs2], outputs=[out2])

    model1.compile(optimizer='Adam', loss='categorical_crossentropy')
    model2.compile(optimizer='Adam', loss='categorical_crossentropy')
    
    
    return model1,model2

def create_metrics(y_test,y_pred):
    y_test=[list_cat[np.argmax(l)] for l in y_test]
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred) # Create CM From Data
    overall_stats=[cm.overall_stat['F1 Micro'],cm.overall_stat['F1 Macro'],cm.overall_stat['ACC Macro'],cm.overall_stat['AUNU'],cm.overall_stat['AUNP']]
    df_overall=pd.DataFrame([overall_stats],columns=['F1 Micro' ,'F1 Macro','ACC MACRO','AUNU','AUNP'])
    #cm.class_stat
    return df_overall
def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all

num_epochs=2000

dim1=len(data["Emb1"]["xtrain"][0])
dim2=len(data["Emb2"]["xtrain"][0])
name_model=SharedPrivate
def main_model(name_model):
	n1=100
	n2=50
	metrics_scores1,metrics_scores2=[],[]
	metrics_scores_val1,metrics_scores_val2=[],[]
	if name_model == MultiClass :
		model1 = name_model(dim1,n1,n2)
		model2 = name_model(dim2,n1,n2)
	else :
		model1, model2 = name_model(dim1,dim2,n1,n2)
	for epoch in range(num_epochs):
		a=len(data["Emb1"]["xtrain"])
		idx = np.random.choice(np.arange(a), 64, replace=False)
		inputs_train_E1=data["Emb1"]["xtrain"][idx]
		inputs_train_E2=data["Emb2"]["xtrain"][idx]
		model1.fit(inputs_train_E1, data["Emb1"]["ytrain"][idx], epochs=1, validation_data=None, verbose=False, )
		model2.fit(inputs_train_E2, data["Emb2"]["ytrain"][idx], epochs=1, validation_data=None, verbose=False, )
		if epoch%100==0 :
	                preds1 = func_preds(model1,data["Emb1"]["xtest"])
	                preds2 = func_preds(model2,data["Emb2"]["xtest"])
	                metrics_scores1.append(create_metrics(data["Emb1"]["ytest"],preds1))
	                metrics_scores2.append(create_metrics(data["Emb2"]["ytest"],preds2))

	                preds1_valid = func_preds(model1,data["Emb1"]["xvalid"])
	                preds2_valid = func_preds(model2,data["Emb2"]["xvalid"])

	                metrics_scores_val1.append(create_metrics(data["Emb1"]["yvalid"],preds1_valid))
	                metrics_scores_val2.append(create_metrics(data["Emb2"]["yvalid"],preds2_valid))
	                
	                


	df1=pd.concat(metrics_scores1,axis=0)
	df2=pd.concat(metrics_scores2,axis=0)
	df_val1=pd.concat(metrics_scores_val1,axis=0)
	df_val2=pd.concat(metrics_scores_val2,axis=0)
	df1=df1.reset_index(drop=True)
	df2=df2.reset_index(drop=True)
	df_val1=df_val1.reset_index(drop=True)
	df_val2=df_val2.reset_index(drop=True)


	ind1=np.argmax(df_val1['ACC MACRO'])
	ind2=np.argmax(df_val2['ACC MACRO'])

	ind3=np.argmax(df_val1['F1 Macro'])
	ind4=np.argmax(df_val2['F1 Macro'])

	ind5=np.argmax(df_val1['F1 Micro'])
	ind6=np.argmax(df_val2['F1 Micro'])

	acc1=df1['ACC MACRO'][ind1]
	acc2=df2['ACC MACRO'][ind2]

	f1mac1=df1['F1 Macro'][ind3]
	f1mac2=df2['F1 Macro'][ind4]

	f1mic1=df1['F1 Micro'][ind5]
	f1mic2=df2['F1 Micro'][ind6]

	df_final=pd.DataFrame(index=['ACC MACRO','F1 MACRO' , 'F1 MICRO'])
	df_final['USE']=[acc1,f1mac1,f1mic1]
	df_final['BERT']=[acc2,f1mac2,f1mic2]
	return df_final

print('MultiClass')
df_final=main_model(MultiClass)
print(df_final)

print('AllShared')
df_final=main_model(AllShared)
print(df_final)

print('SharedPrivate')
df_final=main_model(SharedPrivate)
print(df_final)




