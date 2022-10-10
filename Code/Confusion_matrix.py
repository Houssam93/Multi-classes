#name_data="Cogalex"
name_data="IMDB"
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


epochs=50
model=ModelBaseline_MultiClass()
model.fit(xtrain, ytrain,
                epochs=epochs, validation_data=(xvalid, yvalid),  verbose=False, callbacks=[EarlyStopping(patience=5)])
  


def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all



if name_data =='BBC' or name_data =='BBCSport' or name_data =='Tweets_5C'  or  name_data =='IMDB':
        preds_all=func_preds(model,np.array(xtest))
else :
        preds_all=func_preds(model,xtest)


y_test=ytest
y_pred=preds_all
y_test=[list_cat[np.argmax(l)] for l in y_test]
cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred) # Create CM From Data


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm=pd.DataFrame(cm.matrix)

#plt.figure(figsize=(20,14))
#sn.set(font_scale=1) # for label size
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

f, ax = plt.subplots(figsize=(20, 14))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, ax=ax)

plt.savefig('../images/'+name_data+'_htmpap.png')
#plt.show()