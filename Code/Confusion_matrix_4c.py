
#name_data="Cogalex"
name_data="Linkedin"
print(name_data)
if name_data =='Linkedin' :
    exec(open('Dataprep_Linkedin.py').read())
    exec(open('model_hier_Linkedin.py').read())
    input_dime=1536

exec(open('Models_MultiClass_4C.py').read())


epochs=50
model=ModelBaseline_MultiClass()
model.fit(xtrain, ytrain,
                epochs=epochs, validation_data=(xvalid, yvalid),  verbose=False, callbacks=[EarlyStopping(patience=5)])
  


def func_preds(model,xtest):
    preds = model.predict(xtest, verbose=False)
    preds_all=[list_cat[np.argmax(l)] for l in preds]
    return preds_all

preds_all=func_preds(model,np.array(xtest))
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