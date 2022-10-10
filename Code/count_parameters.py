exec(open('Models_MultiClass.py').read())
exec(open('Models_Hier.py').read())

name_str=['ModelBaseline_OneVsRest','ModelOneVsRest_AllShared',
'ModelOneVsRest_SharedPrivate','ModelOneVsRest_All_shared_2Per2',
'ModelOneVsRest_SharedPrivate_2Per2','ModelOneVsRest_SharedPrivate_2Per2wORandom','ModelOneVsRest_SharedPrivate_3Per3','ModelOneVsRest_SharedPrivate_4Per4',
'hier','hier_connected','hier_fully_connected']

name_models=[ModelBaseline_OneVsRest,ModelOneVsRest_AllShared,ModelOneVsRest_SharedPrivate,
ModelOneVsRest_All_shared_2Per2,ModelOneVsRest_SharedPrivate_2Per2,ModelOneVsRest_SharedPrivate_2Per2wORandom,ModelOneVsRest_SharedPrivate_3Per3,
ModelOneVsRest_SharedPrivate_4Per4,hier,hier_connected,hier_fully_connected]

nb_n1=['50','100','200','300']
nb_n2=['5','20','50','100','150']


result=[]
for nb1 in nb_n1:
	for nb2 in nb_n2:
		print(nb1,nb2)
		for i in range(len(name_str)):
			model=name_models[i](nb1,nb2)
			a=model.count_params()
			result.append([nb1,nb2,name_str[i],a])


df=pd.DataFrame(result,columns=['nb1','nb2','modèle','nb_params'])
df.to_csv('../Results/nb_params.csv')