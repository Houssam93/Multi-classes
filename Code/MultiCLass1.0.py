
from joblib import Parallel, delayed
import os
import numpy as np
np.random.seed(44)

import itertools


nb_n1=['50','100','200','300']
nb_n2=['5','20','50','100','150']
#nb_n2=['1','2','3','4','5']
#nb_n1=['1','2','3','4','5']

def func(nb_1,nb_2) :
	os.system("python3 main_MultiClass1.0_16C.py "+nb_1+" "+nb_2)

#n_jobs=len(list(itertools.product(nb_n1,nb_n2)))
#if n_jobs>10 :
n_jobs=20
Parallel(n_jobs=n_jobs)(delayed(func)(nb_1,nb_2) for (nb_1,nb_2) in list(itertools.product(nb_n1,nb_n2)))

[5,20,50,100,150,200,300]