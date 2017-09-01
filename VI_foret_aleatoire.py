
"""
Created on Fri Sep 01 15:11:43 2017

@author: BOUCHET-08822
"""

"""
Variable Selection Using Random Forests

DAns un 1er temps on met en place la stratégie détaillée page 22 de l'article :
    
VSURF: An R Package for Variable Selection Using Random Forests
by Robin Genuer, Jean-Michel Poggi and Christine Tuleau-Malot

+ cf doc

Random Forests for Big Data
by Robin Genuer, Jean-Michel Poggi, Christine Tuleau-Malot, Nathalie Villa-Vialaneix

et 

Variable selection using Random Forests
by Robin Genuer, Jean-Michel Poggi, Christine Tuleau-Malot
(mais plus orienté génomique)
 
"""



""" 
dans sklearn , la sortie variable importance n'est pas celle que l'on veut 
cf https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined

=> on utilise alors le package de  rf_perm_feat_import (attention je pense que c'est en version python 2.X)

https://github.com/pjh2011/rf_perm_feat_import

"""

print "\n"
print "#######\n-- Variable Selection Using Random Forests --\n#######"


     # on importe le package 
     # 2 techniques, laquelle choisir?       
     # surement du python 2.7 ?
     #=> Firas?

 from setuptools import setup 
 
 
 setup(name='rf_perm_feat_import', 
       version='0.1', 
       description='Random Forest Permutate Feature Importance', 
       url='https://github.com/pjh2011/sklearn_perm_feat_import', 
       author='Peter Hughes', 
       author_email='pethug210@gmail.com', 
       license='MIT', 
       packages=['rf_perm_feat_import'], 
       install_requires=[ 
           'numpy', 
            'sklearn' 
       ], 
       zip_safe=False) 


     # OU (?) :

pip install rf_perm_feat_import
from .RFFeatureImportance import PermutationImportance



 

     # j'importe les données et les packages dont on a besoin 
     # => j'ai utiliser le "main" du 1er sept 2017
     #=> Firas?

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns # pip install seaborn



def get_data(SAS):
	if SAS:
    	data = pd.read_sas('/data.SAS7bDAT', encoding='utf-8', format=True, sep=';')
	else:
		data = pd.read_csv('D://mnist/train.csv', encoding='utf-8', sep=';')
	return data


def extract_test_train_datasets(data):
	df = pd.DataFrame(np.random.randn(4000000, 2))
	msk = np.random.rand(len(df)) < 0.8
	train = df[msk]
	test = df[~msk]
	return (train, test)
 
    
# FIras dernière version du programme à ajouter :
         #=> Firas?
   y = train[Y] 
   X = train[FEATURES] 


     #  on lance la seleciton de variable

   
   #  eventuellement ntree = 2000 et pour mtry on peut surement tester sqrt(250)


def lancement_une_foret(X, y,ntree,mtry):
    
rfR = RandomForestRegressor(n_estimators=ntree, max_features=mtry,oob_score=True)
rfR.fit(X, y)


#print "\n"
#print "#######\n--VI données par sklearn--\n#######"
#print "Weighted Avg Information Gain feature importances:"

VI_imp_def=rfR.feature_importances_

#print "\n"
#print "#######\n--VI données par le package erreur oob --\n#######"
#print "Permutation importances:"
oobR = PermutationImportance()
VI_imp_oob=oobR.featureImportances(rfR, train, test, 1)

 return VI_imp_def,VI_imp_def


# besoin d'initialisation ?
    #=> Firas?

# on lance 50 fois 
	for i in range(50):

... lancement_une_foret(train, test,2000,15)
... 
...  VI_imp_def_sortie=VI_imp_def_sortie, VI_imp_def
...  VI_imp_oob_sortie=VI_imp_oob_sortie, VI_imp_oob
  
 
#moyenne
moyenne_VI_imp_def_sortie=np.mean(VI_imp_def_sortie)
moyenne_VI_VI_imp_oob_sortie=np.mean(VI_imp_oob_sortie)

#ecart-type
ecart_type_VI_imp_def_sortie=np.std(VI_imp_def_sortie)
ecart_type_VI_VI_imp_oob_sortie=np.std(VI_imp_oob_sortie)


    #données à récuperer
    #=> Firas?
    
    
    
    
	for i in range(50):

... lancement_une_foret(train, test,2000,50)
... 
...  VI_imp_def_sortie2=VI_imp_def_sortie2, VI_imp_def
...  VI_imp_oob_sortie2=VI_imp_oob_sortie2, VI_imp_oob
  
 
#moyenne
moyenne_VI_imp_def_sortie2=np.mean(VI_imp_def_sortie)
moyenne_VI_VI_imp_oob_sortie2=np.mean(VI_imp_oob_sortie)

#ecart-type
ecart_type_VI_imp_def_sortie2=np.std(VI_imp_def_sortie)
ecart_type_VI_VI_imp_oob_sortie2=np.std(VI_imp_oob_sortie)


    #données à récuperer
    #=> Firas?
    
	for i in range(50):

... lancement_une_foret(train, test,2000,100)
... 
...  VI_imp_def_sortie3=VI_imp_def_sortie3, VI_imp_def
...  VI_imp_oob_sortie3=VI_imp_oob_sortie3, VI_imp_oob
  
 
#moyenne
moyenne_VI_imp_def_sortie3=np.mean(VI_imp_def_sortie)
moyenne_VI_VI_imp_oob_sortie3=np.mean(VI_imp_oob_sortie)

#ecart-type
ecart_type_VI_imp_def_sortie3=np.std(VI_imp_def_sortie)
ecart_type_VI_VI_imp_oob_sortie3=np.std(VI_imp_oob_sortie)


    #données à récuperer
    #=> Firas?




    
	for i in range(50):

... lancement_une_foret(train, test,100,100)
... 
...  VI_imp_def_sortie4=VI_imp_def_sortie4, VI_imp_def
...  VI_imp_oob_sortie4=VI_imp_oob_sortie4, VI_imp_oob
  
 
#moyenne
moyenne_VI_imp_def_sortie4=np.mean(VI_imp_def_sortie)
moyenne_VI_VI_imp_oob_sortie4=np.mean(VI_imp_oob_sortie)

#ecart-type
ecart_type_VI_imp_def_sortie4=np.std(VI_imp_def_sortie)
ecart_type_VI_VI_imp_oob_sortie4=np.std(VI_imp_oob_sortie)


    #données à récuperer
    #=> Firas?
    
	for i in range(50):

... lancement_une_foret(train, test,500,100)
... 
...  VI_imp_def_sortie5=VI_imp_def_sortie5, VI_imp_def
...  VI_imp_oob_sortie5=VI_imp_oob_sortie5, VI_imp_oob
  
 
#moyenne
moyenne_VI_imp_def_sortie5=np.mean(VI_imp_def_sortie)
moyenne_VI_VI_imp_oob_sortie5=np.mean(VI_imp_oob_sortie)

#ecart-type
ecart_type_VI_imp_def_sortie5=np.std(VI_imp_def_sortie)
ecart_type_VI_VI_imp_oob_sortie5=np.std(VI_imp_oob_sortie)


    #données à récuperer
    #=> Firas?















