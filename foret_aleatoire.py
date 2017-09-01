
"""
Created on Fri Sep 01 15:11:43 2017

@author: BOUCHET-08822
"""

"""
Random Forests sur toutes les variables pour test
 
     
"""
 

print "\n"
print "#######\n-- Random Forests V1 --\n#######"

 
 

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


     #  on lance une foret aléatoire sur toutes les variables pour avoir une idée

     
rfR = RandomForestRegressor(n_estimators=2000, max_features=15,oob_score=True)
rfR.fit(X, y)
 

rfR.feature_importances_
rfR.predict(X)
rfR.score(X)




 