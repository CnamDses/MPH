import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns # pip install seaborn


#charegement des donénes
def get_data(SAS):
	if SAS:
    	data = pd.read_sas('/data.SAS7bDAT', encoding='iso-8859-1', format=True, sep=';')
	else:
		data = pd.read_csv('D://mnist/train.csv', encoding='iso-8859-1', sep=';')
	return data

#extraction échantillon d'apprentissage et de test
def extract_test_train_datasets(data):
	from sklearn.model_selection import train_test_split
	train, test = train_test_split(data, test_size = 0.2)
	return (train, test)

#création de la métrique : pourcentage des prédictions correctes
def score(y_true, y_pred): 
    return np.mean(np.abs(y_true - y_pred)) * 100


def transform_data(train):
	#filter unused columns
	train_filtered = train.filter(regex='$')

	#transform categorical variables
	# features numériques
	feat_age = train.columns.values.filter(regex='age')
	feat_num = train.columns.values.filter(regex='')
	# features date
	feat_dates = train.columns.values.filter(regex='')
	# features catégorielles
	feat_cat = train.columns.values.filter(regex='')

	#Encodage des features catégorielles
	for c in feat_cat:
    	le = LabelEncoder()
    	le.fit(train[c])
    	train[c] = le.transform(train[c])

    return train

#fitting model
def fit_logistic_regression_univ(train, Y):
	from sklearn.linear_model import LogisticRegression
	from sklearn.feature_selection import SelectFromModel
	logisticRegressionUniv = LogisticRegression(penalty="l2",C = 1, dual=False)
	model = SelectFromModel(logisticRegressionUniv, threshold = 0.2, prefit=False)
	model.fit(train, Y)
	X_new = model.transform(train)
	X_new.shape
(150, 3)

			

#Le script 

data = get_data(False)

(train,test) = extract_test_train_datasets(data)

#Préparation des données

	
#Construction du modéle de prédiction
fit_model("RF", 5, feat_cat + feat_num, train, "NomColonneApredire")



