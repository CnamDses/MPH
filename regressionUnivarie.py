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


	feat_age = train.columns.values.filter(regex='age')
	feat_num = train.columns.values.filter(regex='ben')
	# features date
	feat_dates = train.columns.values.filter(regex='')
	# features catégorielles
	feat_cat = train.columns.values.filter(regex='')
def transform_data(train):
	#filter unused columns
	train_filtered = train.filter(regex='$')

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

#remplacement des valeurs manquantes par 0 (A discuter avec tout le monde)
data = data.fillna(0)

#Préparation des données
#Construction de la variable à expliquer
data['urgence_sans_hospit_2014'] = np.where(data['nb_ush_trimestre_1', 'nb_ush_trimestre_2', 'nb_ush_trimestre_3', 'nb_ush_trimestre_4'].max(axis=1) >= 1, 1, 0)
								
data['urgence_avec_hospit_2014'] = np.where(data['nb_uah_trimestre_1', 'nb_uah_trimestre_2', 'nb_uah_trimestre_3', 'nb_uah_trimestre_4'].max(axis=1) >= 1, 1, 0)

data['urgence_sans_hospit_2013'] = np.where(data['nb_ush_trimestre_m1', 'nb_ush_trimestre_m2', 'nb_ush_trimestre_m3', 'nb_ush_trimestre_m4'].max(axis=1) >= 1, 1, 0)
							
data['urgence_avec_hospit_2013'] = np.where(data['nb_uah_trimestre_m1', 'nb_uah_trimestre_m2', 'nb_uah_trimestre_m3', 'nb_uah_trimestre_m4'].max(axis=1) >= 1, 1, 0)

#codage des variables categorielles
feat_cat = data.filter(regex = 'cla_age').columns.values
from sklearn.preprocessing import LabelEncoder
for col in feat_cat:
	le = LabelEncoder()
	le.fit(data[c])
	data[c] = le.transform(data[c])





#Construction du modéle univarié
from sklearn.feature_selection import f_classif
X = data
Y = data['urgence_avec_hospit_2014']
feat = np.concatenate(feat_cat,feat_num)

 # Test with Anova + LogisticRegression
clf = LogisticRegression()
selector = SelectKBest(f_classif, k=3)
selector.fit(X,Y)
for (s,n) in (feat,selector.pvalues_):
	print (s,n)


