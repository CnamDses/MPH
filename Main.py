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

#création de la métrique : pourcentage des prédictions correctes
def score(y_true, y_pred): 
    return np.mean(np.abs(y_true - y_pred)) * 100

# Import des données
data = get_data(False)
# Diviser les données en échantilons d'apprentissage et de tests
(train,test) = extract_test_train_datasets(data)
#Analyse descriptive

#Préparation des données
# features numériques
feat_num = ['nom de colonne']
# features date
feat_dates = []
# features catégorielles
feat_cat = []
# features texte
feat_text = []

#Encodage des features catégorielles
for c in feat_cat:
    le = LabelEncoder()
    le.fit(train[c].append(test[c]))
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])
	
#Création du modéle
train['Y'] = train['ColonneApredire'] #changement de la colonne a predire


	
# Cross validation pour le calibrage du modéle de prédiction
model = "RandomForest"
if(model == "RandomForest"):
	err = 0
	NBROUND = 5 #nomnbre de chunks
	FEATURES = feat_num+feat_cat # on n'utilise que ces features pour ce modèle basique
	for train_index, test_index in KFold(train.shape[0], n_folds=NBROUND):
	    y = train['Y']
	    X = train[FEATURES]
	    X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
	    y_train, y_test = y[train_index], y[test_index]
	    clf = RandomForestRegressor()
	    clf.fit(X_train, y_train)
	    pred = clf.predict(X_test)
	    
	    err += score(y_test, pred)
	    print score(y_test, pred)
	print "*** accuracy_score RandomForest: ", err / NBROUND
	#Calcul des prédictions sur l'échantillon de test
	clf = RandomForestRegressor()
	# On entraine de nouveau le modèle, cette fois sur l'intégralité des données
	clf.fit(train[FEATURES], train['Y'])
	predictions = np.exp(clf.predict(test[FEATURES]))
else if (model == "LogisticRegression"):
	from sklearn import linear_model
	err = 0
	NBROUND = 5 #nomnbre de chunks
	FEATURES = feat_num+feat_cat # on n'utilise que ces features pour ce modèle basique
	for train_index, test_index in KFold(train.shape[0], n_folds=NBROUND):
	    y = train['Y']
	    X = train[FEATURES]
	    X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
	    y_train, y_test = y[train_index], y[test_index]
	    clf = linear_model.LogisticRegression()
	    clf.fit(X_train, y_train)
	    pred = clf.predict(X_test)
	    
	    err += score(y_test, pred)
	    print score(y_test, pred)
	print "*** accuracy_score Logistic Regression : ", err / NBROUND
	#Calcul des prédictions sur l'échantillon de test
	clf = RandomForestRegressor()
	# On entraine de nouveau le modèle, cette fois sur l'intégralité des données
	clf.fit(train[FEATURES], train['Y'])
	predictions = np.exp(clf.predict(test[FEATURES]))


