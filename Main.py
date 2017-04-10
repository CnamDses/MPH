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
    	data = pd.read_sas('/data.SAS7bDAT', encoding='utf-8', format=True, sep=';')
	else:
		data = pd.read_csv('D://mnist/train.csv', encoding='utf-8', sep=';')
	return data

#extraction échantillon d'apprentissage et de test
def extract_test_train_datasets(data):
	df = pd.DataFrame(np.random.randn(4000000, 2))
	msk = np.random.rand(len(df)) < 0.8
	train = df[msk]
	test = df[~msk]
	return (train, test)

#création de la métrique : pourcentage des prédictions correctes
def score(y_true, y_pred): 
    return np.mean(np.abs(y_true - y_pred)) * 100

#fitting model
def fit_model(model, NBROUND, FEATURES, train, Y):
	err = 0
	if(model = "RF"):
		clf = RandomForestRegressor()
		else clf = linear_model.LogisticRegression()

	for train_index, test_index in KFold(train.shape[0], n_folds=NBROUND):
		y = train[Y]
	    X = train[FEATURES]
	    X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
	    y_train, y_test = y[train_index], y[test_index]
			   
	    clf.fit(X_train, y_train)
	    pred = clf.predict(X_test)
			    
	    err += score(y_test, pred)
	    print score(y_test, pred)
		print "*** accuracy_score RandomForest: ", err / NBROUND
	return clf
			

#Le script 

data = get_data(False)

(train,test) = extract_test_train_datasets(data)

#Préparation des données
# features numériques
feat_num = ['nom de colonne1', 'nom de colonne2']
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
	
#Construction du modéle de prédiction
fit_model("RF", 5, feat_cat + feat_num, train, "NomColonneApredire")



