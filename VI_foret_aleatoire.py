
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
     # 3 techniques, laquelle choisir?       
     # on prends la 3ème avec le package écrit "en dur" dans le programme
"""

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


     # OU (?) :
"""
     #=> appeler le prog  "package rf_perm_feat_import version 2.X.py"
      #ou appeler le prog  "package rf_perm_feat_import version 3.X.py"
      


     # j'importe les données et les packages dont on a besoin 
     # => j'ai utiliser le "main" du 1er sept 2017
     #=> Firas  
 
    
# FIras dernière version du programme à ajouter :
         #=> Firas?
   y = train[Y] 
   X = train[FEATURES] 


     #  on lance la selection de variables

   
   #  on test avec ntree = 2000 et pour mtry on peut surement tester sqrt(250)


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
    VI_imp_oob=oobR.featureImportances(rfR, X, y, 1)
    
    return VI_imp_def,VI_imp_def



def boucle_foret(X, y,ntree,mtry,NombreDeForet,nom):
        
    # besoin d'initialisation des variables
    VI_imp_def_sortie=list()
    VI_imp_oob_sortie=list()
    
     
    
az=[oobC.featureImportances(rfC, X, y, 5)]

    az_temp=[oobC.featureImportances(rfC, X, y, 5)]



    # on lance NombreDeForet fois 
    for i in range(NombreDeForet):
    
        lancement_une_foret(X, y,ntree,mtry)
         
        VI_imp_def_sortie.append(VI_imp_def)
        VI_imp_oob_sortie.append(VI_imp_oob)
               
        
    
    
    
    # VI_imp_def_sortie et VI_imp_oob_sortie sont des listes de Array
        
         
    # besoin d'initialisation des variables   
        # V1
    moyenne_VI_imp_def_sortie=list()
    ecart_type_VI_imp_def_sortie=list() 
        # V2
    moyenne_VI_imp_oob_sortie=list()
    ecart_type_VI_imp_oob_sortie=list()
    
    i=0
    while i < len(VI_imp_def_sortie[1]):
        # V1
        moyenneI = np.mean(VI_imp_def_sortie[i] ) 
        moyenne_VI_imp_def_sortie.append(moyenneI)
        
        EcartTypeI = np.std(VI_imp_def_sortie[i] ) 
        ecart_type_VI_imp_def_sortie.append(EcartTypeI)
    
        # V2
        moyenneI = np.mean(VI_imp_oob_sortie[i] ) 
        moyenne_VI_imp_oob_sortie.append(moyenneI)
        
        EcartTypeI = np.std(VI_imp_oob_sortie[i] ) 
        ecart_type_VI_imp_oob_sortie.append(EcartTypeI)
    
        i=i+1
    
     
    
    #moyenne et écart type des 2 sortes de VI pour les 50 forets 
    
    #données à récuperer
    print "\n"
    print "#######\n--VI données par sklearn--\n#######"
    print "Weighted Avg Information Gain feature importances:"
    
    print moyenne_VI_imp_def_sortie    
 
    print "écart type"
    
    print ecart_type_VI_imp_def_sortie 
 

    print "\n"
    print "#######\n--VI données par le package erreur oob --\n#######"
    print "Permutation importances:"
    
    print (moyenne_VI_imp_oob_sortie)
    
    print "écart type"
    
    print (ecart_type_VI_imp_oob_sortie)

 
      return None


# on teste différente cas pour vérif les paramétres et robustesse
    
boucle_foret(X, y,2000,15,50,V1):
    
boucle_foret(X, y,2000,50,50,V1):
    
boucle_foret(X, y,2000,100,50,V1):
    
boucle_foret(X, y,500,100,50,V1):
    
boucle_foret(X, y,100,100,50,V1):
    
       











