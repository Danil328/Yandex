# -*- coding: utf-8 -*-
"""
Created on Tue May 03 11:14:05 2016

@author: Danil
"""
from sklearn.metrics import r2_score
import pandas
import numpy as np
features = pandas.read_csv('./features.csv', index_col='match_id')
featuresy=pandas.DataFrame(features.radiant_win)
features.head()

#pred = clf.predict_proba(X_test)[:, 1]
#Градиентный бустинг в лоб
#1
del features['duration']
del features['radiant_win']
del features['tower_status_radiant']
del features['tower_status_dire']
del features['barracks_status_dire']
del features['barracks_status_radiant']

#2
j=0
rows=np.zeros(102)
for col in features:
    for i in col:
        if i==None or i==np.NaN or i==np.nan:
            rows[j]=rows[j]+1
    j=j+1
    
features.fillna(0, inplace=True)
for stolb in features:
    features[stolb]=preprocessing.normalize(features[stolb]).transpose()
#3
    #пропусков нет
#4
#хз
#5
from sklearn.ensemble import GradientBoostingClassifier
features_test = pandas.read_csv('./features_test.csv', index_col='match_id')
clf = GradientBoostingClassifier(n_estimators=30, verbose=True, random_state=241, learning_rate = 0.2) # определили классификатор с 10 деревьями
clf.fit(features, featuresy) # обучили его на тренировочной выборке   
rezkf = np.array(clf.predict(features))
rezr2=r2_score(rezkf,featuresy)





























