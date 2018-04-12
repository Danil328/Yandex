import pandas
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

#ЗАДАНИЕ 1=====================================================================
features = pandas.read_csv('./features.csv', index_col='match_id') #загрузили таблицу

target = features.radiant_win.values #заранее выделяемый целевой вектор

del features['duration']  #удалили столбцы, связанные с итогами
del features['radiant_win']
del features['tower_status_radiant']
del features['tower_status_dire']
del features['barracks_status_dire']
del features['barracks_status_radiant']

features.fillna(value = 0, inplace = True) 

#scaler = StandardScaler() #так как линейные алгоритмы чувствительны к масштабу
#features_scaled = scaler.fit_transform(features)
#
#
#start_time = datetime.datetime.now()
#
#clf = LogisticRegression(penalty = 'l2', random_state=241)
#k_fold = KFold(n = len(features) ,n_folds=5, shuffle=True, random_state=241)
#s = cross_validation.cross_val_score(X = features_scaled, y=target , estimator=clf, cv=k_fold, scoring = 'roc_auc')
#avg=s.sum()/len(s)
#
#print ('Time elapsed:', datetime.datetime.now() - start_time)

#при любом C качество  = 0,716. Качество не сильно уступает бустингу(почемуууу???). Работает намного быстрей.



#ЗАДАНИЕ 2=====================================================================
#del features['lobby_type']
#del features['r1_hero']
#del features['r2_hero']
#del features['r3_hero']
#del features['r4_hero']
#del features['r5_hero']
#del features['d1_hero']
#del features['d2_hero']
#del features['d3_hero']
#del features['d4_hero']
#del features['d5_hero']
#
#scaler = StandardScaler() #так как линейные алгоритмы чувствительны к масштабу
#features_scaled = scaler.fit_transform(features)
#
#
#start_time = datetime.datetime.now()
#
#clf = LogisticRegression(penalty = 'l2', random_state=241)
#k_fold = KFold(n = len(features) ,n_folds=5, shuffle=True, random_state=241)
#s = cross_validation.cross_val_score(X = features_scaled, y=target , estimator=clf, cv=k_fold, scoring = 'roc_auc')
#avg=s.sum()/len(s)
#
#print ('Time elapsed:', datetime.datetime.now() - start_time)


#качество слегка изменилось в лучшую сторону. почему(????)


#ЗАДАНИЕ 3=====================================================================

unique = features.r1_hero.unique() 
unique.sort() #всего 112 уникальных героев


#ЗАДАНИЕ 4=====================================================================
N = 112
X_pick = np.zeros((features.shape[0], N))

for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

del features['lobby_type']
del features['r1_hero']
del features['r2_hero']
del features['r3_hero']
del features['r4_hero']
del features['r5_hero']
del features['d1_hero']
del features['d2_hero']
del features['d3_hero']
del features['d4_hero']
del features['d5_hero']

scaler = StandardScaler() #так как линейные алгоритмы чувствительны к масштабу
features_scaled = scaler.fit_transform(features)

features_scaled_bagofwords = np.hstack((features_scaled, X_pick))

start_time = datetime.datetime.now()

clf = LogisticRegression(penalty = 'l2', random_state=241)
k_fold = KFold(n = len(features) ,n_folds=5, shuffle=True, random_state=241)
s = cross_validation.cross_val_score(X = features_scaled_bagofwords, y=target , estimator=clf, cv=k_fold, scoring = 'roc_auc')
avg=s.sum()/len(s)

print ('Time elapsed:', datetime.datetime.now() - start_time)

#качество значительно улучшилось. почему(????)










