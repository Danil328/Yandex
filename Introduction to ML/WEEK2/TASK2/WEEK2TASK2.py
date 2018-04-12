from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
import numpy
import operator


boston = load_boston()

boston_features_normalized = preprocessing.scale(boston.data)

 
steps = numpy.linspace(1, 10, num=200)
dict_ = {}


def task (a):
    for i in steps:
        kf = KFold(n = 506 ,n_folds=5, shuffle=True, random_state=42)
        kn = KNeighborsRegressor(n_neighbors=5, metric = 'minkowski', weights='distance', p=i)
        s = cross_validation.cross_val_score(X = boston_features_normalized, y=boston.target , estimator=kn, cv=kf ,scoring='mean_squared_error')
        avg=s.sum()/len(s)
        a[i]=avg
    return
    
task(dict_)
sorted_ = sorted(dict_.items(), key=operator.itemgetter(1)) #тут на ходим параметр p с максимальным значенем
