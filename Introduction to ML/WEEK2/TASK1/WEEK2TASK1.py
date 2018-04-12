import pandas
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import operator
from sklearn import preprocessing

data = pandas.read_csv('wine.data', names=['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])


classes = data.Class.values
features = data[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']].as_matrix()

dict_ = {}

def task (a, b):
    for i in range(1,51):
        kf = KFold(n = 178 ,n_folds=5, shuffle=True, random_state=42)
        kn = KNeighborsClassifier(n_neighbors=i)
        s = cross_validation.cross_val_score(X = b, y=classes, estimator=kn, cv=kf ,scoring='accuracy')
        avg=s.sum()/len(s)
        a[i]=avg
    return
    
task(dict_, features)
sorted_ = sorted(dict_.items(), key=operator.itemgetter(1)) #тут находим максимальную точность и количетсов соседей

# 2 часть задания с нормализацией

x_features = preprocessing.scale(features)

dict2_ = {}
task(dict2_, x_features)

sorted_normilize = sorted(dict2_.items(), key=operator.itemgetter(1)) #тут находим максимальную точность и количетсов соседей после нормализации