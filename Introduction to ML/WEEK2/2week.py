# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:44:02 2016

@author: Danil
"""

import pandas

df=pandas.read_csv('wine.data')
dffs=pandas.read_csv('wine.data')
df.head()
#Масштабирование
from sklearn import preprocessing

df.alcohol=preprocessing.normalize(df.alcohol).transpose()
df.apple=preprocessing.normalize(df.apple).transpose()
df.Yasen=preprocessing.normalize(df.Yasen).transpose()
df.sheloch=preprocessing.normalize(df.sheloch).transpose()
df.magniy=preprocessing.normalize(df.magniy).transpose()
df.phenol=preprocessing.normalize(df.phenol).transpose()
df.flavonoid=preprocessing.normalize(df.flavonoid).transpose()
df.fenol=preprocessing.normalize(df.fenol).transpose()
df.proanathocyanins=preprocessing.normalize(df.proanathocyanins).transpose()
df.intensivnost=preprocessing.normalize(df.intensivnost).transpose()
df.ottenok=preprocessing.normalize(df.ottenok).transpose()
df.od280=preprocessing.normalize(df.od280).transpose()
df.proline=preprocessing.normalize(df.proline).transpose()

import numpy as np
def test_and_train(df, proportion):
    mask = np.random.rand(len(df)) < proportion
    return df[mask], df[~mask]
train, test = test_and_train(df, 0.67)

from math import sqrt
def euclidean_distance(instance1,instance2):
    squares = [(i-j)**2 for i,j in zip(instance1,instance2)]
    return sqrt(sum(squares))

import operator
def get_neighbours(instance, train,k):
    distances = []
    for i in train.ix[:,:-1].values:
        distances.append(euclidean_distance(instance,i))
    distances = tuple(zip(distances, train[u'Sortv'].values))
    return sorted(distances,key=operator.itemgetter(0))[:k]

from collections import Counter
def get_response(neigbours):
    return Counter(neigbours).most_common()[0][0][1]

def get_predictions(train, test, k):
    predictions = []
    for i in test.ix[:,:-1].values:
        neigbours = get_neighbours(i,train,k)
        response = get_response(neigbours)
        predictions.append(response)
    return predictions
#

def mean(instance):
    return sum(instance)/len(instance)
def get_accuracy(test,predictions):
    return mean([i == j for i,j in zip(test[u'Sortv'].values, predictions)])
get_accuracy(test,get_predictions(train, test, 5))

import pylab as pl
from sklearn.neighbors import KNeighborsClassifierr
variables = [u'alcohol',u'apple',u'Yasen',u'sheloch',u'magniy',u'fenol',u'flavonoid',u'fenol',u'proanathocyanins',u'intensivnost',u'ottenok',u'od280',u'proline']
results = []    
for n in range(1,51):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[variables], train[u'Sortv'])
    preds = clf.predict(test[variables])
    accuracy = np.where(preds==test[u'Sortv'], 1, 0).sum() / float(len(test))
    print ("Neighbors: %d, Accuracy: %3f" % (n, accuracy))
    results.append([n, accuracy])
results = pandas.DataFrame(results, columns=["n", "accuracy"])
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

#11111111111111
maxk=1
maxtest=0
count=0
for i in results.accuracy:
    count=count+1
    if i>maxtest:
        maxtest=i;
        maxk=count;
    
f1=open('1.txt','w')
f1.write(str(maxk))
f1.close()

maxa=max(results.accuracy)
f2=open('2.txt','w')
f2.write(str(maxa))
f2.close()

f3=open('3.txt','w')
f3.write(str(maxk))
f3.close()

maxa=max(results.accuracy)
f4=open('4.txt','w')
f4.write(str(maxa))
f4.close()