from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import KFold
import numpy as np

import heapq



newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
kk=vectorizer.fit_transform(newsgroups.data) #числовое представление текстовых данных

grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(len(newsgroups.data), n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(kk, newsgroups.target)

#for a in gs.grid_scores_:
#    a.parameters   #находим минимальный из лучших параметр C=1 (долгая операция)
    
clf2 = SVC(C=1, kernel='linear', random_state=241)
clf2.fit(kk, newsgroups.target)  #обучение с новым параметром

coef = np.abs(clf2.coef_.data)
top10 = heapq.nlargest(10,range(len(coef)),coef.take)
top10_fullpath = clf2.coef_.indices[top10]


feature_mapping = vectorizer.get_feature_names()


dict_ = {}
for i in range(0,10):
    dict_[top10_fullpath[i]] = feature_mapping[top10_fullpath[i]]