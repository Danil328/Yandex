import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv ('titanic.csv', index_col = 'PassengerId')

k = 0
for i in data.Age:
    k += 1
    if np.isnan(i) == True:
        data.drop(k, axis = 0, inplace = True);

target = data['Survived'].values

del(data['Survived'])
del(data['Name'])
del(data['SibSp'])
del(data['Parch'])
del(data['Ticket'])
del(data['Cabin'])
del(data['Embarked'])


##

male = data.Sex == 'male'
data['Sex'][male] = 1            
            
female = data.Sex == 'female'
data['Sex'][female] = 0      


clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, target)    
        
importances = clf.feature_importances_  #массив "важности"


