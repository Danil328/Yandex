import pandas 
import numpy as np
import scipy
from math import exp
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

gbm_data = pandas.read_csv('gbm-data.csv')

data = gbm_data.values

X = scipy.delete(data, 0, 1)  # выделили признаки
y = gbm_data.Activity.values  # выделили целево столбец (правильный ответы)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.8, random_state=241) # далее создали тестовую и тренировочныую выборки




clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = 0.2) # определили классификатор с 250 деревьями
clf.fit(X_train, y_train) # обучили его на тренировочной выборке                                        # learning_rate, вот тут фишка, поидее все надо сделать в
                                                                                                        # функциюю и далее все действия проедлывать для 1 0.5 0.3 0.2 0.1, 
                                                                                                        # но делаем изменения вручную, в консоли строятся графики  
test_loss = np.zeros(len(clf.estimators_))                                                              # для каждого из значений rate в том же порядке, далее анализируем их
train_loss = np.zeros(len(clf.estimators_))                                                             # ответ: overfitting, так как где-то после 50 итерации точность ухудшается
A = np.zeros(750)                                                                                       # для тестовой выборки
B = np.zeros(3001)

min_logloss = 0.7 #инициализируем минимальный лог_лосс больше 0.7 из ходя из графиков и номер его итерации (нужно для второго задания при learning_rate=0.2)
iter_of_minlogloss = 300

for i, pred in enumerate(clf.staged_decision_function(X_train)):
    for r in range(0,750): 
        A[r] = 1/(1+exp(-pred[r])) # для i итерации получили вектор ответов pred, далее преобразуем полученное предсказание с помощью сигмоидной функции 
    train_loss[i] = log_loss(y_train, A) # сравниваем с истинными ответам и получаем величину ошибки лог_лос (чем меньше, тем лучше)
    
for j, pred1 in enumerate(clf.staged_decision_function(X_test)):
    for t in range(0,3001):
        B[t] = 1/(1+exp(-pred1[t]))
    test_loss[j] = log_loss(y_test, B) 
    if test_loss[j]<=min_logloss: # нахождение минимального логлосса и его итерации (нужно для 2 задания при 0,2, в других случаях не обращать на него внимание)
        min_logloss = test_loss[j]
        iter_of_minlogloss = j  # ответ: 0.53 36

plt.figure()
plt.plot(test_loss, 'r', linewidth=2)
plt.plot(train_loss, 'g', linewidth=2)
plt.legend(['test', 'train'])


# далее 3 задание

rfc = RandomForestClassifier(random_state=241, n_estimators = 36) # определили классификатор с числом деревьев 36, так при этом количестве деревьев
rfc.fit(X_train, y_train)                                         # градиентный бустинг показал лучшую точность

y_pred = rfc.predict_proba(X_test) # нашли вектор принадлежности к классу

A = log_loss(y_test, y_pred) # нашли точность классификара
# ответ: 0.54, так как рандомный лес чутка уступает градиентному бустингу и поэтому велечина ошибок чутка выше (0,54 вместо 0,53)
























