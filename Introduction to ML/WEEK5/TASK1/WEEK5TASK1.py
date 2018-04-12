import pandas 
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score

data = pandas.read_csv('abalone.csv')

F = data.Sex == 'F'
data['Sex'][F] = -1

M = data.Sex == 'M'
data.Sex[data.Sex == 'M'] = 0

I = data.Sex == 'I'
data.Sex[data.Sex == 'I'] = 1

target = data.Rings.values
features = data[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']].as_matrix()

dict_ = {}

for T in range(1, 51):
    print(T)
    clf = RandomForestRegressor(random_state=1, n_estimators=T)
    k_fold = KFold(n = 4177 ,n_folds=5, shuffle=True, random_state=1)
    s = cross_validation.cross_val_score(X = features, y=target , estimator=clf, cv=k_fold, scoring = 'r2')
    avg=s.sum()/len(s)
    dict_[T] = avg


# ответ 23, так как 0,52< начинается с 23 деревьев в лесу