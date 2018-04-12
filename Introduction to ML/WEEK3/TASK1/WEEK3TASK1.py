import pandas 
from sklearn.svm import SVC

svm_data = pandas.read_csv('svm-data.csv', names = ['TarCol', 'FirstFeatCol', 'SecondFeatCol'])

target = svm_data.TarCol.values
features = svm_data[['FirstFeatCol', 'SecondFeatCol']].as_matrix()

clf = SVC(C=100000, kernel='linear', random_state=241)
parametr=clf.get_params()
clf.fit(features, target) 

mass = clf.support_ # ОТВЕТ 4 5 10
