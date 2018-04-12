import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = pandas.read_csv('perceptron-train.csv', names = ['TarCol','FirstFeatCol','SecondFeatCol'])
test = pandas.read_csv('perceptron-test.csv', names = ['TarCol','FirstFeatCol','SecondFeatCol'])

target_train = train.TarCol.values
features_train = train[['FirstFeatCol','SecondFeatCol']].as_matrix()

target_test = test.TarCol.values
features_test = test[['FirstFeatCol','SecondFeatCol']].as_matrix()

clf = Perceptron(random_state=241)
clf.fit(features_train, target_train) # здесь произошло обучение персептрона

predictions = clf.predict(features_test) # прогнозы для тестовой выборке

accuracy_for_test = accuracy_score(target_test, predictions) # получаем точность прогноза на тестовой выборке

# дальше все тоже самое для нормализованныхвыборок

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(features_train)
X_test_scaled = scaler.transform(features_test)

clf1 = Perceptron(random_state=241)
clf1.fit(X_train_scaled, target_train)

predictions1 = clf1.predict(X_test_scaled)

accuracy_for_test_scaled = accuracy_score(target_test, predictions1)


diff = accuracy_for_test_scaled-accuracy_for_test
