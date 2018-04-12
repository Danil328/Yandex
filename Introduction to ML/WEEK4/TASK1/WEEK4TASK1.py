import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

# сначала обучим
data = pandas.read_csv('salary-train.csv')

data.LocationNormalized = data.LocationNormalized.str.lower()
data.FullDescription = data.FullDescription.str.lower() #тексты к нижнему регистру 

data = data.replace('[^a-zA-Z0-9]', ' ', regex = True) #все, кроме букв и цифр, на пробелы

vectorizer = TfidfVectorizer(min_df=5)
Tf = vectorizer.fit_transform(data.FullDescription) #тексты в векторы признаков

data.LocationNormalized.fillna('nan', inplace=True)
data.ContractTime.fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

fin_matrix = hstack([X_train_categ,Tf])


clf = Ridge(alpha=1.0, random_state=241)
clf.fit(fin_matrix, data.SalaryNormalized)

# теперь тестируем на тренировочной выборке


data_train = pandas.read_csv('salary-test-mini.csv')

data_train.LocationNormalized = data_train.LocationNormalized.str.lower()
data_train.FullDescription = data_train.FullDescription.str.lower()

data_train = data_train.replace('[^a-zA-Z0-9]', ' ', regex = True)

Tf2 = vectorizer.transform(data_train.FullDescription)

X_train_categ2 = enc.transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

fin_matrix2 = hstack([X_train_categ2,Tf2])

answer = clf.predict(fin_matrix2)

# ответ: 56555.62 37188.32