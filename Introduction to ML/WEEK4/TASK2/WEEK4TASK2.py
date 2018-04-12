import pandas
from sklearn.decomposition import PCA
from numpy import corrcoef

data = pandas.read_csv('close_prices.csv')


features = data[['AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','T','TRV','UNH','UTX','V','VZ','WMT','XOM']].as_matrix()

pca = PCA(n_components=10)
pca.fit(features) # обучили

print(pca.explained_variance_ratio_) # ответ: 92.77 при 4

transformed_features = pca.transform(features) # применили обучение, выделили 10 главных признаков
new_transformed_features = pandas.DataFrame(data = transformed_features, columns = ['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth'])

djindx = pandas.read_csv('djia_index.csv')

dgi = djindx['^DJI'].values
first = new_transformed_features['first'].values

pirscorr = corrcoef(first, dgi) # ответ: 0.91


aaa = pca.components_ # строчка 0 соответсвует 1 компоненте, двигаясь по ней находи максимальное значение 0,588, что соответсвует компанни "V"
print(pca.components_)







