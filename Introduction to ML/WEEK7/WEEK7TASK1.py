import pandas
import time
import datetime
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold

#ЗАДАНИЕ 1=====================================================================
features = pandas.read_csv('./features.csv', index_col='match_id') #загрузили таблицу

target = features.radiant_win.values #заранее выделяемый целевой вектор

del features['duration']  #удалили столбцы, связанные с итогами
del features['radiant_win']
del features['tower_status_radiant']
del features['tower_status_dire']
del features['barracks_status_dire']
del features['barracks_status_radiant']


#ЗАДАНИЕ 2=====================================================================
missing_data_dict = {} #создаем словарь, куда запишем столбцы с пропусками
for j in features: 
    if features.count()[j] != len(features):
        missing_data_dict[j] = len(features)-features.count()[j] #в missing_data_dict видим названия столбцов и с количество пропусков
        
#Пропусков не так уж и много. Больше всего пропусков в колонке "второй игрок, причастный к событию первая кровь", что наверное можно
#объяснить отсутсвием асситирующего игрока при первом убийстве, тоесть игрок сделал ферст блад в соло.
#На втором месте по пропускам расположена колонка "время приобретения предмета flying_courier", в этом случае пропуски обьясняются
#отсутвием закупа куры по причине нескилованности тимы, отсутсвием голды на неё у игроков или просто она была куплена после 5 минуты игры.
        
        
#ЗАДАНИЕ 3=====================================================================
features.fillna(value = 0, inplace = True) #заполняем пропуски нулями


#ЗАДАНИЕ 4=====================================================================
#столбец radiant_win содержит целевую перменную(??????????????????)


#ЗАДАНИЕ 5=====================================================================

a = [10, 20, 30, 50, 100, 200]
quality = {}

start_time = datetime.datetime.now()
for T in a:
    print(T)
    clf = GradientBoostingClassifier(n_estimators=T,max_depth=4, random_state=241)
    k_fold = KFold(n = len(features) ,n_folds=5, shuffle=True, random_state=241)
    s = cross_validation.cross_val_score(X = features, y=target , estimator=clf, cv=k_fold, scoring = 'roc_auc')
    avg=s.sum()/len(s)
    quality[T] = avg

print ('Time elapsed:', datetime.datetime.now() - start_time)


#кросс-валидация для градиентного бустинга с 30 деревьями проводилась 1мин 20 сек при качестве 0,689.

#Да, имеет смысл использовать больше 30 деревьев. Чтобы ускорить кросс-валидацию можно поиграться с рпараметрами:
#learning_rate, max_depth, max_features или использовать для кросс-валидация только подмножество






















