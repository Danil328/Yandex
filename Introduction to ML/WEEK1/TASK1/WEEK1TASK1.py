import pandas
import operator

data = pandas.read_csv ('titanic.csv', index_col = 'PassengerId')


# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
male = data.Sex.value_counts().get_value('male')
female = data.Sex.value_counts().get_value('female')
print(str(male) + ' ' + str(female))

# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
dead = data.Survived.value_counts().get_value(0)
survived = data.Survived.value_counts().get_value(1)
per = survived / (survived + dead)*100
print(round(per, 2))

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен)
all_class = data.count().get_value('Pclass')
first_class = data.Pclass.value_counts().get_value(1)
per = first_class/all_class*100
print(round(per, 2))

# Какого возраста были пассажиры?
# Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
mean = round(data.Age.mean(), 2)
median = round(data.Age.median(), 2)
print(str(mean) + ' ' + str(median))

# Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
corr = data.SibSp.corr(data.Parch, method = 'pearson')
print(corr)

# Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
# Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и
# выработать правило для извлечения имен, а также разделения их на женские и мужские.
i = 0
src = data.Name.get_values()
names = []
for row in src:
    is_woman = ("Mrs." in row) or ("Miss." in row) 
    if is_woman:
        braceIndex = row.find("(")
        if braceIndex != -1:
            row = row[braceIndex + 1:]
            index_space = row.find(" ")
            row = row[:index_space]
            names.append(row)
        else:
            index_point = row.find(".")
            row = row[index_point + 2:]
            index_space = row.find(" ")
            row = row[:index_space]
            names.append(row)
        i = i + 1

dict_ = {}
for name in names:
    value = dict_.get(name)
    if value is None:
        dict_[name] = 1
    else:
        dict_[name] = value + 1
sorted_x = sorted(dict_.items(), key=operator.itemgetter(1))
print(sorted_x)
   






















