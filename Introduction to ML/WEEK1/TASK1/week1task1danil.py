# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
data=pandas.read_csv('titanic.csv', index_col='PassengerId')
data.head()
z1m=data.Sex[data.Sex=='male'].count()
z1f=data.Sex[data.Sex=='female'].count()
print (str(z1m)+' '+str(z1f))

#1 
f1=open('1.txt','w')
f1.write(str(z1m)+' '+str(z1f))
f1.close()

sur=data['Survived'].value_counts(1)
print ('Удалось выжить: '+str(round(sur[1],2)))

#2
f2 = open('2.txt','w')
f2.write(str(round(sur[1]*100,2)))
f2.close()

clas1=data.Pclass[data.Pclass==1].count()
numberofpeople=len(data)
clas1per=clas1/float(numberofpeople)*100
print ('Пассажиры первого класса: '+str(round(clas1per,2)))

#3
f3=open('3.txt','w')
f3.write(str(round(clas1per,2)))
f3.close()

agemed=data.Age.median()
ageavr=data.Age.mean()
print (str(round(ageavr,2))+' '+str(agemed))
#4
f4=open('4.txt','w')
f4.write(str(round(ageavr,2))+' '+str(agemed))
f4.close()
#корреляция
pirs=data['SibSp'].corr(data['Parch'], method='pearson')
print ("Корреляция: "+str(round(pirs,2)))
#5
f5=open('5.txt','w')
f5.write(str(round(pirs,2)))
f5.close();
#Популярное женское имя
def searchfist(s):
    indexf=2;
    for bookva in s:
        if bookva == ".":
            return indexf
        indexf=indexf+1

def searchlast(s,first):
    indexl=first;
    while s[indexl]!=" " and indexl!=len(s)-1:
        indexl=indexl+1
    return indexl
#1. Создадим массив с женскими именами
popname=[];
popnameN=[];
proverka=[];
for i in range(220):
    popnameN.append(1)
    
count=0
for name in data.Name:
    first=0
    last=0
    good=1
    count=count+1;
    if name.find("Mrs")!=-1 or name.find("Miss")!=-1:
        first=searchfist(name)
        last=searchlast(name,first)
        name=name[first:last]
        counte=0
        for element in popname:
           if element==name:
               good=0
               popnameN[counte]=popnameN[counte]+1
           counte=counte+1
        if good==1:
            popname.append(name)
    proverka.append(name)
maxn=1  
count=0  
for namenumb in popnameN:
    count=count+1
    if namenumb>maxn:
        maxn=namenumb
        maxN=count-1
        
print (popname[maxN])

#6
f6=open('6.txt','w')
f6.write(popname[maxN])
f6.close()

        
        