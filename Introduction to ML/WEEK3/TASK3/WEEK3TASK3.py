import pandas
from math import exp
from sklearn.metrics import roc_auc_score


data = pandas.read_csv('data-logistic.csv', names = ['Y', 'Xone', 'Xtwo'])

w1 = 0
w2 = 0

k = 0.1
l = len(data)
sum1 = 0
sum2 = 0
w1pred = 0
w2pred = 0
p = 1
chetcik = 0



while (p > 0.00001):
    sum1 = 0
    sum2 = 0
    chetcik += 1
    for i in range(0, 205):
        sum1 = sum1 + data.Y[i]*data.Xone[i]*(1-1/(1+exp(-data.Y[i]*(w1*data.Xone[i]+w2*data.Xtwo[i]))))
        sum2 = sum2 + data.Y[i]*data.Xtwo[i]*(1-1/(1+exp(-data.Y[i]*(w1*data.Xone[i]+w2*data.Xtwo[i]))))
    w1 = w1 + k*(1/l)*sum1
    w2 = w2 + k*(1/l)*sum2
    p = ((w1-w1pred)**2 + (w2-w2pred)**2)**0.5
    print(p)
    w1pred = w1
    w2pred = w2

    
A = [ 0 for i in range(205)]
for i in range(205):
    A[i] = 1/(1+exp(-w1*data.Xone[i]-w2*data.Xtwo[i]))    
    
rac = roc_auc_score(data.Y, A) #0.927

# /////////////////////////////////////////////////////////////////////////
w1_ = 0
w2_ = 0

k_ = 0.1
l_ = len(data)
sum1_ = 0
sum2_ = 0
w1pred_ = 0
w2pred_ = 0
p_ = 1
chetcik_ = 0

while (p_ > 0.00001):
    sum1_ = 0
    sum2_ = 0
    chetcik_ += 1
    for i in range(0, 205):
        sum1_ = sum1_ + data.Y[i]*data.Xone[i]*(1-1/(1+exp(-data.Y[i]*(w1_*data.Xone[i]+w2_*data.Xtwo[i])))) - k*10*w1_
        sum2_ = sum2_ + data.Y[i]*data.Xtwo[i]*(1-1/(1+exp(-data.Y[i]*(w1_*data.Xone[i]+w2_*data.Xtwo[i])))) - k*10*w2_
    w1_ = w1_ + k_*(1/l_)*sum1_
    w2_ = w2_ + k_*(1/l_)*sum2_
    p_ = ((w1_-w1pred_)**2 + (w2_-w2pred_)**2)**0.5
    print(p_)
    w1pred_ = w1_
    w2pred_ = w2_


B = [ 0 for i in range(205)]
for i in range(205):
    B[i] = 1/(1+exp(-w1_*data.Xone[i]-w2_*data.Xtwo[i]))    
    
rac1 = roc_auc_score(data.Y, B) #0.937



























