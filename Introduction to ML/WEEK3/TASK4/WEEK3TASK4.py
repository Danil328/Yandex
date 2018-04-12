import pandas 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

data = pandas.read_csv('classification.csv')


Tp = 0
for i in range(len(data)):
    if (data.true[i] == 1) and (data.pred[i] == 1):
        Tp += 1
    
Fp = 0
for i in range(len(data)):
    if (data.true[i] == 0) and (data.pred[i] == 1):
        Fp += 1
        
Fn = 0
for i in range(len(data)):
    if (data.true[i] == 1) and (data.pred[i] == 0):
        Fn += 1
        
Tn = 0
for i in range(len(data)):
    if (data.true[i] == 0) and (data.pred[i] == 0):
        Tn += 1

#ответ: 43 34 59 64
#/////////////////////////////////////////
Accuracy = accuracy_score(data.true, data.pred)
Precision = precision_score(data.true, data.pred)
Recall = recall_score(data.true, data.pred)
F = f1_score(data.true, data.pred)

#ответ: 0.54 0.56 0.42 0.48
#//////////////////////////////////////////

scores = pandas.read_csv('scores.csv')

logreg = roc_auc_score(scores.true, scores.score_logreg)
svm = roc_auc_score(scores.true, scores.score_svm)
knn = roc_auc_score(scores.true, scores.score_knn)
tree = roc_auc_score(scores.true, scores.score_tree)

#ответ: score_logreg
#//////////////////////////////////////////
logreg_curve = precision_recall_curve(scores.true, scores.score_logreg)
logregpd = pandas.DataFrame(data = list(zip(logreg_curve[0], logreg_curve[1])), columns=['precision','recall'])
A = [ 0 for i in range(198)]
for i in range(0, 198):
    if logregpd.recall[i] > 0.7:
        A[i] = logregpd.precision[i]
#0.57

svm_curve = precision_recall_curve(scores.true, scores.score_svm)
svmpd = pandas.DataFrame(data = list(zip(svm_curve[0], svm_curve[1])), columns=['precision','recall'])
B = [ 0 for i in range(200)]
for i in range(0, 200):
    if svmpd.recall[i] > 0.7:
        B[i] = svmpd.precision[i] 
#0.57
        

knn_curve = precision_recall_curve(scores.true, scores.score_knn)
knnpd = pandas.DataFrame(data = list(zip(knn_curve[0], knn_curve[1])), columns=['precision','recall'])
C = [ 0 for i in range(105)]
for i in range(0, 105):
    if knnpd.recall[i] > 0.7:
        C[i] = knnpd.precision[i] 
#0.61        
        
tree_curve = precision_recall_curve(scores.true, scores.score_tree)
treepd = pandas.DataFrame(data = list(zip(tree_curve[0], tree_curve[1])), columns=['precision','recall'])
D = [ 0 for i in range(13)]
for i in range(0, 13):
    if treepd.recall[i] > 0.7:
        D[i] = treepd.precision[i] 
#0.65

#ответ: score_tree





































