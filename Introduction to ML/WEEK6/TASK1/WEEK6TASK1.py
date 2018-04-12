from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import pylab
import numpy as np
import copy
#from skimage.measure import compare_psnr
from math import log10

image = imread('parrots.jpg')
pylab.imshow(image)

image = img_as_float(image)
X=np.reshape(image,(337962,3))
X_orig = copy.deepcopy(X)

#/////////////////////////////////////////////////////////////////////////

for n_of_clusters in range(1,3):
    X = copy.deepcopy(X_orig)
    clf=KMeans(init='k-means++', n_clusters=n_of_clusters, random_state=241)
    clf.fit(X)
    X_pred=clf.predict(X)


    R = {}
    G = {}
    B = {}

    for j in range(0, n_of_clusters):
        R[j] = [0,0]
        G[j] = [0,0]
        B[j] = [0,0]

    for i in range(0,337962):
        clusternum = X_pred[i]
        R_sum = R[clusternum][0]
        G_sum = G[clusternum][0]
        B_sum = B[clusternum][0]
    
        R_sum = R_sum + X[i][0]
        G_sum = G_sum + X[i][1]
        B_sum = B_sum + X[i][2]
    
        R_num = R[clusternum][1]
        G_num = G[clusternum][1]
        B_num = B[clusternum][1]
    
        R_num = R_num + 1
        G_num = G_num + 1
        B_num = B_num + 1
    
        R[clusternum] = [R_sum, R_num]
        G[clusternum] = [G_sum, G_num]
        B[clusternum] = [B_sum, B_num]

    for j in range(0, n_of_clusters):
        avgR = R[j][0]/R[j][1]
        R[j] = [R[j][0], R[j][1], avgR]
    
        avgG = G[j][0]/G[j][1]
        G[j] = [G[j][0], G[j][1], avgG]
    
        avgB = B[j][0]/B[j][1]
        B[j] = [B[j][0], B[j][1], avgB]
    

    for i in range(0,337962):
        clusternum = X_pred[i]
        X[i][0] = R[clusternum][2]
        X[i][1] = G[clusternum][2]
        X[i][2] = B[clusternum][2]
    
    new_image = np.reshape(X,(474,713,3))
    pylab.figure()
    pylab.imshow(new_image)

    #print(compare_psnr(X_orig, X))
    
    sum1 = 0
    sum_R = 0
    sum_G = 0
    sum_B = 0
    for i in range(0,337962):
        R_Xo = X_orig[i][0]
        G_Xo = X_orig[i][1]
        B_Xo = X_orig[i][2]
        
        R_X = X[i][0]
        G_X = X[i][1]
        B_X = X[i][2] 
        
        sum1 = sum1 + (abs(R_Xo-R_X)+abs(G_Xo-G_X)+(abs(B_Xo-B_X)))**2


    
    MAX = 2**(n_of_clusters)-1
    
    MSE = (sum1)/(3*474*713)
    PSNR = 20*log10(MAX/MSE)
    
    print(PSNR)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


