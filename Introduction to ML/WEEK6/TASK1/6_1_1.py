# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:02:29 2016

@author: Danil
"""

from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import pylab
import numpy as np
import copy
from skimage.measure import compare_psnr



image = imread('parrots.jpg')
pylab.imshow(image)

image = img_as_float(image)
X=np.reshape(image,(337962,3))
X_orig=copy.deepcopy(X)
Otvet=np.zeros(20)
for i in range(1,20):
    print(i)
    clf=KMeans(init='k-means++', n_clusters=i, random_state=241)
    clf.fit(X)
    X_pred=clf.predict(X)
    R = np.zeros(i)
    G = np.zeros(i)
    B = np.zeros(i)
    for klast in range(0,i):
        for j in range(0,337962):
            if X_pred[j]==klast:
                R[klast]=R[klast]+X[j][0]
                G[klast]=G[klast]+X[j][1]
                B[klast]=B[klast]+X[j][2]
                R[klast]=R[klast]/337962
                G[klast]=G[klast]/337962
                B[klast]=B[klast]/337962
    
    for klast in range(0,8):
        for j in range(0,337962):
            if X_pred[j]==klast:
                X[j][0]=R[klast]*5
                X[j][1]=G[klast]*5
                X[j][2]=B[klast]*5
                
    new_image = np.reshape(X,(474,713,3))
#    pylab.imshow(new_image)
    Otvet[i]=compare_psnr(X_orig, X)
