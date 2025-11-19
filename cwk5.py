

import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt
from scipy.linalg import svd 
# import time
import timeit
from random import randint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# python wk1_test.py 2>&1 | tee bb.log

import sys

    

    
def top():    


    df1 = pd.read_csv("data/x1.csv")
    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    
    y_train = df1.pop("label").values
    X_train = df1.values
    print(y_train[:500])
    return X_train,y_train
    


def kmean():
    X,y= top()
    from sklearn.cluster import KMeans
    print(X.shape)
    # print(X[:1])
    n_clusters=2
    cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
    y_pred = cluster.labels_#获取训练后对象的每个样本的标签    
    for n in range(0,4):
        print("N = "+str(n))
        print(y_pred[n*500:(n+1)*500])
        print(y[n*500:(n+1)*500])
        
        
    print()
    
    # centroid = cluster.cluster_centers_
    # color=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    # fig, axi1=plt.subplots(1)
    # for i in range(n_clusters):
        # axi1.scatter(X[y_pred==i, 0], X[y_pred==i, 1],
                   # marker='o',
                   # s=8,
                   # c=color[i])
    # axi1.scatter(centroid[:,0],centroid[:,1],marker='x',s=100,c='black')



    # plt.show()


def find_common_index():
 
    rate=7
    ii = 11
    a = np.load("random/labeled_"+str(ii)+"_"+str(rate)+"%.npy")
    for ii in range(12,21):    

        b = np.load("random/labeled_"+str(ii)+"_"+str(rate)+"%.npy")
        a = np.append(a,b)
    
    unique,cnt = np.unique(a,return_counts = True)
    c = dict(zip(unique,cnt))
    d5 = list(filter(lambda x: c[x] > 2, c))
    print(type(c))
    print(len(d5),d5)
    df1 = pd.read_csv("random/x1.csv")
    
    ind = d5 
    all_ind = df1.index.tolist()
    res = list( set(all_ind).difference(set(ind)))
    X_train = df1.iloc[ind]
    X_val = df1.iloc[res]
    
    t1 = X_train[X_train["label"] == 0]
    t2 = X_train[X_train["label"] == 1]
    t3 = X_train[X_train["label"] == 2]
    
    print(t1.shape[0],format(100*t1.shape[0]/X_train.shape[0],'.1f'))
    print(t2.shape[0],format(100*t2.shape[0]/X_train.shape[0],'.1f'))
    print(t3.shape[0],format(100*t3.shape[0]/X_train.shape[0],'.1f'))     
    return d5

def find_common_bad():
 
    # rate= [5,5,5,8,8,9,9]
    # num = [9,6,4,5,10,3,6]
    rate= [5,5,5,5,5]
    num = [1,5,12,16,19]
    a = np.load("random/labeled_"+str(num[0])+"_"+str(rate[0])+"%.npy")
    for ii in range(1,len(rate)):    

        b = np.load("random/labeled_"+str(num[ii])+"_"+str(rate[ii])+"%.npy")
        a = np.append(a,b)
    
    unique,cnt = np.unique(a,return_counts = True)
    c = dict(zip(unique,cnt))
    d5 = list(filter(lambda x: c[x] > 1, c))
    print(type(c))
    print(len(d5),d5)
    return d5
    # df1 = pd.read_csv("random/x1.csv")
    
    # ind = d5 
    # all_ind = df1.index.tolist()
    # res = list( set(all_ind).difference(set(ind)))
    # X_train = df1.iloc[ind]
    # X_val = df1.iloc[res]
    
    # t1 = X_train[X_train["label"] == 0]
    # t2 = X_train[X_train["label"] == 1]
    # t3 = X_train[X_train["label"] == 2]
    
    # print(t1.shape[0],format(100*t1.shape[0]/X_train.shape[0],'.1f'))
    # print(t2.shape[0],format(100*t2.shape[0]/X_train.shape[0],'.1f'))
    # print(t3.shape[0],format(100*t3.shape[0]/X_train.shape[0],'.1f'))     
    
def main():
    s1 = timeit.default_timer()
    L1 = find_common_index()
    L2 = find_common_bad()
    res = list( set(L1).intersection(set(L2)))
    path2 = "random/"
    np.save(path2+"bad_2.npy", res)
    print(res)
    # top()
    
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

