

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



    
from sklearn.utils import shuffle

def ramdon_split_tr():
    rate  = 80
    path1 = "data/"
    df1 = pd.read_csv(path1+"X_val.csv")
    df2 = pd.read_csv(path1+"X_train.csv")
    
    df2 = pd.concat([df1, df2], ignore_index=True)


    df4 = df2[df2["label"] == 0]
    df1 = df2[df2["label"] == 1]
    df5 = df2[df2["label"] == 2]
    
    r1 = df4.shape[0]/df2.shape[0]
    r2 = df1.shape[0]/df2.shape[0]
    r3 = df5.shape[0]/df2.shape[0]
    
    df4 = shuffle(df4)
    df1 = shuffle(df1)
    df5 = shuffle(df5)
    
    a1 = int(r1*0.01*rate*df2.shape[0])
    a2 = int(r2*0.01*rate*df2.shape[0])
    a3 = int(r3*0.01*rate*df2.shape[0])
    
    X_train = df4.iloc[:a1] 
    temp1 = df1.iloc[:a2]
    temp2 = df5.iloc[:a3]
    X_train  = pd.concat([X_train, temp1])
    X_train  = pd.concat([X_train, temp2])     
    
    X_test = df4.iloc[a1:] 
    temp1 = df1.iloc[a2:]
    temp2 = df5.iloc[a3:] 
    
    X_test  = pd.concat([X_test, temp1])
    X_test  = pd.concat([X_test, temp2]) 
    print(X_train.shape)
    
    
    # i = 1
    path2 = "random/"
    tt = X_train.index.tolist()

    X_train.to_csv(path2+"X_train.csv",index =0)
    X_test.to_csv(path2+"X_val.csv",index =0)

    



    
def generate_data():
    for rate in range(1,10):
    # for j in range(10,100,5):
        for i in range(1,11):
            filter_data(rate,i)

        
def main():
    s1 = timeit.default_timer()

    ramdon_split_tr()
    
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

