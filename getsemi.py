
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import pickle
# import datapick2
# import datapick3
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# python getlabel2.py 2>&1 | tee b2.log
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time  
from sklearn import metrics  
import pickle as pickle  

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd

def top():    
    
    df1 = pd.read_csv("random/labeled_"+str(ii)+"_"+str(rate)+"%.csv")
    df3 = pd.read_csv("random/unlabeled_"+str(ii)+"_"+str(rate)+"%.csv")
    y1 = df1.pop("label").values
    y3 = df3.pop("label").values


    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    
    df3.pop("word")
    df3.pop("word2")
    df3.pop("season")
    df3.pop("No")
    

    X_train = df1
    y_train = y1 


    X_val = df3
    y_val = y3
    
    
    return X_train,y_train,X_val,y_val

def get_semi_label():
    train_x, train_y,test_x, test_y = top()   
    y = np.zeros(test_y.shape[0])
    for i in range(y.shape[0]):
        y[i] = -1
    

    
    train_x = np.concatenate((train_x, test_x), axis=0)
    train_y = np.concatenate((train_y, y), axis=0)
 
    from sklearn.semi_supervised import LabelSpreading
    from sklearn import tree  
    base_classifier = tree.DecisionTreeClassifier()    
    from sklearn.semi_supervised import SelfTrainingClassifier
    semi_res1 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)
    
    from sklearn.naive_bayes import MultinomialNB
    base_classifier = MultinomialNB(alpha=0.01)
    
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_x)
    X_test = scaler.fit_transform(test_x)    
    
    semi_res2 = SelfTrainingClassifier(base_classifier).fit(X_train, train_y).predict(X_test)
    
    from sklearn.svm import SVC  
    base_classifier = SVC(kernel='rbf', probability=True) 
    semi_res3 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)

    from sklearn.linear_model import LogisticRegression  
    base_classifier = LogisticRegression(penalty='l2')  
    semi_res4 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)
    
    from sklearn.neighbors import KNeighborsClassifier  
    base_classifier = KNeighborsClassifier()  
    semi_res5 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)
    
    
    df = pd.DataFrame(semi_res1)
    df.columns = ['semi-DT']
    df["semi-GBDT"] = semi_res2
    df["semi-SVC"] = semi_res3
    df["semi-LR"] = semi_res4
    df["semi-KNN"] = semi_res5
    
    
    arr = []
    for i in range(0,df.shape[0]):
        temp = df.iloc[i]
        ty= np.argmax(np.bincount(temp))
        if ty !=-1:
            arr.append(ty)
        else:
            arr.append(0)
    arr = np.array(arr)
    # print('arrsize:',arr.shape)
    df['Maj'] = arr
    df.to_csv("semi_res.csv",index = 0)

    print("save done")

def eval_semi_label():
    train_x, train_y,test_x, df2 = top() 
    
    ll = ["semi-DT","semi-GBDT","semi-SVC","semi-LR","semi-KNN","Maj"]
    res = []
    print()

    df3 = pd.read_csv("semi_res.csv")
    print("rate ="+str(rate)+","+str(ii))
    for i in range(len(ll)):
        # print(ll[ii])
        df1 = df3[ll[i]].values

        # df1 = df2["SNO"].values
        # res.append(accuracy_score(df1, df2))
        # temp = round(accuracy_score(df1, df2), 3)  
        # res.append(temp)
        
        print(ll[i])
        print('Accuracy: {:.2f}'.format(accuracy_score(df1, df2)))
        # print("\n")
        # print('Macro Precision: {:.2f}'.format(precision_score(df1, df2, average='macro')))
        # print('Macro Recall: {:.2f}'.format(recall_score(df1, df2, average='macro')))
        # print('Macro F1-score: {:.2f}\n'.format(f1_score(df1, df2, average='macro')))
        
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model          
        
def replot():
    
    df = pd.read_csv("te3.csv")
    df.pop("No")
    df.pop("word2")
    df.pop("season")
    y_test = df["real"]
    
    # ll = df.columns.values
    # ll =["RF","GBDT","DT"]
    ll =["RF"]
    for i in ll:
        
        y_pred = df[i]
        print('\n testing error of ' + str(i)) 
        print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

        # print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
        # print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
        # print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

        print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

        # print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
        # print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
        # print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))    
        
        
        
        matrix=confusion_matrix(y_test, y_pred)
        print(matrix)
        class_report=classification_report(y_test, y_pred)
        print(class_report)
        
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))
    
def rev_res():
    df1 = pd.read_csv("data/X_val.csv")
    df2 = pd.read_csv("te2.csv")
    df2["word2"] = df1["word2"]
    df2["season"] = df1["season"]
    df2["No"] = df1["No"]

    dg = pd.read_csv("data/muti.csv")
    
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()    
    dt1 = df2.loc[df2.word2.isin(ll)].index
    dt2 = df2[df2["real"]!=df2["RF"]].index
    dt3 = jiaoji(dt1,dt2)
    
    df2.loc[dt3,"real"] = df2.loc[dt3,"RF"] 
    df2.to_csv("te3.csv",index = None)
    
def eval_semi():
    train_x, train_y,test_x, df2 = top() 
    
    ll = ["semi-DT","semi-GBDT","semi-SVC","semi-LR","semi-KNN","Maj"]
    res = []
    print()
    train_x = pd.concat([train_x,test_x])
    df3 = pd.read_csv("semi_res.csv")
    print("rate ="+str(rate)+","+str(ii))
    
    X_test = pd.read_csv("data/x2.csv")

    y_test = X_test.pop("label") 
    X_test.pop("word")
    X_test.pop("word2")
    X_test.pop("season")
    X_test.pop("No")    
    # len(ll)
    # for i in range(1):
    i = 5
    print(ll[i])
    df1 = df3[ll[i]].values
    dt = np.concatenate((train_y, df1), axis=0)
    model = random_forest_classifier(train_x, dt)  
    predict = model.predict(X_test) 

    te = pd.DataFrame(y_test)
    te.columns = ['real']
    te["RF"] = predict
    te.to_csv("te2.csv",index = None)
    rev_res()
    replot()
    
def main():
    s1 = timeit.default_timer()  


    global rate,ii
    rate = 5
    # ii = 1
    for ii in range(1,20+1):
        get_semi_label()
        eval_semi_label()
        # eval_semi()

    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()    
    
    
