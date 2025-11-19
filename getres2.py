

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
import sys

import warnings
warnings.filterwarnings("ignore")
import pickle
# import datapick2
# import datapick3
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import  precision_score, recall_score, f1_score
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



import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
from operator import truediv
# from pandas._testing import assert_frame_equal
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from rno_fun import data_lowfreq

    
    

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier  
def estimate():  
    
    
    train_x, train_y,test_x, test_y =datapick2.top()  
    print(train_x.shape) 
    model = RandomForestClassifier(n_estimators=100)  
    lsvc = model.fit(train_x,train_y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(train_x)
    print(X_new.shape)  

def topp():    

    df1 = pd.read_csv("data/x1.csv")
    df2 = pd.read_csv("data/x2.csv")
    
    y = df1.pop("label") 
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    

    X_train = df1
    y_train = y 

    y = df2.pop("label") 
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")
    
    X_test = df2
    y_test = y 
    return X_train,y_train,X_test,y_test

from pwk1 import gen

def buji(listA,listB):
    return list(set(listB).difference(set(listA)))




def snoit():    
    df = pd.read_csv(path+"wk_1_tr2.csv")
    for i in range(4,5):
        df2 = pd.read_csv(path+"wk_1_tr"+str(i)+".csv")
        df  = pd.concat([df, df2], axis=1)
    tr = df.values
    
    df = pd.read_csv(path+"wk_1_test2.csv")
    for i in range(4,5):
        df2 = pd.read_csv(path+"wk_1_test"+str(i)+".csv")
        df  = pd.concat([df, df2], axis=1)
 
    
    test = df.values
    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality= 3, verbose=True)
    label_model.fit(L_train=tr, n_epochs=500, log_freq=100, seed=123)


    pred = label_model.predict(test)
    pred = pd.DataFrame(pred)
    pred.to_csv('predd.csv',index=None)
    
def bingji(listA,listB):
    return list(set(listA).union(set(listB)))



def gen():

    snoit()


# For the labeling function learning
# X_train means labeled 
# X_val means unlabeled  
def top():    
    # rate=3
    # ii = 1
    ind = np.load("random/labeled_"+str(ii)+"_"+str(rate)+"%.npy")
    sub1 = np.load("random/bad_1.npy")
    sub2 = np.load("random/bad_2.npy")
    df1 = pd.read_csv("random/x1.csv")
    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    # jiandiao = list( set(sub1).union(set(sub2)))
    jiandiao = sub1
    all_ind = df1.index.tolist()
    res = list( set(all_ind).difference(set(ind)))
    ind = list( set(ind)-set(jiandiao))
    res = list( set(res)-set(jiandiao))
    X_train = df1.iloc[ind]
    X_val = df1.iloc[res]
    
    t1 = X_train[X_train["label"] == 0]
    t2 = X_train[X_train["label"] == 1]
    t3 = X_train[X_train["label"] == 2]
    
    print(t1.shape[0],format(100*t1.shape[0]/X_train.shape[0],'.1f'))
    print(t2.shape[0],format(100*t2.shape[0]/X_train.shape[0],'.1f'))
    print(t3.shape[0],format(100*t3.shape[0]/X_train.shape[0],'.1f'))
    
    X_train.loc[X_train["label"] == 2, "label"] = 0
     
    
    t1 = X_val[X_val["label"] == 0]
    t2 = X_val[X_val["label"] == 1]
    t3 = X_val[X_val["label"] == 2]
    
    print(t1.shape[0],format(100*t1.shape[0]/X_val.shape[0],'.1f'))
    print(t2.shape[0],format(100*t2.shape[0]/X_val.shape[0],'.1f'))
    print(t3.shape[0],format(100*t3.shape[0]/X_val.shape[0],'.1f'))
    X_val.loc[X_val["label"] == 2, "label"] = 0
    
    y_train = X_train.pop("label") 
    y_val = X_val.pop("label")

    # X_train = df1
    # X_val = df2
    # print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)
    return X_train,y_train,X_val,y_val
    
    
from sklearn.utils import shuffle

# For the testing 
# X_train means labeled +  estimated unlabeled  
# X_val means testing data

def top2(order):    

    X_train,y_train,X_val,y_val2 = top()
   
    X_train = pd.concat([X_train,X_val])

    y1 = y_train
    y2= pd.read_csv("tcresult/pred_"+str(order)+"th.csv")
    y2 = y2["real"].values
    
    # y2= pd.read_csv("exp/semi_res.csv")
    # y2 = y2["SelfTrain"].values
    
    y_train = np.append(y1,y2,axis =0)

    df2 = pd.read_csv("random/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")

    X_val = df2
    print(X_train.shape,y_train.shape,X_val.shape,y_val2.shape,y2.shape)

    return X_train,y_train,X_val,y_val,y_val2,y2

def top_semi():    

    X_train,y_train,X_val,y_val2 = top()
   
    X_train = pd.concat([X_train,X_val])

    y1 = y_train
    # y2= pd.read_csv("tcresult/pred_"+str(order)+"th.csv")
    # y2 = y2["real"].values
    
    y2= pd.read_csv("exp/semi_res.csv")
    y2 = y2["SelfTrain"].values
    
    y_train = np.append(y1,y2,axis =0)

    df2 = pd.read_csv("random/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")

    X_val = df2
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    return X_train,y_train,X_val,y_val,y_val2,y2

def top_semi2():    

    X_train,y_train,X_val,y_val2 = top()
   
    X_train = pd.concat([X_train,X_val])

    y1 = y_train
    # y2= pd.read_csv("tcresult/pred_"+str(order)+"th.csv")
    # y2 = y2["real"].values
    
    y2= pd.read_csv("exp/semi_res.csv")
    y2 = y2["LabelSpread"].values
    
    y_train = np.append(y1,y2,axis =0)

    df2 = pd.read_csv("random/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")

    X_val = df2
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    return X_train,y_train,X_val,y_val,y_val2,y2
    
def top_sno():    
    
    X_train,y_train,X_val,y_val2 = top()
   
    X_train = pd.concat([X_train,X_val])

    y1 = y_train
    y2= pd.read_csv("predd.csv")
    y2 = y2["0"].values
    
    # y2= pd.read_csv("exp/semi_res.csv")
    # y2 = y2["SelfTrain"].values
    
    y_train = np.append(y1,y2,axis =0)

    df2 = pd.read_csv("random/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")

    X_val = df2
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    return X_train,y_train,X_val,y_val,y_val2,y2

def run_sno():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  
    gen()
    # train_x, train_y,test_x, test_y =datapick2.top()   
    train_x, train_y,test_x, test_y,y_val2,y2  = top_sno()  

    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
    # print(X_train.head())
    # print(X_train.columns.tolist())
    
# def gg(): 

    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
        test_classifiers = [

             # 'KNN', 
             # 'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            # 'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      # 'KNN':knn_classifier,  
                       # 'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     # 'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            # print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict

        te.to_csv("te2.csv",index = None)
        
    y1 = y_val2
    matrix=confusion_matrix(y1,y2)
    word =["semi","","2 order","3 order","4 order"]
    print('\n SNO Accuracy: {:.3f}'.format(accuracy_score(y1,y2)))
    print(matrix)
    
    rev_res()
    replot()

    
def run(order):
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  
    # gen()
    # train_x, train_y,test_x, test_y =datapick2.top()   
    if order==0:
        train_x, train_y,test_x, test_y,y_val2,y2  = top_semi() 
        
    elif order==1:
        train_x, train_y,test_x, test_y,y_val2,y2  = top_semi2()  
    else:
        train_x, train_y,test_x, test_y,y_val2,y2  = top2(order) 
        
    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
    # print(X_train.head())
    # print(X_train.columns.tolist())
    
# def gg(): 

    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
        test_classifiers = [
            'RF', 
            ]  
            
            
        classifiers = {

                       'RF':random_forest_classifier,  

        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            # print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict

        te.to_csv("te2.csv",index = None)
        
    # _,_,_,y1 = top()
    y1 = y_val2
    matrix=confusion_matrix(y1,y2)
    word =["semi-SelfTrain","semi-LabelSpread","2 order","3 order","4 order"]
    print('\n'+word[order]+' Label Accuracy: {:.3f}'.format(accuracy_score(y1,y2)))
    
    print(matrix)
    list_diag = np.diag(matrix)
    list_raw_sum = np.sum(matrix,axis =1)
    each_acc = np.nan_to_num(truediv(list_diag,list_raw_sum))
    # print(accuracy_score(y1,y2),each_acc)
    
    # return accuracy_score(y1,y2) ,each_acc[0],each_acc[1],each_acc[2]

feature= "RF"
path = "exp/"
    
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

    
    # let real value change to the muti- labels
def rev_res():
    df1 = pd.read_csv("data/X_val.csv")
    df2 = pd.read_csv("te2.csv")
    df2["word2"] = df1["word2"]
    df2["season"] = df1["season"]
    df2["No"] = df1["No"]
    path2 = "data/"
    dg = pd.read_csv(path2 +"muti.csv")
    
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()    
    dt1 = df2.loc[df2.word2.isin(ll)].index

    dt2 = df2[df2["real"]!=df2[feature]].index
    dt3 = jiaoji(dt1,dt2)
    
    df2.loc[dt3,"real"] = df2.loc[dt3,feature] 
    
    df2.to_csv("te3.csv",index = None)

def replot():
    
    df = pd.read_csv("te3.csv")
    df.pop("No")
    df.pop("word2")
    df.pop("season")
    y_test = df["real"]
    
    # ll = df.columns.values
    # ll =["RF","GBDT","DT"]
    ll =[feature]
    for i in ll:
        
        y_pred = df[i]
        print('\n rate=' + str(rate)+' and ii =' + str(ii)) 
        print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

        print('Macro Precision: {:.3f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.3f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.3f}\n'.format(f1_score(y_test, y_pred, average='macro')))

        matrix=confusion_matrix(y_test, y_pred)
        print(matrix)
        class_report=classification_report(y_test, y_pred)
        print(class_report)
        
    # return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'),recall_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='macro')
        


def main():
    s1 = timeit.default_timer()  
    
    global rate,ii,jj
    rate =int(sys.argv[1])
    ii = int(sys.argv[2])
    jj = int(sys.argv[3])
    run(jj)
    rev_res()
    replot()
    
    
def gg():
    lab_acc =[]
    test_acc =[]
    
    for j in range(2,5):

        e,f,g,h = run(j)
        rev_res()
        a,b,c,d = replot()
        
        lab_acc.append(round(e,3))
        test_acc.append(round(a,3))
    s2 = timeit.default_timer()
    print('lab_acc ', lab_acc )
    print('test_acc ', test_acc )
    #running time
    print('Time(sec): ', (s2 - s1) )
    
    
if __name__ == '__main__':  

    main()

