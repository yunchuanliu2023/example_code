

# import matplotlib.pyplot as plt
# import tensorflow as tf
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

def label_acc():
    gen(rate,ii)
    df1 = pd.read_csv("random/labeled_"+str(ii)+"_"+str(rate)+"%.csv")
    df3 = pd.read_csv("random/unlabeled_"+str(ii)+"_"+str(rate)+"%.csv")
    y1 = df1.pop("label") 
    y3 = df3.pop("label").values

    df1 = pd.concat([df1,df3])
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    

    y2= pd.read_csv("predd.csv")
    y2 = y2["0"].values
    
    # print(y1.shape)
    # print(y2.shape)
    y = np.append(y2,y3,axis =0)

    X_train = df1
    y_train = y 

    df2 = pd.read_csv("data/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")
    
    X_train = df1
    X_val = df2
    
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y2,y3)))
    # print('Macro Precision: {:.2f}'.format(precision_score(y2,y3, average='macro')))
    # print('Macro Recall: {:.2f}'.format(recall_score(y2,y3, average='macro')))
    # print('Macro F1-score: {:.2f}\n'.format(f1_score(y2,y3, average='macro'))) 
    return round(accuracy_score(y2,y3),3)
    
def buji(listA,listB):
    return list(set(listB).difference(set(listA)))


def get_nosiy_label2():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y =top()

    X_train = train_x.iloc[:,:18]
    X_test = test_x.iloc[:,:18]
    print(X_test.shape)
    # X_train = train_x.iloc[:,18:36]
    # X_test = test_x.iloc[:,18:36]

    y_test = test_y
    y_train = train_y
    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
       
        test_classifiers = [
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["v_"+classifier] = train_out
            te["v_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)

        # df1 = pd.read_csv("data/wk_1_tr.csv")
        # df2 = pd.read_csv("data/wk_1_test.csv")
        # print(df1.shape)
        # print(df2.shape)
        # ll = tr.columns.values.tolist()
        # for k in ll[1:]:
        
            # df1[k] = tr[k]
            # df2[k] = te[k]
        tr.pop("real")
        te.pop("real")
        tr.to_csv(path+"wk_1_tr2.csv",index = None)
        te.to_csv(path+"wk_1_test2.csv",index = None)




def get_nosiy_label3():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()  

    # X_train = train_x.iloc[:,:18]
    # X_test = test_x.iloc[:,:18]
    # print(X_test.shape)
    X_train = train_x.iloc[:,18:36]
    X_test = test_x.iloc[:,18:36]

    y_test = test_y
    y_train = train_y
    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
        test_classifiers = [
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["i_"+classifier] = train_out
            te["i_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)

        # df1 = pd.read_csv("data/wk_1_tr.csv")
        # df2 = pd.read_csv("data/wk_1_test.csv")
        # print(df1.shape)
        # print(df2.shape)
        # ll = tr.columns.values.tolist()
        # for k in ll[1:]:
        
            # df1[k] = tr[k]
            # df2[k] = te[k]
        tr.pop("real")
        te.pop("real")
        tr.to_csv(path+"wk_1_tr3.csv",index = None)
        te.to_csv(path+"wk_1_test3.csv",index = None)

def get_nosiy_label4():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()

    # X_train = train_x.iloc[:,:18]
    # X_test = test_x.iloc[:,:18]
    # print(X_test.shape)
    X_train = train_x.iloc[:,36:]
    X_test = test_x.iloc[:,36:]

    y_test = test_y
    y_train = train_y
    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
        test_classifiers = [
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["rof_"+classifier] = train_out
            te["rof_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)

        # df1 = pd.read_csv("data/wk_1_tr.csv")
        # df2 = pd.read_csv("data/wk_1_test.csv")
        # print(df1.shape)
        # print(df2.shape)
        # ll = tr.columns.values.tolist()
        # for k in ll[1:]:
        
            # df1[k] = tr[k]
            # df2[k] = te[k]
        tr.pop("real")
        te.pop("real")
        tr.to_csv(path+"wk_1_tr4.csv",index = None)
        te.to_csv(path+"wk_1_test4.csv",index = None)

def snoit():    

        
    df = pd.read_csv(path+"wk_1_tr2.csv")
    for i in range(3,5):
        df2 = pd.read_csv(path+"wk_1_tr"+str(i)+".csv")
        df  = pd.concat([df, df2], axis=1)
    tr = df.values
    
    df = pd.read_csv(path+"wk_1_test2.csv")
    for i in range(3,5):
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

def top():    

    df = pd.read_csv("data/x1.csv")
    # df2 = pd.read_csv("data/x2.csv")

    ll = [19,15,27,39,74,24,11,10,13,52,62]
    df1 = df[df["season"]==11]
   
    
    df1 = df1[df1['No'].isin(ll)]
    l2 = df1.index.tolist()
    ll = [11,19,20,188,204,53]
    df1 = df[df["season"]==1]
    df1 = df1[df1['No'].isin(ll)]
   
    ll2 = df1.index.tolist()
    l2 = bingji(l2,ll2)
    l3 = df.index.tolist()
    l4 = buji(l2,l3)

    
    df.pop("word")
    df.pop("word2")
    df.pop("season")
    df.pop("No")
    
    X_train = df.iloc[l2]
    X_test = df.iloc[l4]
    y_train = X_train.pop("label")
    y_test = X_test.pop("label")

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    return X_train,y_train,X_test,y_test
    

def gen():
    get_nosiy_label2()
    get_nosiy_label3()
    get_nosiy_label4()
    snoit()

def top2():    

    df = pd.read_csv("data/x1.csv")
    

    ll = [19,15,27,39,74,24,11,10,13,52,62]
    df1 = df[df["season"]==11]
   
    
    df1 = df1[df1['No'].isin(ll)]
    l2 = df1.index.tolist()
    ll = [11,19,20,188,204,53]
    df1 = df[df["season"]==1]
    df1 = df1[df1['No'].isin(ll)]
   
    ll2 = df1.index.tolist()
    l2 = bingji(l2,ll2)
    l3 = df.index.tolist()
    l4 = buji(l2,l3)

    df.pop("word")
    df.pop("word2")
    df.pop("season")
    df.pop("No")
    
    df1 = df.iloc[l2]
    y1 = df1.pop("label").values
    df3 = df.iloc[l4]
    df3.pop("label")
    y2= pd.read_csv("predd.csv")
    y2 = y2["0"].values
    y_train = np.append(y1,y2,axis =0)
    df1 = pd.concat([df1,df3])

    

    df2 = pd.read_csv("data/x2.csv")
    y_val = df2.pop("label")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")
    
    X_train = df1
    X_val = df2
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    return X_train,y_train,X_val,y_val



def run():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  
    gen()
    # train_x, train_y,test_x, test_y =datapick2.top()   
    train_x, train_y,test_x, test_y  = top2()  

    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
    print(X_train.head())
    print(X_train.columns.tolist())
    
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
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    

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
        print('\n testing error of ' + str(i)) 
        print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

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
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'),recall_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='macro')
        

def main():
    s1 = timeit.default_timer()  
    # top()
    run()
    rev_res()
    replot()
    s2 = timeit.default_timer()
    #running time
    print('Time(sec): ', (s2 - s1) )
if __name__ == '__main__':  

    main()

