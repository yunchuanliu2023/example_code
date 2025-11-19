

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

def fix_data():
    df1,X_test=filter_data()
    df1.to_csv("data/x1.csv",index =0)
    X_test.to_csv("data/x2.csv",index =0)
    
    
def top2():    

    df1 = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")
    y = df1.pop("label") 

    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    

    
    a = int(0.05*y.shape[0])
    
    X_train = df1.iloc[:a] 
    y_train = y.iloc[:a]   
    X_val = df1.iloc[a:] 
    y_val = y.iloc[a:]   

    # print(X_train.shape)
    # print(X_val.shape)
    
    return X_train,y_train,X_val,y_val
    
def top():    
    # rate=3
    # ii = 1
    ind = np.load("random/labeled_"+str(ii)+"_"+str(rate)+"%.npy")

    df1 = pd.read_csv("random/x1.csv")
    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    
    all_ind = df1.index.tolist()
    res = list( set(all_ind).difference(set(ind)))
    X_train = df1.iloc[ind]
    X_val = df1.iloc[res]
    
    y_train = X_train.pop("label") 
    y_val = X_val.pop("label")

    
    # X_train = df1
    # X_val = df2
    # print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    
    return X_train,y_train,X_val,y_val
    
from sklearn.utils import shuffle

def filter_data(rate,i):
    # rate  = 10
    path1 = "data/"
    df2 = pd.read_csv(path1+"X_train.csv")
    
    
    
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
    print(len(tt))
    print(tt)
    np.save(path2+"labeled_"+str(i)+"_"+str(rate)+"%.npy", tt)
    
    # tt2 = X_test.index.tolist()

    # X_train.to_csv(path2+"labeled_"+str(i)+"_"+str(rate)+"%.csv",index =0)
    # X_test.to_csv(path2+"unlabeled_"+str(i)+"_"+str(rate)+"%.csv",index =0)

    
def static():
    path1 = "data/"
    df2 = pd.read_csv(path1+"X_train.csv")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")   
    
    df = df2[df2["label"] == 0]
    
    df.describe().to_csv("Line.csv")
    
    df = df2[df2["label"] == 1]
    
    df.describe().to_csv("Trans.csv")

    df = df2[df2["label"] == 2]
    
    df.describe().to_csv("Freq.csv")    
    
    # print(df2.describe())
    

  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
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

def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier()
    # model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500,max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10) 
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  


path = "exp/"
    
def get_nosiy_label():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   

    X_train = train_x
    X_test = test_x
    
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
            
            tr[classifier] = train_out
            te[classifier] = predict
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)
            
        tr.to_csv(path+"wk_1_tr.csv",index = None)
        
        te.to_csv(path+"wk_1_test.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    

def get_nosiy_label2():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   

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
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)

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
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)

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
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)

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
        
        
def get_semi_label():
    train_x, train_y,test_x, test_y = top()   
    y = np.zeros(test_y.shape[0])
    for i in range(y.shape[0]):
        y[i] = -1
    
    # print(train_x.shape)
    # print(train_y.shape)
    
    # print(test_x.shape)
    # print(y.shape)
    
    train_x = np.concatenate((train_x, test_x), axis=0)
    train_y = np.concatenate((train_y, y), axis=0)
    
    # print(train_x.shape)
    # print(train_y.shape)
    
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.semi_supervised import SelfTrainingClassifier
    base_classifier = RandomForestClassifier(n_estimators=100)  
    semi_res1 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)
    semi_res2 =  LabelSpreading().fit(train_x, train_y).predict(test_x)
    print(semi_res1.shape)
    print(semi_res2.shape)
    
    df = pd.DataFrame(semi_res1)
    df.columns = ['SelfTrain']
    df["LabelSpread"] = semi_res2
    df.to_csv("semi_res.csv",index = 0)
    
def packit():
    df = pd.read_csv(path+"wk_1_test.csv")
    for i in range(2,5):
        df2 = pd.read_csv(path+"wk_1_test"+str(i)+".csv")
        df  = pd.concat([df, df2], axis=1)
    df.to_csv(path+"test_all4.csv",index = None)
        
    df = pd.read_csv(path+"wk_1_tr.csv")
    for i in range(2,5):
        df2 = pd.read_csv(path+"wk_1_tr"+str(i)+".csv")
        df  = pd.concat([df, df2], axis=1)
    df.to_csv(path+"tr_all4.csv",index = None)
    
def generate_data():
    # for rate in range(0,3,0.5):
    # for j in range(10,100,5):
    for i in range(1,11):
        filter_data(5,i)

        
def main():
    s1 = timeit.default_timer()
    # generate_data()
    
    global rate,ii
    rate =int(sys.argv[1])

    ii = int(sys.argv[2])
    get_nosiy_label()
    get_nosiy_label2()
    get_nosiy_label3()
    get_nosiy_label4()
    
    # get_semi_label()
    packit()
    
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

