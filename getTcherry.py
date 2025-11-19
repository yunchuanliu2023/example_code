
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
from operator import truediv

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

def top2():    
    


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

    
    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)
    
    return X_train,y_train,X_val,y_val





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
        print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

        print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))


    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'),recall_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='macro')
    
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
    dt2 = df2[df2["real"]!=df2[feature]].index
    dt3 = jiaoji(dt1,dt2)
    
    df2.loc[dt3,"real"] = df2.loc[dt3,feature] 
    df2.to_csv("te3.csv",index = None)
    
feature= "RF"
def run_Tcherry():
    train_x, train_y,test_x, test_y = top2() 
    
    print("round"+str(ii))
    train_x = pd.concat([train_x,test_x])
    X_test = pd.read_csv("random/x2.csv")
    X_train = train_x
    y_test = X_test.pop("label") 
    X_test.pop("word")
    X_test.pop("word2")
    X_test.pop("season")
    X_test.pop("No")    
    

    
    
    # df1 = pd.read_csv("saveT/"+str(rate)+"-"+str(ii)+".csv")
    df1 = pd.read_csv(str(knum)+"th/"+str(rate)+"-"+str(ii)+".csv")
    # df1 = pd.read_csv("pred.csv")
    df1 = df1["real"].values

    dt = np.concatenate((train_y, df1), axis=0)


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
        
 
                      # 'KNN':knn_classifier,  
                       # 'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                     # 'GBDT':gradient_boosting_classifier  
        }  

        tr = pd.DataFrame(train_y)
        te = pd.DataFrame(y_test)
        te.columns = ['real']

        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            # print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](train_x, dt)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            # train_out = model.predict(X_train)  
            
            # tr[classifier] = train_out
            te[classifier] = predict

        
        te.to_csv("te2.csv",index = None)
        # print("result save done")





    te = pd.DataFrame(y_test)
    te.columns = ['real']
    te[feature] = predict
    te.to_csv("te2.csv",index = None)


def eval_Tcherry_label():


    train_x, train_y,test_x, test_y = top2() 

    df1 = pd.read_csv(str(knum)+"th/"+str(rate)+"-"+str(ii)+".csv")
    df1 =df1["real"].values
    matrix=confusion_matrix(test_y, df1)
    print(matrix)
    list_diag = np.diag(matrix)
    list_raw_sum = np.sum(matrix,axis =1)
    each_acc = np.nan_to_num(truediv(list_diag,list_raw_sum))
    print(each_acc)
    # print("round"+str(ii))
    # print('Accuracy: {:.2f}'.format(accuracy_score(df1, test_y)))
    # print("\n")
    return accuracy_score(df1, test_y),each_acc[0],each_acc[1],each_acc[2]

def packthem():
    df = pd.read_csv("Rate_10_3_th_Tcherry.csv")
    for j in range(1,17):
        rate = 10+5*j
        df2 = pd.read_csv("Rate_"+str(rate)+"_3_th_Tcherry.csv")
        
        df[str(rate)] = df2["acc"]
    df.to_csv("3th_Tcherry.csv",index = None)
    
def main():
    s1 = timeit.default_timer()  

    

    global rate,ii,knum
    
    knum = 5
    # ll = [10,20,30,40,50,60,70,80,90]
    ll =[10]
    for rate in ll:
    
        acc =[]
        pre=[]
        rec =[]
        F1=[] 
        lacc=[]
        a1=[]
        a2=[]
        a3=[]
        
        for ii in range(1,11):  
        # for ii in range(1,2):  
            e,f,g,h = eval_Tcherry_label()
            run_Tcherry()
            rev_res()
            a,b,c,d=replot()        
            
            acc.append(a*100)
            pre.append(b*100)
            rec.append(c*100)
            F1.append(d*100)
            lacc.append(e*100)
            a1.append(f*100)
            a2.append(g*100)
            a3.append(h*100)
            
        df= pd.DataFrame(acc)
        df.columns = ['acc']
        df["pre"] = pre
        df["rec"] = rec
        df["F1"] = F1
        df["lacc"] = lacc
        df["a1"] = a1
        df["a2"] = a2
        df["a3"] = a3
        df.to_csv("Rate_"+str(rate)+"_"+str(knum)+"th_Tcherry.csv",index = None)
        


    s2 = timeit.default_timer()

    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()    
    # packthem()
    
