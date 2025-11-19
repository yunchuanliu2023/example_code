# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  

# python wk2.py 2>&1 | tee sno.log
def get_sno_label():
    tr = pd.read_csv('data/wk_1_tr.csv')
    test = pd.read_csv('data/wk_1_test.csv')
    tr.pop("real")
    test.pop("real")
    
    tr2 = pd.read_csv('data/wk_1_tr2.csv')
    test2 = pd.read_csv('data/wk_1_test2.csv')  


    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)
    
    
    tr2 = pd.read_csv('data/wk_1_tr3.csv')
    test2 = pd.read_csv('data/wk_1_test3.csv')  


    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)


    tr2 = pd.read_csv('data/wk_1_tr4.csv')
    test2 = pd.read_csv('data/wk_1_test4.csv')  


    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)



    # tr.pop("LR")
    # test.pop("LR")
    
    # tr.pop("v_LR")
    # test.pop("v_LR")

    # tr.pop("GBDT")
    # test.pop("GBDT")
    
    # tr.pop("v_GBDT")
    # test.pop("v_GBDT")

    # tr.pop("i_LR")
    # test.pop("i_LR")    
    
    
    print(tr.shape)
    print(test.shape)
    y_tr = tr.pop("real")
    y_test = test.pop("real")
    
    
    # ll = tr.columns.tolist()
    # print(ll)
    # for i in range(len(ll)):
        # tr[ll[i]] = tr[ll[i]].apply(str)
        # test[ll[i]] = test[ll[i]].apply(str)
        
    tr.to_csv("data/18_tr.csv",quoting = 1,index= 0)
    test.to_csv("data/18_test.csv",quoting = 1,index= 0)   
    
    y_tr.to_csv("data/label_tr.csv",index= 0)
    y_test.to_csv("data/label_test.csv",index= 0)
    
# def cc():
    
    # tr.pop('real')
    # test.pop('real')
    # test.pop('RF')
    tr = tr.values
    test = test.values






    # arr = []
    # for i in range(0,test.shape[0]):
        # temp = test[i]
        # arr.append(np.argmax(np.bincount(temp)))

    # arr = np.array(arr)
    # print('arrsize:',arr.shape)
    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality= 3, verbose=True)
    label_model.fit(L_train=tr, n_epochs=500, log_freq=100, seed=123)


    pred = label_model.predict(test)
    tt = pd.read_csv('data/wk_1_test.csv')
    jj = tt.pop('real')
    y_test = jj.values
    # label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")[
        # "accuracy"]
    # print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
# def cc():

    # tt = pd.read_csv('data/wk_1_test.csv')
    # test.pop('LR')
    # jj = tt.pop('real')
    # print(test.head())

    # model = random_forest_classifier(tr, y_tr) 
    # print(tr.shape)
    # print(test.shape)
    

    # rf_pred = model.predict(test)  
    
    tt['SNO'] = pred
    # tt['Maj'] = arr
    # tt['rf_pred'] = rf_pred
    tt['real'] = jj
    semi = pd.read_csv('semi_res.csv')
    # tt['SelfTrain'] = semi['SelfTrain']
    # tt['LabelSpread'] = semi['LabelSpread']
    tt.to_csv('wk_2_test.csv',index = None)

def esti():
    
    df = pd.read_csv('wk_2_test.csv')
    y_test = df["real"]

    ll = df.columns.values
    # ll =["RF","GBDT"]
    for i in range(0,df.shape[1]-1):
        
        y_pred = df[ll[i]]
        print('\n testing error of ' + str(ll[i])) 
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

def esti2():

    df = pd.read_csv('wk_2_test.csv')
    df2 = pd.read_csv('pred.csv')
    y_test = df["real"]

    ll = df.columns.values

    y_pred = df2["real"]
    print('\n testing error of T-cherry' )
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))


    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    
    
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)

# get_sno_label()
esti()
esti2()
# print('wk_2_test update done')



