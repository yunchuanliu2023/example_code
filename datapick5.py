

import numpy as np
import math
import pandas as pd

import timeit

# dc = pd.read_csv("des.csv")
# a =  list(dc.columns)
# print(type(a))
# print((a))

# dc = pd.read_csv("des.csv").values

def rd_data():
    path = "data/"
    k =1 
    df = pd.read_csv(path+"Ss"+str(k)+".csv")
    for k in range(2,14):
        df2 = pd.read_csv(path+"Ss"+str(k)+".csv")    
        df = pd.concat([df,df2])
    df.pop("word")
    df.pop("word2")
    df.pop("season")
    
    y = df.pop("label")
    df.describe().to_csv("des.csv")
    # print()
    
def OR_rule(num,j):
    if num <dc[4][j]:
        return 0
    elif num <dc[5][j]:
        return 1
    elif num <dc[6][j]:
        return 2
    elif num >=dc[6][j]:
        return 3
        
def eval(labels_true, labels):

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Rand Index: %0.3f" % metrics.rand_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print("Fowlkes-Mallows scores: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))

def rd_rof(ii):
    path1 = "data/"
    df  = pd.read_csv(path1 +'Ss'+str(ii)+'.csv')
    full_ind = df.index

    dg = pd.read_csv(path1 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==7)]
    ll = dg["new"].values.tolist()
    # print(df.shape[0])
    st = []
    
    dt2 = df.loc[df["word"].str.contains("Line_Trip|Line_Lightning")==True ].index
    dt4 = df.loc[df["word"].str.contains("Transformer")==True ].index
    dt3 = df.loc[df["word"].str.contains("Transformer_Trip|Transformer_Lightning|Transformer_Planned")].index
    dt1 = df.loc[df.word2.isin(ll)].index
    ind1 = bingji(dt2,dt4)
    ind1 = bingji(dt1,ind1)
    # print(len(dt2))

    # print("  "+str(len(dt1)))
    # print(len(ind1))
    #step1 get rid of the data
    ind3 = list(set(full_ind)^set(ind1))
    ind2 = bingji(dt3,ind3)
    # print(len(ind2))


    #step2 label data again 
    
    dg = pd.read_csv(path1 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    dt2 = df.loc[df["word"].str.contains("Trans")==True ].index
    
    
    dg = pd.read_csv(path1 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==5)]
    ll = dg["new"].values.tolist()
    dt3 = df.loc[df.word2.isin(ll)].index
    dt4 = df.loc[df["word"].str.contains("Freq")==True ].index    
    
    
    df.label = 0

    df.label[dt2] = 1
    df.label[dt1] = 1
    
    
    df.label[dt3] = 2
    df.label[dt4] = 2
    
    res = df
    # res = df.iloc[ind2]
    res.pop("word")
    res.pop("word2")
    res.pop("season")
    res.pop("No")
    y = res.pop("label")
    res = res.iloc[:,]
    res = res.replace([np.inf, -np.inf], np.nan)
    ind = res[res.isnull().T.any()].index
    res.loc[ind,:] = -1
    return res,y


def rd_all():

    k =1 
    dt,y = rd_rof(k)
    for k in range(2,14):
        dt2,y = rd_rof(k)   
        dt = pd.concat([dt,dt2])
    # dt.pop("word")
    # dt.pop("word2")
    # dt.pop("season")
    
    return  dt

def diff(listA,listB):
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    
        
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

def bingji(listA,listB):
    return list(set(listA).union(set(listB)))
    
    
def part(k):
    path ="data/"
    dt = pd.read_csv(path+"Ss"+str(k)+".csv")
    print(dt.shape)
    st1= []
    col_name = []
    for j in range(1,55):
        st2=[]
        for i in range(dt.shape[0]):
            st2.append(OR_rule(dt.loc[i,a[j]],j))
        st1.append(st2)
        col_name.append(a[j])
    st1 =np.array(st1)
    tt =pd.DataFrame(st1)
    tt = tt.transpose()
    return tt
    
def after_PCA():
    ll = []
    ll.append(0)
    df = part(1)
    temp = df.shape[0]
    ll.append(temp)
    for j in range(2,14):
        df2 = part(j)
        # temp = df.shape[0]
        df = pd.concat([df,df2])
        temp2 = df.shape[0]
        ll.append(df.shape[0])
    print(ll)
    df= df.fillna(-1)
    from sklearn.decomposition import PCA
    X_reduced = PCA(n_components=8).fit_transform(df)
    
    col_name = ["a","b","c","d","e","f","g","h"]
    for k in range(1,14):
        tt = X_reduced[ll[k-1]:ll[k]]
        tt = pd.DataFrame(tt)
        tt.columns = col_name
        tt["label"] = rd_rof(k)
        
        tt.to_csv("data/after_LF_"+str(k)+".csv",index = 0)
        
    
    
    
def rd_static(k):    

    # k = 2
    path ="data/"
    dt = pd.read_csv(path+"Ss"+str(k)+".csv")
    print(dt.shape)
    st1= []
    col_name = []
    for j in range(1,19):
        st2=[]
        for i in range(dt.shape[0]):
            st2.append(OR_rule(dt.loc[i,a[j]],j))
        st1.append(st2)
        col_name.append(a[j])
    st1 =np.array(st1)
    tt =pd.DataFrame(st1)
    tt = tt.transpose()
    # print(tt.head())
    # print(tt.shape)
    # print(col_name)
    tt.columns = col_name

    tt["label"] = rd_rof(k)
    tt.to_csv("data/after_LF_"+str(k)+".csv",index = 0)
    
    # print(a[1])
    # print(df[4][1])
    # print(df[5][1])
    # print(df[6][1])

    # print(dt.loc[1,a[1]])

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics

def change(x):
    if x.mean_v_dn<=0 and x.min_v_dn>0.01 and min_v_dup<=0:
        return 1
    elif x.mean_v_dn>0 and x.mean_v_adn >  0.14:
        return 2
    else:
    
        return 0
    
def compute_impurity(feature, impurity_criterion):
    """
    0: entropy
    1: gini
    This function calculates impurity of a feature.
    Supported impurity criteria: 'entropy', 'gini'
    input: feature (this needs to be a Pandas series)
    output: feature impurity
    """
    probs = feature.value_counts(normalize=True)
    
    if impurity_criterion == 0:
        impurity = -1 * np.sum(np.log2(probs) * probs)
    elif impurity_criterion == 1:
        impurity = 1 - np.sum(np.square(probs))
    else:
        raise ValueError('Unknown impurity criterion')
        
    return(round(impurity, 3))
    
def fun2(v1, v2,v3,v4):
    
    # print(df.head())

    temp = df.values
    st =[]
    for i in range(temp.shape[0]):
        if temp[i,4]<=v3 and temp[i,5]>v1 and temp[i,8]<=v4:
            st.append(1)
        elif temp[i,4]>v3 and temp[i,16]> v2:
            st.append(2)    
        else:
            st.append(0)      
    temp = pd.DataFrame(st)
    # df["esti"] = st

    a = compute_impurity(temp, 0)
    b = compute_impurity(temp, 1)
    c = metrics.mutual_info_score
    print(a,b)
    return (b)
    
    # print(df.head())
    # X_reduced = df[["esti","mean_v_dn","min_v_dn","mean_v_adn"]]
    
    
from bayes_opt import BayesianOptimization   
# import sys
# sys.setrecursionlimit(10000)
# bayesian-optimization 
def ck():
    s1 = timeit.default_timer()  
    global df,real
    df = rd_all()

    # Bounded region of parameter space
    pbounds2 = {'v1': (0.01, 0.5),
                'v2': (0.01, 0.5),
                'v3': (0, 0.1),
                'v4': (0, 1)
                }
                

    optimizer = BayesianOptimization(
        f=fun2,
        pbounds=pbounds2,
        random_state= 62,
    )
    optimizer.maximize(
        init_points=20,
        n_iter=280
    )
    tt = str(optimizer.max)
    with open('30.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    print(tt)        
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))


def run():
    df,real = rd_rof(1)
    temp = df.values
    st =[]
    v1 = 0.424
    v2 = 0.119
    
    for i in range(temp.shape[0]):
        if temp[i,4]<=0 and temp[i,5]>v1 and temp[i,8]<=0:
        # if temp[i,4]<=0 and temp[i,5]>0.01 and temp[i,8]<=0:
            st.append(1)
        elif temp[i,4]>0 and temp[i,16]> v2:
        # elif temp[i,4]>0 and temp[i,16]>  0.14:
            st.append(2)
        else:
            st.append(0)   
    model = kmm(X_reduced)
    # Prediction on the entire data
    all_predictions = model.predict(X_reduced)


    print(all_predictions[:20])

    X_reduced = df[["esti","mean_v_dn","mean_v_adn"]]
    model = kmm(X_reduced)
    all_predictions = model.predict(X_reduced)
    print(all_predictions[:20])
    
    
    X_reduced = df[["esti","mean_v_dn","min_v_dn"]]
    model = kmm(X_reduced)
    all_predictions = model.predict(X_reduced)
    print(all_predictions[:20])

    X_reduced = df[["esti","max_r_dup",	"mean_r_dup","min_r_dup","max_r_ddn","mean_r_ddn","min_r_ddn","max_r_aup"]]
    
    X_reduced = df[["esti","mean_v_dn","min_v_dn"]]
    model = kmm(X_reduced)
    all_predictions = model.predict(X_reduced)
    print(all_predictions[:20])
    print(real[:20].values)
    
def kmm(X_reduced):
    

    
    # print(y.shape)
    model = KMeans(n_clusters=3)

    # Fitting Model
    model.fit(X_reduced)
    return model
    
    
def main():
    s1 = timeit.default_timer() 
    ck()
    # run()

    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
