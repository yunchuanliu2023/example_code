from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')


# import rpy2.robjects as robjects

# res = robjects.StrVector(['abc', 'def'])
# print(res.r_repr())
# res = robjects.IntVector([1, 2, 3])
# print(res.r_repr())
# res = robjects.FloatVector([1.1, 2.2, 3.3])
# print(res.r_repr())

# robjects.r('v=c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6)') #在R空间中生成向量v
# m = robjects.r('matrix(v, nrow = 2)') #在R空间中生成matrix，并返回给python中的对象m
# print(m)




import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import pandas as pd
#直接声明，data frame强制转为DataFrame
pandas2ri.activate()

#R代码
r_code = """
library(tcherry)
# car <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
           # header = FALSE, sep = ",", dec = ".")
          
# print(car)
# print(mode(car))
# encoding="UTF-8"
car <-read.csv("data/18_tr.csv" ,quote = "",header = TRUE)
# print(car)
print(mode(car))
print(head(car))
# names(car) <- c("buying", "maint", "doors", "persons", "lug_boot",
                   # "safety", "class")
tch3 <- k_tcherry_p_lookahead(data = car, k = 3, p = 2, smooth = 0.001)

print(tch3$adj_matrix)
tch3$adj_matrix
"""



# r_df = ro.DataFrame({'int_values': ro.IntVector([1,2,3]),
                     # 'str_values': ro.StrVector(['abc', 'def', 'ghi'])})
# with localconverter(ro.default_converter + pandas2ri.converter):
    # r_from_pd_df = ro.conversion.rpy2py(r_df)
# print(r_from_pd_df.head())
  
#运行R代码
# with localconverter(ro.default_converter + pandas2ri.converter):
# robjects.r(r_code)
df = pd.DataFrame(ro.conversion.rpy2py(robjects.r((r_code))))
df =df.values
print(df)
print(df.shape)

list_a =[]
list_b =[]

dt = pd.read_csv("data/18_tr.csv")
dt2 = pd.read_csv("data/label_tr.csv")
dt["real"] = dt2
print(dt.head())
cc = dt.columns.values
for j in range(df.shape[1]):
    for i in range(j):   
        if(df[i,j]==1):
            list_a.append(j)
            list_b.append(i)
        
# print(type(cc[list_a[i]]))
ll = []
for i in range(len(list_a)):
    ll.append((cc[list_a[i]],cc[list_b[i]]))
    
for i in range(df.shape[1]-1):
    ll.append(("real",cc[i]))
    
from pgmpy.base import DAG
tan_structure = DAG()
tan_structure.add_nodes_from(nodes=cc)


tan_structure.add_edges_from(ebunch=ll)

print(tan_structure.edges)
print(tan_structure.nodes)


from pgmpy.models import NaiveBayes, BayesianModel
from pgmpy.inference import VariableElimination

from estimators import TreeAugmentedNaiveBayesSearch, BNAugmentedNaiveBayesSearch, ForestAugmentedNaiveBayesSearch
import fun
import networkx as nx
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tan_model = BayesianModel(tan_structure.edges)
# nx.draw_circular(tan_model, with_labels=True, arrowsize=20, node_size=1000, font_size=10, alpha=0.8)
# plt.show()



tree_check = tan_structure.copy()
# tree_check.remove_node("LR")
if nx.is_tree(tree_check):
    print('The TAN model, without the target variable, is a tree.')
else:
    print('The TAN model, without the target variable, is not a tree.')
nx.draw_circular(tree_check, with_labels=True, arrowsize=20, node_size=1000, font_size=10, alpha=0.8)
plt.show()

tan_model.fit(dt)
tan_cpds = tan_model.get_cpds()
if not tan_model.check_model():
    print('The TAN model has errors.')
else:
    print('The TAN model has no errors.')
    
    
te = pd.read_csv("data/18_test.csv")
te2 = pd.read_csv("data/label_test.csv")
target_variable  ="real"
te[target_variable] = te2

test_no_target = te.copy().drop(target_variable, axis=1).reset_index()
tan_inference = VariableElimination(tan_model)
print('The TAN model has no errors.')

tan_results = fun.predict(test_no_target, tan_inference, target_variable )
tan_mean_results = {}
print('The TAN model has no errors.')
for k, v in tan_results.items():
    # print(k,v)
    tan_mean_results[k] = np.nanmean(v)
print(tan_mean_results)
print(type(tan_results))
"""
"""