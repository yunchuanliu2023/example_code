from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
import timeit

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
# pandas2ri.activate()

#R代码
r_code = """


library(tcherry)
car <-read.csv("D:/work/py_file/week/data/18_tr_all.csv" ,quote = "",header = TRUE)
# str(car)

car$KNN = NULL
car$LR  = NULL
car$RF = NULL
car$GBDT = NULL

car$v_KNN = as.character(car$v_KNN)
car$v_LR  = as.character(car$v_LR)
car$v_RF = as.character(car$v_RF)
car$v_GBDT = as.character(car$v_GBDT)

car$i_KNN = as.character(car$i_KNN)
car$i_LR  = as.character(car$i_LR)
car$i_RF = as.character(car$i_RF)
car$i_GBDT = as.character(car$i_GBDT)

car$rof_KNN = as.character(car$rof_KNN)
car$rof_LR  = as.character(car$rof_LR)
car$rof_RF = as.character(car$rof_RF)
car$rof_GBDT = as.character(car$rof_GBDT)
car$real = as.character(car$real)




tch3 <- k_tcherry_p_lookahead(data = car[1:50], k = 3, p = 2, smooth = 0.001)
# tch3 <- increase_order_complete_search(ChowLiu_cliques, data,
                                       # smooth = 0.1))
adj_matrix_DAG<-tch3$adj_matrix
# adj_matrix_DAG



for (i in 1:13) {
    for (j in 1:i)
        adj_matrix_DAG[i,j]=0
}



# gg <- as(adj_matrix_DAG, "graphNEL")
# plot(gg)

rownames(adj_matrix_DAG) <- colnames(adj_matrix_DAG) <- names(car)

library(gRain)
print(CPT(adj_matrix_DAG, car,bayes_smooth =1))
CPTs <- compileCPT(CPT(adj_matrix_DAG, car,bayes_smooth =1))
G <- grain(CPTs)

querygrain(G, nodes = names(car), type = "joint")

sim <- simulate.grain(object = G, nsim = 100, seed = 43)


tch3_step <- k_tcherry_step(data = sim, k = 3, smooth = 0.001)
# tch3_step$weight



library(gRain)
graph_tch3_step <- as(tch3_step$adj_matrix, "graphNEL")
G_tch3_step <- grain(x = graph_tch3_step, data = sim, smooth = 0.001)
querygrain(G_tch3_step, nodes = names(car), type = "joint")


test <-read.csv("D:/work/py_file/week/data/18_test.csv" ,quote = "",header = TRUE)
# str(test)


test$KNN = NULL
test$LR  = NULL
test$RF = NULL
test$GBDT = NULL

test$v_KNN = as.character(test$v_KNN)
test$v_LR  = as.character(test$v_LR)
test$v_RF = as.character(test$v_RF)
test$v_GBDT = as.character(test$v_GBDT)

test$i_KNN = as.character(test$i_KNN)
test$i_LR  = as.character(test$i_LR)
test$i_RF = as.character(test$i_RF)
test$i_GBDT = as.character(test$i_GBDT)

test$rof_KNN = as.character(test$rof_KNN)
test$rof_LR  = as.character(test$rof_LR)
test$rof_RF = as.character(test$rof_RF)
test$rof_GBDT = as.character(test$rof_GBDT)
test$real = as.character(test$real)

pp<-predict(object = G_tch3_step, response = "real", newdata = test)

write.csv(x = pp,file = "D:/work/py_file/week/pred.csv",row.names = FALSE)

          

           
"""


s1 = timeit.default_timer()

ro.conversion.rpy2py(robjects.r((r_code)))

s2 = timeit.default_timer()
print('Time:(min) ', (s2 - s1)/60 )