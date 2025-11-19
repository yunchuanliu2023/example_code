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


# car <-read.csv("D:/work/py_file/week/data/18_tr.csv" ,quote = "",header = TRUE)

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
 var1 <- c(sample(c(1, 2), 50, replace = TRUE))
 var2 <- var1 + c(sample(c(1, 2), 50, replace = TRUE))
 var3 <- var1 + c(sample(c(0, 1), 50, replace = TRUE,
                         prob = c(0.9, 0.1)))
 var4 <- c(sample(c(1, 2), 50, replace = TRUE))


 data <- data.frame("var1" = as.character(var1),
                    "var2" = as.character(var2),
                    "var3" = as.character(var3),
                    "var4" = as.character(var4))

 adj_matrix_DAG <- matrix(c(0, 0, 0, 0,
                            1, 0, 0, 0,
                            1, 0, 0, 0,
                            0, 1, 0, 0),
                           nrow = 4)
rownames(adj_matrix_DAG) <- colnames(adj_matrix_DAG) <- names(data)

library(gRain)
print(CPT(adj_matrix_DAG, data))
CPTs <- compileCPT(CPT(adj_matrix_DAG, data))
G <- grain(CPTs)

querygrain(G, nodes = c("var1", "var2", "var3", "var4"), type = "joint")
sim <- simulate.grain(object = G, nsim = 100, seed = 43)
# tch2 <- k_tcherry_step(data = sim, k = 2, smooth = 0.001)
tch3_step <- k_tcherry_step(data = sim, k = 3, smooth = 0.001)
tch3_step$weight

# tch3_2_lookahead <- k_tcherry_p_lookahead(data = sim, k = 3, p = 2,
                                          # smooth = 0.001)
# tch3_2_lookahead$weight
# tch3_increase <- increase_order2(tch_cliq = tch2$cliques, data = sim, smooth = 0.001)
# tch3_increase$weight
# tch3_complete <- tcherry_complete_search(data = sim, k = 3, smooth = 0.001)
# tch3_complete$model$weight
# tch3_complete$n_models
# tch3_increse_complete <- 
# increase_order_complete_search(tch_cliq = tch2$cliques, data = sim, smooth = 0.001)
# tch3_increse_complete$model$weight

library(gRain)
graph_tch3_step <- as(tch3_step$adj_matrix, "graphNEL")
G_tch3_step <- grain(x = graph_tch3_step, data = sim, smooth = 0.001)

querygrain(G_tch3_step, nodes = c("var1", "var2", "var3", "var4"), type = "joint")


new_data <- data.frame("var1" =  rep(NA, 4),
                       "var2" = c(NA, "l1", "12","1"),
                       "var3" = c("l2", "l2", "l2","1"),
                        "var4" = c("l2", "l2", "l2","1")
                       )

predict(object = G_tch3_step, response = "var1", newdata = new_data)


          

           
"""

ro.conversion.rpy2py(robjects.r((r_code)))
# df = pd.DataFrame()
# df =df.values
# print(df)


# packageurl <- "https://cran.rstudio.com/bin/windows/contrib/3.4/gRain_1.3-0.zip"
# install.packages(packageurl, repos=NULL, type="source")