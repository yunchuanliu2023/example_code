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

          
set.seed(94)
tch_random <- random_tcherry(n = 6, n_levels = rep(2, 6))
# print(tch_random)
library(gRain)
CPTs <- compileCPT(tch_random$CPTs)
G <- grain(CPTs)
querygrain(G, nodes = c("V1", "V2"), type = "joint", evidence = list("V3" = "l1"))
sim <- simulate.grain(object = G, nsim = 100, seed = 43)
tch3_step <- k_tcherry_step(data = sim, k = 3, smooth = 0.001)


graph_tch3_step <- as(tch3_step$adj_matrix, "graphNEL")
G_tch3_step <- grain(x = graph_tch3_step, data = sim, smooth = 0.001)
querygrain(G_tch3_step, nodes = c("V1", "V2"), type = "joint",
           evidence = list("V3" = "l1"))
new_data <- data.frame("V1" = rep(NA, 3),
                       "V2" = c(NA, "l1", "l2"),
                       "V3" = c("l2", "l2", "l2"),
                       "V4" = c("l1", NA, NA),
                       "V5" = c("l1", NA, "l1"),
                       "V6" = c(NA, NA, "l2"))
print(G_tch3_step)                       
# print(new_data)
print(sessionInfo())         
predict(object = G_tch3_step, response = "V1", newdata = new_data)
predict(object = G_tch3_step, response = "V1", newdata = new_data,
              type = "distribution")
set.seed(43)
var1 <- c(sample(c(1, 2), 100, replace = TRUE))
var2 <- var1 + c(sample(c(1, 2), 100, replace = TRUE))
var3 <- var1 + c(sample(c(0, 1), 100, replace = TRUE,
                        prob = c(0.9, 0.1)))
var4 <- c(sample(c(1, 2), 100, replace = TRUE))
var5 <- var2 + var3
var6 <- var1 - var4 + c(sample(c(1, 2), 100, replace = TRUE))
var7 <- c(sample(c(1, 2), 100, replace = TRUE))
data <- data.frame("var1" = as.character(var1),
                   "var2" = as.character(var2),
                   "var3" = as.character(var3),
                   "var4" = as.character(var4),
                   "var5" = as.character(var5),
                   "var6" = as.character(var6),
                   "var7" = as.character(var7))
tch3 <- k_tcherry_step(data = data, k = 3, smooth = 0.001)
thinned <- thin_edges(tch3$cliques, tch3$separators, data, smooth = 0.001)
thinned$n_edges_removed
library(gRain)
library(Rgraphviz)
tch3_graph <- as(tch3$adj_matrix, "graphNEL")
thinned_graph <- as(thinned$adj_matrix, "graphNEL")
par(mfrow = c(1, 2))
# plot(tch3_graph)
# plot(thinned_graph)                   

           
"""

ro.conversion.rpy2py(robjects.r((r_code)))
# df = pd.DataFrame()
# df =df.values
# print(df)


# packageurl <- "https://cran.rstudio.com/bin/windows/contrib/3.4/gRain_1.3-0.zip"
# install.packages(packageurl, repos=NULL, type="source")