from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
import timeit





import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import pandas as pd


#R代码
r_code = """


library(tcherry)
car <-read.csv("D:/work/py_file/week/data/18_tr_all.csv" ,encoding = "UTF-8",quote = "",header = TRUE)


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
print(str(car))
tch3 <- k_tcherry_p_lookahead(data = car[1:50], k = 3, p = 2, smooth = 0.001)


"""

s1 = timeit.default_timer()

ro.conversion.rpy2py(robjects.r((r_code)))

s2 = timeit.default_timer()
print('Time:(min) ', (s2 - s1)/60 )