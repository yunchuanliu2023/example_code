


t1<-Sys.time()
library(tcherry)
car <-read.csv("D:/work/py_file/week/exp/tr_all.csv" ,quote = "",header = TRUE)
# str(car)

knum = 4

car$KNN = as.character(car$KNN)
car$LR  = as.character(car$LR)
car$RF = as.character(car$RF)
car$GBDT = as.character(car$GBDT)

car$v_KNN = NULL
car$v_LR  = NULL
car$v_RF = NULL
car$v_GBDT = NULL

#car$v_KNN = as.character(car$v_KNN)
#car$v_LR  = as.character(car$v_LR)
#car$v_RF = as.character(car$v_RF)
#car$v_GBDT = as.character(car$v_GBDT)

car$i_KNN = NULL
car$i_LR  = NULL
car$i_RF = NULL
car$i_GBDT = NULL

#car$i_KNN = as.character(car$i_KNN)
#car$i_LR  = as.character(car$i_LR)
#car$i_RF = as.character(car$i_RF)
#car$i_GBDT = as.character(car$i_GBDT)


car$rof_KNN = as.character(car$rof_KNN)
car$rof_LR  = as.character(car$rof_LR)
car$rof_RF = as.character(car$rof_RF)
car$rof_GBDT = as.character(car$rof_GBDT)
car$real = as.character(car$real)

# car$real = NULL


tch3 <- k_tcherry_p_lookahead(data = car, k = knum, p = 3, smooth = 0.001)
# tch3 <- increase_order_complete_search(ChowLiu_cliques, data,
# smooth = 0.1))
adj_matrix_DAG<-tch3$adj_matrix
# adj_matrix_DAG

adj_matrix_DAG

for (i in 1:9) {
  for (j in 1:i)
    adj_matrix_DAG[i,j]=0
}



gg <- as(adj_matrix_DAG, "graphNEL")
plot(gg)

rownames(adj_matrix_DAG) <- colnames(adj_matrix_DAG) <- names(car)

library(gRain)
print(CPT(adj_matrix_DAG, car,bayes_smooth =1))
CPTs <- compileCPT(CPT(adj_matrix_DAG, car,bayes_smooth =1))
G <- grain(CPTs)

querygrain(G, nodes = names(car), type = "joint")

sim <- simulate.grain(object = G, nsim = 3500, seed = 433)


tch3_step <- k_tcherry_step(data = sim, k = knum, smooth = 0.001)

#gg <- as(tch3_step$adj_matrix, "graphNEL")
#plot(gg)


library(gRain)
graph_tch3_step <- as(tch3_step$adj_matrix, "graphNEL")
G_tch3_step <- grain(x = graph_tch3_step, data = sim, smooth = 0.001)
querygrain(G_tch3_step, nodes = names(car), type = "joint
           ")


test <-read.csv("D:/work/py_file/week/exp/test_all.csv" ,quote = "",header = TRUE)
# str(test)


test$KNN = as.character(test$KNN)
test$LR  = as.character(test$LR)
test$RF = as.character(test$RF)
test$GBDT = as.character(test$GBDT)

test$v_KNN = NULL
test$v_LR  = NULL
test$v_RF = NULL
test$v_GBDT = NULL

#test$v_KNN = as.character(test$v_KNN)
#test$v_LR  = as.character(test$v_LR)
#test$v_RF = as.character(test$v_RF)
#test$v_GBDT = as.character(test$v_GBDT)

test$i_KNN = NULL
test$i_LR  = NULL
test$i_RF = NULL
test$i_GBDT = NULL

#test$i_KNN = as.character(test$i_KNN)
#test$i_LR  = as.character(test$i_LR)
#test$i_RF = as.character(test$i_RF)
#test$i_GBDT = as.character(test$i_GBDT)

test$rof_KNN = as.character(test$rof_KNN)
test$rof_LR  = as.character(test$rof_LR)
test$rof_RF = as.character(test$rof_RF)
test$rof_GBDT = as.character(test$rof_GBDT)

test$real = as.character(test$real)




pp<-predict(object = G_tch3_step, response = "real", newdata = test)

write.csv(x = pp,file = "D:/work/py_file/week/tcresult/pred_4th.csv",row.names = FALSE)
t2<-Sys.time()
t2-t1        





