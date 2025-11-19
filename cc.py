import pandas as pd
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
    
for i in range(df.shape[1]):
    ll.append(cc[i],"real")
    
from pgmpy.base import DAG
tan_structure = DAG()
tan_structure.add_nodes_from(nodes=cc)


tan_structure.add_edges_from(ebunch=ll)

print(tan_structure.edges)
print(tan_structure.nodes)