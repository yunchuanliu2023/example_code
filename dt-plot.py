from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

iris = datasets.load_iris()
X = iris.data
y = iris.target
# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)

# text_representation = tree.export_text(clf)
# print(text_representation)

print(iris.feature_names)
print(iris.target_names)
def gg():

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                       feature_names=iris.feature_names,  
                       class_names=iris.target_names,
                       filled=True)
    fig.savefig("decistion_tree.png")                   
