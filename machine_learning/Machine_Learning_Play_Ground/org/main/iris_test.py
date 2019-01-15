import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import ScrappyKNN
import pydot

#load iris data set
iris = load_iris()
# print(iris)
#iris feature data
header = iris.target_names
print(header)
X = iris.data
print(X[1])
#iris label data
y = iris.target
print(y[1])
#auto split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

test_idx = [0, 50, 100]

#manule split train and test data
#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#DecisionTreeClassifier
#clf = tree.DecisionTreeClassifier()

#nearest neighbors classifier
clf = KNeighborsClassifier()

#Home made classifier
#clf = ScrappyKNN.ScrappyKNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print predictions

print  accuracy_score(y_test, predictions)

#viz decision tree
# dot_data = StringIO()
# tree.export_graphviz(clf,
#                      out_file=dot_data,
#                      feature_names=iris.feature_names,
#                      filled=True, rounded=True,
#                      impurity=False)
#
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
#
# print(graph)
#
# graph[0].write_pdf("iris.pdf")








# print iris.feature_names
# print iris.target_names
# print iris.data
# print iris.target
# print iris.data[0]
# print iris.target[0]