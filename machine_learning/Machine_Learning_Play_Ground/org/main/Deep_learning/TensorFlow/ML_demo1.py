# In this demo, we use decision tree of sk-learn to classify iris data

import sklearn.datasets as data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

# load iris dataset
iris = data.load_iris()

# get iris features
features = pd.DataFrame(iris.data, columns=iris.feature_names)
feature_data_column = iris.feature_names
print(features)

# get iris labels
labels = iris.target

# There are three possible values
# setosa: 0
# virginica: 1
# versicolor: 2
print(labels)

# split data to train and test data
train_X, test_X, train_y, test_y = train_test_split(features, labels, random_state=0)

# specify the ML model as decision tree
model = DecisionTreeClassifier()

# train the model with training data
model.fit(train_X, train_y)

# predict the test data with the trained model
prediction = model.predict(test_X)

# validate the model with its accuracy
accuracy = metrics.accuracy_score(test_y, prediction)
print("Our model has accuracy : " + str(accuracy))

# check the prediction with the test labels
print(test_y[1])
print(prediction[1])

# check the decision tree model contents
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=6)
plt.show()
