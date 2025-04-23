from sklearn import tree


#bumpy for 0 smooth for 1
features=[[140, 1], [130, 1], [150, 0], [170, 0]]

#0 for apple, 1 for orange
labels=[0, 0, 1, 1]

#chose a classifier algo classifier is for 0,1 decision(classification problem)
# determin the house price is a regression problem, you need to use
clf = tree.DecisionTreeClassifier()

#train the classifier
clf = clf.fit(features, labels)

testSample = [[150, 0]]
print clf.predict(testSample)

