from sklearn import datasets


##################################################################################################################
######################### Machine learning Terminology ###########################################################
#################################################################################################################

"""
In a data set,

- Each row is an observation (a.k.a : sample, example, instance, record)
- Each column is a feature (a.k.a : predictor, attribute, independent variable, input, regressor )\
- Each value we are predicting is the response (a.k.a : target, outcome, label, dependent variable)


1. Supervised learning : if the output/response is explicitly given, for example, the data set sould have
(inputs, correct output), inputs will be a set of predictor, the correct output is the label.

2. Unsupervised learning: there is no given output/response, the data set looks like this (inputs, ?), which means you have
no ideas what the output looks like

To illustrate the difference between supervised and unsupervised, Suppose we have a coin regnition problem

The coin data set has three columns, diameter, mass, and type, For example a row 
20 mm, 10 gram, 10 centime. 30mm, 20gram, 50 centime, 35mm, 35gram, 1 euros. So if you give me a coin with ~22mm, ~11 gram, 
we know this coin is a 10 centime. This is supervised learning.
 
If we remove the column of type, we only have the diameter and mass column. What we can do is observe the patterns in 
the  data set. we can divide all the data set into three clusters. The coin with diameters close to 20mm and mass close
to 10 gram is a cluster called coin type 1. The same thing for coin type2 and type3. If you give me a new coin, I will 
know it's a type1, type2 or type3. This is unsupervised learning.


3. Reinforcement learning : When you associate your output with a grade(inputs, some output, grade for this output), 
we call it reinforcement learning. It's very useful in design of game AI. 


4. Classification is supervised learning in which the response is categorical.
5. Regression is supervised learning in which the response is ordered and continuous.



"""
###################################################################################################################
###########################################When we need to use machine learning #####################################
################################################################################################################

"""
Machine learning is used when
- A pattern exists 
- We cannot pin it down mathematically (no determintistic function exists)
- We have enough data on it

(e.g. house price depends on location, size, etc. We know there is a pattern. But we can't represent the exact pattern 
with a function. We need enough data to build a model which is close enough to the pattern)
"""

"""
The Hoeffing's Inequality P[|v-m|>e]<=2e^-2(e^2)N
"""

#####################################################################################################################
######################## Requirements for working with data in scikit-learn #########################################
#####################################################################################################################


"""
1. Features and labels are separate objects
2. Features and labels should be numeric
3. Features and labels should be NumPy arrays
4. Features and labels should have specific shapes
"""

iris=datasets.load_iris()

# type
print(type(iris.data))
print(type(iris.target))

# shape data rows must equals target rows
print(iris.data.shape)
print(iris.target.shape)

# value
print(iris.data)
print(iris.target)