# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pydot
from sklearn.externals.six import StringIO


"""
This is lesson 1 of the machine learning

In this tutorial we will use a house price data set to build a regression model to predict house price. We will follow 
the below sop.

The sop
1. Understand data and choose feature data and label data
1.1 print(data.shape) # Determine the size of your data set 
1.2 print(data.dtypes) # Get all column names and types
1.3 print(data.isnull().sum()) # Get all empty value cells
1.4 print(data.describe()) # Get basic statistic info of each columns
1.5 print(data.head(5)) # Get top 5 lines of the data set

2. Based on the study of step 1. You can build a training data with feature data set and a prediction target (label) data set
3. Choose a model based on your data set
4. Train your model with the training data
5. Test your model prediction
6. Validate your model with test data (Determine accuracy for classification problem, mae for regression problem)
7. Tune your model to improve your model accuracy or mae.
"""
input_file = "/home/pliu/Downloads/data_set/python_ml/train.csv"
data = pd.read_csv(input_file)






##################################################################
#####Understand data and choose feature data ####################
################################################################

"""
In our example, we want to predict the price of house, so the label data will be the price of houses.
The following code isolated the column SalePrice and see what they looks like
"""
#get a sub data frame of one column (our label data set)
sale_price=data.SalePrice
#print(sale_price.describe())

"""
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64

This seems correct, the mean house value is 180921 in dollar. We can use it directly without any curation effort
"""

"""
In this lesson, we don't discuss how to choose feature data, because it will be an entire lesson for it.
We just pick some feature data which is relevant of house price (e.g. Year Built, overall condition, location (LotArea))
"""
#get a sub data frame of multi column (example of feature data set)
columns_of_interest=['LotArea','SalePrice','YearBuilt','OverallCond']
sub_data=data[columns_of_interest]
#print(sub_data.describe())

# Column of interest:
# LotArea: Lot size in square feet
# SalePrice
# YearBuilt
# OverallCond


###########################################################
######build model with decision tree algo #################
###########################################################
"""
After choosen feature data and label data, we need to find a model to do the prediction.
There many models, and they have pros and cons. Based on your data, you need to find the right 
model which fits better. 

Here we choose DecisionTree, which is the simplest model
"""
house_price_model=tree.DecisionTreeRegressor()

"""
In ml, we called predication target (label data) as y, in our case, it is the sale price of the house
"""
y=sale_price

# We choose the following column as feature data
# LotArea: Lot size in square feet
# YearBuilt
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# FullBath: Full bathrooms above grade
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

feature_data_column=['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

feature_data=data[feature_data_column]

# we use X to name the feature_data
X=feature_data

"""After building the feature data and label data, we need to train the model"""

# train the model
house_price_model.fit(X,y)

"""After the model is trained, we can predict the house price"""
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(house_price_model.predict(X.head()))


################################################################
############# Model validation #################################
################################################################

""" 
In most (though not necessarily all) applications, the relevant measure of model quality is predictive accuracy. 
In other words, will the model's predictions be close to what actually happens. 

Even with this simple approach, you'll need to summarize the model quality into a form that someone can understand. 
If you have predicted and actual home values for 10000 houses, you will inevitably end up with a mix of good and 
bad predictions. Looking through such a long list would be pointless.

There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error 
(also called MAE). Let's break down this metric starting with the last word, error.

The prediction error for each house is: 

error=actual−predicted

So, if a house cost $150,000 and you predicted it would cost $100,000 the error is $50,000.

With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. 
We then take the average of those absolute errors. This is our measure of model quality. In plain English, 
it can be said as

On average, our predictions are off by about X

We first load the Melbourne data and create X and y. That code isn't shown here, since you've already seen it a couple times.
"""

# This two lines calculate the "in-sample" mae scores
predicted_house_price=house_price_model.predict(X)

mae = metrics.mean_absolute_error(y,predicted_house_price)

print("MAE value is : "+str(mae))

"""
The Problem with "In-Sample" Scores

The measure we just computed can be called an "in-sample" score. We used a single set of houses (called a data sample) 
for both building the model and for calculating it's MAE score. This is bad.

Imagine that, in the large real estate market, door color is unrelated to home price. However, in the sample of data 
you used to build the model, it may be that all homes with green doors were very expensive. The model's job is to find 
patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with 
green doors.

Since this pattern was originally derived from the training data, the model will appear accurate in the training data.

But this pattern likely won't hold when the model sees new data, and the model would be very inaccurate 
(and cost us lots of money) when we applied it to our real estate business.

Even a model capturing only happenstance relationships in the data, relationships that will not be repeated 
when new data, can appear to be very accurate on in-sample accuracy measurements.
"""

# I split the data set into traing data and test data.
# random_state argument guarantees we get the same split every time we
# run this script.

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

price_model=tree.DecisionTreeRegressor()

price_model.fit(train_X,train_y)

price_predication= price_model.predict(test_X)

out_sample_mae=metrics.mean_absolute_error(test_y,price_predication)

print ("Out sample mae : "+str(out_sample_mae))


######################################################################
#############Viz decision tree########################################
#####################################################################
"""To better understand how decision tree works, we cloud print the decision tree"""

# dot_data = StringIO()
# tree.export_graphviz(house_price_model,
#                       out_file=dot_data,
#                       feature_names=feature_data_column,
#                       filled=True, rounded=True,
#                       impurity=False)
#
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
#
# print(graph)
#
# graph[0].write_pdf("house_price_predication_decision_tree.pdf")




