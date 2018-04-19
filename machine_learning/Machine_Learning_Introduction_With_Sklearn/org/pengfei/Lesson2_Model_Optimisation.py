# -*- coding: utf-8 -*-

from utils import split_data
from utils import get_mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

"""
This is lesson 2 of the machine learning

In this tutorial we will continue to use the house price data set to build a regression model to predict house price. 
This time we will try to optimize the model which can make it do better prediction.

1. Tune the model to resolve overfitting and underfitting problem
2. Use other model which can do better facing overfitting and underfitting problems
3. Compare the accuracy or mae to find the best solution

P.S. The validation data (test data), should never be in the train data set. Because it may destroy the validation 
accuracy totally. For example, in a train data set, all the houses which have blue doors have a very high price. The 
model which we trained will think there is a correlation between blue door and price. In reality, there is no link 
between blue door and price. If the validation data comes from training data, the mae value will not review this.

"""

"""
Models can suffer from either:

Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or

Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
"""
input_file = "/home/pliu/Downloads/data_set/python_ml/train.csv"
feature_data_column = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# build the trainning data and test data
train_X, test_X, train_y, test_y = split_data(input_file_path=input_file,featrue_data_names=feature_data_column)

##################################################################################################
#################################Optimize decision tree model###################################
###############################################################################################

"""
We're still using Decision Tree models, which are not very sophisticated by modern machine learning standards. Because 
it does not do well with overfitting and underfitting problems.

In the following example, we will use max_leaf_nodes option to optimize the model
"""

# test the mae for each max_leaf_nodes config
for max_leaf_nodes in [5,50,500,5000]:
    my_mae=get_mae(max_leaf_nodes=max_leaf_nodes,train_X=train_X,test_X=test_X,train_y=train_y,test_y=test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

############ Output ########################################
# Max leaf nodes: 5  		     Mean Absolute Error:  35190
# Max leaf nodes: 50  		     Mean Absolute Error:  27825
# Max leaf nodes: 500  		     Mean Absolute Error:  32662
# Max leaf nodes: 5000  		 Mean Absolute Error:  33382

########################################################

"""
We could notice with 50 leaf, we have the best mae
"""

####################################################################################
#####################Random_forest##################################################
#################################################################################

"""
We have seen the limit of decision tree model. We may need another model which fit better to our needs.
In this section, we choose random forest model
"""

"""
Introduction of random forest algorithm

What is Random forest algorithm?

Random forest algorithm is a supervised classification algorithm. As the name suggest, this algorithm creates the 
forest with a number of trees.

In general, the more trees in the forest the more robust the forest looks like. In the same way in the random forest 
classifier, the higher the number of trees in the forest gives the high accuracy results.

Random forest algorithm advantages.

1. The same random forest algorithm or the random forest classifier can use for both classification and the regression 
   task.
2. Random forest classifier will handle the missing values.
3. When we have more trees in the forest, random forest classifier won’t overfit the model.
4. Can model the random forest classifier for categorical values also.

How random forest algorithm works?

The pseudocode for random forest algorithm can split into two stages.

1. Random forest creation pseudocode.
2. Pseudocode to perform prediction from the created random forest classifier.

First, let’s begin with random forest creation pseudocode

Random Forest pseudocode:
Step 1. Randomly select “k” features from total “m” features. Where k << m
Step 2. Among the “k” features, calculate the node “d” using the best split point.
Step 3. Split the node into daughter nodes using the best split.
Step 4. Repeat 1 to 3 steps until “l” number of nodes has been reached.
Step 5. Build forest by repeating steps 1 to 4 for “n” number times to create “n” number of trees.

Random forest prediction pseudocode:

Step 1. Takes the test features and use the rules of each randomly created decision tree to predict the outcome and 
stores the predicted outcome (target)
Step 2. Calculate the votes for each predicted target.
Step 3. Consider the high voted predicted target as the final prediction from the random forest algorithm.

For example, Suppose we formed 100 random decision trees to from the random forest. With the given test data the 100 
random decision trees are predict 3 unique targets x, y, z then 60 trees are predicting the target will be x. Then the 
final random forest returns the x as the predicted target.

This concept of voting is known as majority voting.

"""
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(test_X)
print("Random forest model Mean Absolute Error : " +str(mean_absolute_error(test_y, melb_preds)))

# Mean Absolute Error : 23288.5705936

"""
We cloud notice that the mae is much better than the decision tree
"""
#################################################################################################
#########################XGBoost model###############################################################
#################################################################################################

"""
XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, 
as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.

Follow this link to better understand XGBoost 
http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

To reach peak accuracy, XGBoost models require more knowledge and model tuning than techniques like Random Forest. 
After the following tutorial, you'ill be able to
1. Follow the full modeling workflow with XGBoost
2. Fine-tune XGBoost models for optimal performance

XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm (scikit-learn has another version of 
this algorithm, but XGBoost has some technical advantages.)

Step 1. Build a naive model
Step 2. Calculate Errors
Step 3. Build model predicting errors
Step 4. Add last model to ensemble
Step 5. Repeat step 2 to 4

We go through cycles that repeatedly builds new models and combines them into an ensemble model. We start the cycle 
by calculating the errors for each observation in the dataset. We then build a new model to predict those. We add 
predictions from this error-predicting model to the "ensemble of models."

To make a prediction, we add the predictions from all previous models. We can use these predictions to calculate 
new errors, build the next model, and add it to the ensemble.

There's one piece outside that cycle. We need some base prediction to start the cycle. In practice, 
the initial predictions can be pretty naive. Even if it's predictions are wildly inaccurate, subsequent additions 
to the ensemble will address those errors.

"""

#####################################################
###############Native xgb regressor #################
#####################################################


# input_file = "/home/pliu/Downloads/data_set/python_ml/train.csv"
# feature_data_column = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# train_X, test_X, train_y, test_y = split_data(input_file_path=input_file,featrue_data_names=feature_data_column)

xgb_model = XGBRegressor()

xgb_model.fit(train_X,train_y,verbose=False)
predictions = xgb_model.predict(test_X)

print("Naive xgb model Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

#Mean Absolute Error : 22242.5581336

##################################################
####### Tuning xgb model ########################
################################################

"""
XGBoost has a few parameters that can dramatically 
affect your model's accuracy and training speed. 
The first parameters you should understand are:

1. n_estimators -> specifies how many times to go through the modeling cycle

2. early_stopping_rounds -> Early stopping causes the model to stop iterating 
when the validation score stops improving, even if we aren't at the hard stop 
for n_estimators. 

It's smart to set a high value for n_estimators and then use early_stopping_rounds 
to find the optimal time to stop iterating.

3. learning_rate -> Instead of getting predictions by simply adding up the predictions
from each component model, we will multiply the predictions from each model by a small 
number before adding them in. This means each tree we add to the ensemble helps us less. 
In practice, this reduces the model's propensity to overfit. learning_rate value is between
0 and 1.

4. n_jobs -> On larger datasets where runtime is a consideration, 
you can use parallelism to build your models faster. It's common to 
set the parameter n_jobs equal to the number of cores on your machine. 
On smaller datasets, this won't help.

This will not change the model accuracy, it only reduces the training time
"""

# in the following example, we set the n_estimators as 1000
# early_stopping_rounds=5
xgb_tuning_model = XGBRegressor(n_estimators=1000)
xgb_tuning_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)

tuning_predictions = xgb_tuning_model.predict(test_X)

print("xgb model n_est=1000 early_stopping_rounds=5 Mean Absolute Error : " + str(mean_absolute_error(tuning_predictions, test_y)))
#Mean Absolute Error : 21630.9005137

xgb_tuning_model_with_learning_rate=XGBRegressor(n_estimators=1000, learning_rate=0.08)
xgb_tuning_model_with_learning_rate.fit(train_X, train_y, early_stopping_rounds=5,
             eval_set=[(test_X, test_y)], verbose=False)

tuning_predictions_with_learning_rate=xgb_tuning_model_with_learning_rate.predict(test_X)
print("xgb model n_est=1000 early_stopping_rounds=5 learning_rate=0.08 Mean Absolute Error : " + str(mean_absolute_error(tuning_predictions_with_learning_rate, test_y)))
