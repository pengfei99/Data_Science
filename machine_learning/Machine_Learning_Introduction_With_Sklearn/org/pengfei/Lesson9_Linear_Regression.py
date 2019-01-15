import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

######################################################################################
###########################Introduction ##############################################
######################################################################################
"""
In previous lessons, we've learned the basic steps for doing machine learning. 
1. Preprocessing - Understand the data, do feature selection, feature engineering, Once the features is selected, we 
                   need to clean the data, remove duplicates, impute empty cells.
2. Train the model - fit the model with training data
3. Validate the model - Use the model validation method (e.g. cross validation) to validate the model.

"""
"""
In this lesson, we will use algo linear regression to predict boston housing prices. The data sets are from scikit-learn
library. We could use .DESCR to print the data set descriptions. Beware, this only works for dataset which are provided 
by scikit learn library. 

The attributes descriptions:

- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
- MEDV     Median value of owner-occupied homes in $1000's
"""
data=datasets.load_boston()
#print(data.DESCR)

df = pd.DataFrame(data.data,columns=data.feature_names)
target=pd.DataFrame(data.target,columns=["MEDV"])
full_df=pd.concat([df,target],axis=1)
# get the shape of dataframe
#print(df.shape)
# show a row of the dataframe
print(full_df.head(1))
# check if there are missing values
#print(df.isnull().sum())
# get the basic stats
#print(df.describe())
# get the type of each columns
print(full_df.dtypes)


####################################################################################
###################### Feature selections #########################################
###################################################################################

"""
We know that MEDV will be our label(answer) columns, becasue we want to predict house price. Now we need to choose 
feature columns which can help us to predict the label columns.

First, I will try column RM (room numbers). In common sense, more rooms means more expensive. 
"""

# full_df.plot(x="RM",y='MEDV',style='o')
# plt.title("Rooms vs Price")
# plt.xlabel("Room numbers")
# plt.ylabel("Price")
# plt.show()



##########################################################################################
######################## Train the model #################################################
#########################################################################################

X=df
y=target['MEDV']

lm= linear_model.LinearRegression()
scores=cross_val_score(lm,X,y,scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))
lm.fit(X,y)
print(lm.coef_)
print(lm.intercept_)