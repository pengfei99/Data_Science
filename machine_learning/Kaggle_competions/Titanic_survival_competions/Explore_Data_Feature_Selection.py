# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils import split_titanic_passanger_names
from utils import get_titanic_passanger_title
from utils import get_titanic_passanger_surname
from utils import capitalize_first_letter
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils import split_train_test_data
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
import seaborn as sns

train_data_input_file="/home/pliu/Downloads/data_set/titanic_competition/train.csv"
test_data_input_file="/home/pliu/Downloads/data_set/titanic_competition/test.csv"

train_data=pd.read_csv(train_data_input_file,index_col=0)
test_data=pd.read_csv(test_data_input_file,index_col=0)

#######################################################
############### Get data shape, type ##################
#######################################################

# print(train_data.dtypes)
# print(train_data.shape)
# print(test_data.shape)
#
# print(train_data.isnull().sum())
# print(train_data.head())

#####################################################
################Merge the train and test data #######
#####################################################

# The shape of train (891, 11), the shape of test (418, 10)
# The test data set miss a column of Survived
# Create the column of Survived with Na value in test data set
test_data['Survived']=np.nan
# print(test_data.head())

# merge the train and test data
full_data=pd.concat([train_data,test_data])
# print(full_data.shape)
# print(full_data.head())
# now the full data set is (1309, 11)
#######################################################
############ Visulize important data ################
#####################################################

############## how many survived, how many died

# survived_passanger=full_data['Survived'].value_counts()
# print(survived_passanger)
# survived_passanger.plot.bar()
# plt.show()
# 0 (die)    549, 1 (live)    342

############# how many are males, how many are females
# sex_of_passanger=full_data['Sex'].value_counts()
# print(sex_of_passanger)
# sex_of_passanger.plot.bar()
# plt.show()

######## the chance of survival between male and female


# Draw a nested barplot to show survival for class and sex
# sns.countplot(x="Sex", hue="Survived", data=full_data)
# plt.show()
# we could notice that, the female have much more chance to survive.

###### the chance of survival between passenser class
# sns.countplot(x="Pclass",hue="Survived",data=full_data)
# plt.show()


# we could notice that, the first class have much more chance to survive than 3rd class




# first step, we cloud eliminate the column Name, ticket number, embarked Port, cabin number

#feature_col=['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
# feature_col=['Pclass', 'Sex', 'Age', 'Fare']
# X = pd.get_dummies(train_data[feature_col])
# y = train_data.Survived
#
# X_test_submit=pd.get_dummies(test_data[feature_col])

##############################################################
##################Feature Engineering########################
#############################################################

######Creating the title, surname variable

# str="Futrelle, Mrs. Jacques Heath (Lily May Peel)"
# surname,title=split_titanic_passanger_names(str)
# print(surname)
# print(title)

# We split the full name into surname and title
# the apply function apply a function on a column of a data frame
# lambda row, row is the value of each row of the column, in our case
# it's the fullname
# the two function is defined in utils
full_data['Title'] = full_data['Name'].apply(lambda row: get_titanic_passanger_title(row))
full_data['Surname'] = full_data['Name'].apply(lambda row: get_titanic_passanger_surname(row))



#####Creating a new variable class_sex
"""
Details that are lost when using the predictors separately were actually already mentioned 
in the beginning of this section (Pclass 1 and 2 are almost guaranteed survival for women, 
and Pclass 2 is almost as bad as Pclass 3 for men).
"""
# For example, Pclass = 3, Sex = male => P3Male
full_data['Pclass_Sex'] = full_data['Pclass'].apply(lambda row: "P"+str(row))+full_data['Sex'].apply(lambda row: capitalize_first_letter(row))
# full_data['Pclass_Sex'].value_counts().plot.bar()
# plt.show()
# sns.countplot(x="Pclass_Sex",hue="Survived",data=full_data)
# plt.show()
#print(full_data.head())
####################################################################
################Solution without pipeline ##########################
####################################################################

train_data=full_data[full_data['Survived'].notnull()]
print(train_data.shape)
test_data=full_data[full_data['Survived'].isnull()]
print(test_data.shape)


# we find out the age column has 177 empty cells
#print(X.isnull().sum())

# we will use Imputer to fill out this empty cells
# my_Imputer=Imputer()
# X_after_imputation=my_Imputer.fit_transform(X)
# # after transform, the data frame X become numpy.ndarray
# X_test_submit_after_imputation=my_Imputer.fit_transform(X_test_submit)
#
# train_X,test_X,train_y,test_y=split_train_test_data(X_after_imputation,y)
# # print(raw_data.shape)
# # print(raw_data.head())
#
# xgb_model = XGBClassifier(n_estimators=10000, learning_rate=0.02)
# xgb_model.fit(train_X,train_y, early_stopping_rounds=1000,
#              eval_set=[(test_X, test_y)],verbose=False)
# predictions = xgb_model.predict(test_X)
#
# score=accuracy_score(test_y,predictions)
# print("The score is "+str(score))
#
# predict_test_data=xgb_model.predict(X_test_submit_after_imputation)
# print(type(predict_test_data))
# accuracy before imputation 0.817164179104
# accuracy after imputation 0.828358208955
# with n_estimators= 10000, learning_rate=0.02, early_stopping_rounds=1000, the accuracy 0.835820895522388

####################################################################
######################Solution with pipeline ######################
##################################################################
# train_X_raw,test_X_raw,train_y_raw,test_y_raw=split_train_test_data(X,y)
# my_pipeline=make_pipeline(my_Imputer,xgb_model)
# my_pipeline.fit(train_X_raw,train_y_raw)
#
# pipeline_predictions=my_pipeline.predict(test_X_raw)
# pip_score=accuracy_score(test_y_raw,pipeline_predictions)
# print(pip_score)

#########################################################
##################Cross validation#######################
#########################################################

###########################################################
####### Data set description #############################
########################################################
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
#
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
#
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
#
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.