import pandas as pd
from sklearn.preprocessing import Imputer
from utils import missing_value_split_train_test_data
from utils import missing_value_score_dataset

input_file="/home/pliu/Downloads/data_set/python_ml/melbourne.csv"

data = pd.read_csv(input_file)

#print(data.describe())
print(data.sample(5))

# show the empty value cell
print(data.isnull().sum())


################################################
##########Drop columns with Missing Values ####
###############################################

# axis = 1 means drop column, axis = 0 means drop rows
# how='all' means drop the column contains all nan,
# how='any' means drop the column contains any nan
data_without_missing_values = data.dropna(axis=1)
#print(data_without_missing_values.sample(5))
# if your data has been split into train and test, you need to drop the
# same column in both train and test data.

column_with_missing_value= [col for col in data.columns if data[col].isnull().any()]
#print(column_with_missing_value)

# ['Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom',
# 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea',
# 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount']

# We can't use this solution, because we drop even the Price column

###################################################
####### Imputation of missing value ##############
##################################################

"""
Imputation fills in the missing value with some number. 
The imputed value won't be exactly right in most cases, 
but it usually gives more accurate models than dropping 
the column entirely.

"""

data_imputer=Imputer()

# the imputer only can fills numeric empty values,
# all columns with string(object) type will cause error
# if you knew which columns is object, you can drop them before,
# data.drop(['Regionname','CouncilArea'],axis=1)
# all you can select only dtypes which is not object
data_with_imputed_values = data_imputer.fit_transform(data.select_dtypes(exclude=['object']))

print(data_with_imputed_values)

######################################################
#########Compare accuracy of two different solutions##
######################################################

#print(data.sample(5))

# Prepare data, eliminate all non numeric data
melb_file_path="/home/pliu/Downloads/data_set/python_ml/melb_data.csv"
melb_data = pd.read_csv(melb_file_path)
label_data = melb_data.Price

feature_data = melb_data.drop(['Price'], axis=1)

feature_data_numeric = feature_data.select_dtypes(exclude=['object'])

print(feature_data_numeric.sample(5))

train_X, test_X, train_y, test_y = missing_value_split_train_test_data(feature_data_numeric,label_data)

# get score when droping
cols_with_missing = [col for col in train_X.columns
                                 if train_X[col].isnull().any()]
print(cols_with_missing)
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_test_X = test_X.drop(cols_with_missing, axis=1)
print(reduced_train_X.sample(5))
print("Mean Absolute Error from dropping columns with Missing Values:")
print(missing_value_score_dataset(reduced_train_X, reduced_test_X, train_y, test_y))

# get score when imp

my_imputer = Imputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
print("Mean Absolute Error from Imputation:")
print(missing_value_score_dataset(imputed_train_X, imputed_test_X, train_y, test_y))

# Get Score from Imputation with Extra Columns Showing What Was Imputed
# Create a new column for all columnt with missing value, for each row,
# if the value is missing ,the adding column value is true, otherwise is false

imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()

cols_with_missing = (col for col in train_X.columns
                                 if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

print(imputed_X_train_plus.sample(5))
# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(missing_value_score_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y))