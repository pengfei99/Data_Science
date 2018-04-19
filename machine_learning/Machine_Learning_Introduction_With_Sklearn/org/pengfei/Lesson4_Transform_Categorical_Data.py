import pandas as pd
from utils import get_mae_random_forest

"""
Introduction

All the ml model are not text (string) friendly, we need to transform text to digit(e.g. int, float, etc.)
Categorical data is data that takes only a limited number of values (e.g. Color red, blue, etc.).

One-Hot Encoding : The Standard Approach for Categorical Data

One hot encoding is the most widespread approach, and it works very well unless your categorical variable takes on a 
large number of values (i.e. you generally won't use one hot encoding for variables taking more than 15 different 
values. It'd be a poor choice in some cases with fewer values, though that varies.)

One hot encoding creates new (binary) columns, indicating the presence of each possible value from the original data. 
For example, we have a column color which contain 2 possible values (red, blue)

COLOR                           RED     BLUE
 red                             1        0
 blue                            0        1
 
"""

train_input_file = "/home/pliu/Downloads/data_set/python_ml/train.csv"
test_input_file = "/home/pliu/Downloads/data_set/python_ml/test.csv"
train_data = pd.read_csv(train_input_file)
test_data = pd.read_csv(test_input_file)

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns.
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
cols_with_missing = [col for col in train_data.columns
                                 if train_data[col].isnull().any()]
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]

# print("Categorical column name"+str(low_cardinality_cols))

numeric_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
# print("Numric column name" + str(numeric_cols))
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# get 5 sample of the train data set
# print(train_predictors.sample(5))

# print(train_predictors.dtypes.sample(5))

# get the data set line and column size before the transformation
# print(train_predictors.shape)


"""
Object indicates a column has text (there are other things it could be theoretically be, but that's unimportant for 
our purposes). It's most common to one-hot encode these "object" columns, since they can't be plugged directly into 
most models. Pandas offers a convenient function called get_dummies to get one-hot encodings. Call it like this:
"""

###########################################################################################
################ one hot encoded categorical data##########################################
###################################################################
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
#print(one_hot_encoded_training_predictors.shape)

"""
You could notice that, the column numbers changed from 57 to 159.
"""

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae_random_forest(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae_random_forest(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

###########################################################################
######################Applying to Multiple Files###########################
##########################################################################

"""
So far, you've one-hot-encoded your training data.  What about when you have multiple files (e.g. a test dataset, 
or some other data that you'd like to make predictions for)?  Scikit-learn is sensitive to the ordering of columns, 
so if the training dataset and test datasets get misaligned, your results will be nonsense.  This could happen if a 
categorical had a different number of values in the training data vs the test data.

Ensure the test data is encoded in the same manner as the training data with the align command:
"""

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left',
                                                                    axis=1)

"""
Conclusion

The world is filled with categorical data. You will be a much more effective data scientist if you know how to use this 
data. Here are resources that will be useful as you start doing more sophisticated work with cateogircal data.

Pipelines: Deploying models into production ready systems is a topic unto itself. While one-hot encoding is still 
a great approach, your code will need to built in an especially robust way. Scikit-learn pipelines are a great tool 
for this. Scikit-learn offers a class for one-hot encoding (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 
and this can be added to a Pipeline. Unfortunately, it doesn't handle text or object values, which is a common use case.

Applications To Text for Deep Learning: Keras (https://keras.io/preprocessing/text/#one_hot) and 
TensorFlow (https://www.tensorflow.org/api_docs/python/tf/one_hot) have fuctionality for one-hot encoding, 
which is useful for working with text.

Categoricals with Many Values: Scikit-learn's FeatureHasher (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher)
uses the hashing trick to store high-dimensional data. This will add some complexity to your modeling code.
"""