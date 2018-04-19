import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

"""
In cross-validation, we divide one data set into many pieces, we pick one piece as test 
data, and rest of the pieces are used as training data. We train our model with training 
data and calculate the score of test data. We repeat this for all pieces.

For example, if we divide a data set into five pieces, first run piece-1 will be test data
second run piece-2 will be test data. After 5 run, we have the score of all 5 run.


"""

# create pipeline
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# build dataset
input_file_path="/home/pliu/Downloads/data_set/python_ml/melb_data.csv"
data=pd.read_csv(input_file_path)
#print(data.isnull().sum())
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

"""
You may notice that we specified an argument for scoring. 
This specifies what measure of model quality to report. 
The docs for scikit-learn show a list of options.
http://scikit-learn.org/stable/modules/model_evaluation.html
"""
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

"""
Conclusion

Using cross-validation gave us much better measures of model quality, 
with the added benefit of cleaning up our code (no longer needing to keep 
track of separate train and test sets. So, it's a good win.

"""