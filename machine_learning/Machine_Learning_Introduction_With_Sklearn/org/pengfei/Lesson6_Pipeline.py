import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from utils import missing_value_split_train_test_data
from sklearn.metrics import mean_absolute_error

"""
What Are Pipelines

Pipelines are a simple way to keep your data processing and modeling code organized. Specifically, a pipeline bundles 
preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

Many data scientists hack together models without pipelines, but Pipelines have some important benefits. Those include:

- Cleaner Code: You won't need to keep track of your training (and validation) data at each step of processing. 
                Accounting for data at each step of processing can get messy. With a pipeline, you don't need to 
                manually keep track of each step.
- Fewer Bugs: There are fewer opportunities to mis-apply a step or forget a pre-processing step.
- Easier to Productionize: It can be surprisingly hard to transition a model from a prototype to something deployable 
                           at scale. We won't go into the many related concerns here, but pipelines can help.
- More Options For Model Testing: For example cross-validation.
"""
melb_file_path="/home/pliu/Downloads/data_set/python_ml/melb_data.csv"
melb_data = pd.read_csv(melb_file_path)
label_data = melb_data.Price

feature_data = melb_data.drop(['Price'], axis=1)

feature_data_numeric = feature_data.select_dtypes(exclude=['object'])

print(feature_data_numeric.sample(5))

train_X, test_X, train_y, test_y = missing_value_split_train_test_data(feature_data_numeric,label_data)

# build pipeline
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X,train_y)

predictions = my_pipeline.predict(test_X)

print("Mean Absolute Error:")
print(mean_absolute_error(test_y,predictions))


"""
Understanding Pipelines

Most scikit-learn objects are either transformers or models.

- Transformers are for pre-processing before modeling. The Imputer class (for filling in missing values) is an example 
  of a transformer. Over time, you will learn many more transformers, and you will frequently use multiple transformers 
  sequentially.

- Models are used to make predictions. You will usually preprocess your data (with transformers) before putting it in a 
  model.

Eventually you will want to apply more transformers and combine them more flexibly. We will cover this later in an Advanced Pipelines tutorial.
"""
