import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# calculate an accuracy from the confusion matrix
def get_model_accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


# Prepare data for ml model
input_path = "/home/pliu/data_set/argo_data_pipeline/pokemon-cleaned.csv"
input_df = pd.read_csv(input_path, index_col=0)
label_data = input_df.legendary
label_sample = label_data.sample(5)
feature_data = input_df.drop(['legendary', 'generation', 'total'], axis=1).select_dtypes(exclude=['object'])
feature_sample = feature_data.sample(5)
# split data into training_data and test_data
train_X, test_X, train_y, test_y = train_test_split(feature_data, label_data, train_size=0.8, test_size=0.2,
                                                    random_state=0)


# Here we choose three hyper parameters to optimize our model
# 1. n_estimators: The n_estimators parameter specifies the number of trees in the forest of the model.
#    The default value for this parameter is 10(changed to 100 in 0.22 version), which means that 10 different
#    decision trees will be constructed in the random forest.
# 2. max_depth: The max_depth parameter specifies the maximum depth of each tree. The default value for max_depth
#    is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of
#    the data on the leaf comes from the same class.
# 3. min_samples_split: The min_samples_split parameter specifies the minimum number of samples required to
#    split an internal leaf node. The default value for this parameter is 2, which means that an internal node
#    must have at least two samples before it can be split to have a more specific classification.

# n_jobs: The number of jobs to run in parallel
# random_state: Controls both the randomness of the bootstrapping of the samples used when building trees

def mlflow_record(n_estimator, max_depth, min_samples_split):
    with mlflow.start_run():
        # create a random forest classifier
        rf_clf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        n_jobs=2, random_state=0)
        # train the model with training_data
        rf_clf.fit(train_X, train_y)
        # predict testing data
        predicts_val = rf_clf.predict(test_X)

        # Generate a cm
        cm = confusion_matrix(test_y, predicts_val)
        model_accuracy = get_model_accuracy(cm)
        print("RandomForest model (n_estimator=%f, max_depth=%f, min_samples_split=%f):" % (n_estimator, max_depth,
                                                                                            min_samples_split))
        print("accuracy: %f" % model_accuracy)
        mlflow.log_param("n_estimator", n_estimator)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("model_accuracy", model_accuracy)
        mlflow.sklearn.log_model(rf_clf, "model")


n_estimator_list = [160, 180, 200]
max_depth_list = [15, 20, 25]
min_samples_split_list = [2, 2, 2]

for i in range(3):
    mlflow_record(n_estimator_list[i], max_depth_list[i], min_samples_split_list[i])
