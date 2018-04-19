from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score



def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    # build a model with max leaf nodes
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    # train the model with training data (featutre -> X,label-> y)
    model.fit(train_X, train_y)
    # predict testing data
    preds_val = model.predict(test_X)
    # calculate mae of testing data
    mae = mean_absolute_error(test_y, preds_val)
    return(mae)


def get_mae_random_forest(X,y):
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring='neg_mean_absolute_error').mean()


def split_data(input_file_path,featrue_data_names):
    data = pd.read_csv(input_file_path)
    feature_data = data[featrue_data_names]
    sale_price = data.SalePrice
    train_X, test_X, train_y, test_y = train_test_split(feature_data, sale_price, random_state=0)
    return train_X, test_X, train_y, test_y


def missing_value_score_dataset(train_X, test_X, train_y, test_y):
    model = RandomForestRegressor()
    model.fit(train_X,train_y)
    predications=model.predict(test_X)
    return mean_absolute_error(test_y,predications)


def missing_value_split_train_test_data(feature_data, label_data):
    train_X, test_X, train_y, test_y = train_test_split(feature_data, label_data, train_size=0.7, test_size=0.3, random_state=0)
    return train_X, test_X, train_y, test_y

# get a data frame with the required column in the given file
# Imputer the given column
def Patial_DP_get_data(input_file_path, cols_to_use):
    data = pd.read_csv(input_file_path)
    print(data.sample(5))
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y