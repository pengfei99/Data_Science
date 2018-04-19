from utils import Patial_DP_get_data
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

"""
What are Partial Dependence Plots?

Some people complain machine learning models are black boxes. These people will argue we cannot see how these models 
are working on any given dataset, so we can neither extract insight nor identify problems with the model.

By and large, people making this claim are unfamiliar with partial dependence plots. Partial dependence plots 
show how each variable or predictor affects the model's predictions. This is useful for questions like:

How much of wage differences between men and women are due solely to gender, as opposed to differences in education 
backgrounds or work experience?

Controlling for house characteristics, what impact do longitude and latitude have on home prices? To restate this, 
we want to understand how similarly sized houses would be priced in different areas, even if the homes actually at 
these sites are different sizes.

Are health differences between two groups due to differences in their diets, or due to other factors?

If you are familiar with linear or logistic regression models, partial dependence plots can be interepreted similarly 
to the coefficients in those models. But partial dependence plots can capture more complex patterns from your data, 
and they can be used with any model. If you aren't familiar with linear or logistic regressions, don't get caught up 
on that comparison.

We will show a couple examples below, explain what they mean, and then talk about the code.
"""
#################################################
######## melborn house price example #############
#################################################

"""
Interpreting Partial Dependence Plots

We'll start with 2 partial dependence plots showing the relationship (according to our model) between Price and 
a couple variables from the Melbourne Housing dataset. We'll walk through how these plots are created and interpreted.
"""
input_file_path = "/home/pliu/Downloads/data_set/python_ml/melb_data.csv"

cols_to_use = ['Distance', 'Landsize', 'BuildingArea']

X, y = Patial_DP_get_data(input_file_path, cols_to_use)
# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
xgb_model = GradientBoostingRegressor()

# train the model as usual
xgb_model.fit(X,y)

print(X)

# Here we make the plot
my_plots = plot_partial_dependence(xgb_model,
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
plt.show()

"""
The left plot shows the partial dependence between our target, Sales Price, and the distance variable. Distance in this 
dataset measures the distance to Melbourne's central business district.

The partial dependence plot is calculated only after the model has been fit. The model is fit on real data. 
In that real data, houses in different parts of town may differ in myriad ways (different ages, sizes, etc.) But after 
the model is fit, we could start by taking all the characteristics of a single house. Say, a house with 2 bedrooms, 
2 bathrooms, a large lot, an age of 10 years, etc.

We then use the model to predict the price of that house, but we change the distance variable before making a prediction. 
We first predict the price for that house when sitting distance to 4. We then predict it's price setting distance to 5. 
Then predict again for 6. And so on. We trace out how predicted price changes (on the vertical axis) as we move from 
small values of distance to large values (on the horizontal axis).

In this description, we used only a single house. But because of interactions, the partial dependence plot for a single 
house may be atypical. So, instead we repeat that mental experiment with multiple houses, and we plot the average 
predicted price on the vertical axis. You'll see some negative numbers. That doesn't mean the price would sell for 
a negative price. Instead it means the prices would have been less than the actual average price for that distance.

In the left graph, we see house prices fall as we get further from the central business distract. Though there seems to 
be a nice suburb about 16 kilometers out, where home prices are higher than many nearer and further suburbs.

The right graph shows the impact of building area, which is interpreted similarly. A larger building area means higher 
prices.

These plots are useful both to extract insights, as well as to sanity check that your model is learning something you 
think is sensible.
"""


"""
Some tips related to plot_partial_dependence:

-The features are the column numbers from the X array or dataframe that you wish to have plotted. This starts to look bad
beyond 2 or 3 variables. You could make repeated calls to plot 2 or 3 at a time.

-There are options to establish what points on the horizontal axis are plotted. The simplest is grid_resolution which 
we use to determine how many different points are plotted. These plots tend to look jagged as that value increases, 
because you will pick up lots of randomness or noise in your model. It's best not to take the small or jagged 
fluctuations too literally. Smaller values of grid_resolution smooth this out. It's also much less of an issue for 
datasets with many rows.

-There is a function called partial_dependence to get the raw data making up this plot, rather than making the visual 
plot itself. This is useful if you want to control how it is visualized using a plotting package like Seaborn. With 
moderate effort, you could make much nicer looking plots.
"""
#################################################
########## Titanic surviver example #############
#################################################

# titanic_data = pd.read_csv('/home/pliu/Downloads/data_set/python_ml/titanic_train.csv')
# print(titanic_data.sample(5))
# titanic_y = titanic_data.Survived
# clf = GradientBoostingClassifier()
# titanic_X_colns = ['PassengerId','Age', 'Fare',]
# titanic_X = titanic_data[titanic_X_colns]
# my_imputer = Imputer()
# imputed_titanic_X = my_imputer.fit_transform(titanic_X)
#
# clf.fit(imputed_titanic_X, titanic_y)
# titanic_plots = plot_partial_dependence(clf, features=[1,2], X=imputed_titanic_X,
#                                         feature_names=titanic_X_colns, grid_resolution=8)
#
# plt.show()

"""
The partial dependence plot seem surprising at first glance. But they show some interesting insights:

-Being young increased your odds of survival. This is consistent with historical recountings that they got women and 
children off the Titanic first.

-People who paid more had better odds of survival. It turns out that higher fares got you a cabin that was closer to 
the top of the boat, and may have given you better odds of getting a life-boat.
"""