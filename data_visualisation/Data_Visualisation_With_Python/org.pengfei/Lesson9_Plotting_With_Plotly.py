import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt


##################################################################################################
#####################Incomplete, need to revisit #############################################
############################################################################################
"""
Introduction to plotly

So far in this tutorial we have been using seaborn and pandas, two mature libraries designed around matplotlib. 
These libraries all focus on building "static" visualizations: visualizations that have no moving parts. In other words, 
all of the plots we've built thus far could appear in a dead-tree journal article.

The web unlocks a lot of possibilities when it comes to interactivity and animations. There are a number of plotting 
libraries available which try to provide these features.

In this section we will examine plotly, an open-source plotting library that's one of the most popular of these 
libraries.
"""

input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data-130k-v2.csv'

reviews=pd.read_csv(input_file, index_col=0)
# print(reviews.dtypes)
# print(reviews.shape)
# print(reviews.isnull().sum())
# print(reviews.head(5))

# This iplot only works for IPython notebook, need to change
iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])
plt.show()
