# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import remean_points

##################################################################
###################Introduction ##################################
##################################################################
"""
In this lesson, we will learn how to do basic statistics on a dataframe

"""

csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
df=pd.read_csv(csvFPath,index_col=0)

######################################################################
#####################Summary functions ###############################
#####################################################################

"""
Pandas provides many simple "functions" which restructure the data in some useful way
For example, the describe method can do the basic statistic of each column of a data frame

"""

# vin_points_stats=df.points.describe()
# print(vin_points_stats)

"""
The describe method is type aware, for numeric column, it will give the following output

count    129971.000000 -> total values number of this column
mean         88.447138 -> mean value (not median value)
std           3.039730 -> standard deviation
min          80.000000 -> min value
25%          86.000000 -> the value at 25% (like the box plot)
50%          88.000000 -> the value at 50% (like the box plot, not the mean value)
75%          91.000000 -> the value at 75% (like the box plot)
max         100.000000 -> max value
"""

# taster_name_stats=df.taster_name.describe()
# print(taster_name_stats)

"""
For string column, it will give the following output

count         103727 -> total values number of this column
unique            19 -> unique values number
top       Roger Voss -> most frequent value
freq           25514 -> the frequence number of the most frequent value
"""


"""
If you want to get some particular simple summary statistic about a column in a DataFrame or a Series, there is usually 
a handful pandas function that makes it happen. For example, to see the mean of the points allotted (e.g. how well 
an averagely rated wine does), we can use the mean function:
"""

# average_vin_point=df.points.mean()
# print(average_vin_point)

"""
Get a array of unique value of a column
"""

# unique_taster_name=df.taster_name.unique()
# print(unique_taster_name)

"""
To see a list of unique values and how often they occur in the dataset, we can use the value_counts method:
it's like a select taster_name,count(taster_name) as count from data groupby taster_name orderby count desc;   
"""

# taster_name_count=df.taster_name.value_counts()
# print(taster_name_count)

###############################################################################
########### Apply custom function on dataframe  #############################
##############################################################################

"""
In pandasm, we have three functions which can apply functions on data

- Map: It iterates over each element of a series.
df[‘column1’].map(lambda x: 10+x), this will add 10 to each element of column1.
df[‘column2’].map(lambda x: ‘AV’+x), this will concatenate “AV“ at the beginning of each element of column2 (column format is string).

- Apply: As the name suggests, applies a function along any axis of the DataFrame.
df[[‘column1’,’column2’]].apply(sum), it will returns the sum of all the values of column1 and column2.

- ApplyMap: This helps to apply a function to each element of dataframe.
func = lambda x: x+2
df.applymap(func), it will add 2 to each element of dataframe (all columns of dataframe must be numeric type)
"""

"""
A "map" is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another 
set of values. In data science we often have a need for creating new representations from existing data, or for 
transforming data from the format it is in now to the format that we want it to be in later. Maps are what handle 
this work, making them extremely important for getting your work done!

There are two mapping functions that you will use often. The Series map is the first, and slightly simpler one. 
For example, suppose that we wanted to remean the scores the wines recieved to 0. We can do this as follows:
"""

# get the mean of points
# vin_points_mean=df.points.mean()

# use map function to do the data transformation
# actual_mean_diff=df.points.map(lambda point: point-vin_points_mean)

# print(actual_mean_diff)

"""
The function map takes every value in the column it is being called on and converts it some new value using a function 
you provide it. In our example, it's a lambda expression, point-vin_points_mean

map takes a Series as input. The DataFrame apply function can be used to do the same thing across columns, on the level 
of the entire dataset. Thus apply takes a DataFrame as input.
"""
review_points_mean=df.points.mean()
print(review_points_mean)

df['points']=df['points'].apply(remean_points, args = (review_points_mean,))

print(df.head(1))



