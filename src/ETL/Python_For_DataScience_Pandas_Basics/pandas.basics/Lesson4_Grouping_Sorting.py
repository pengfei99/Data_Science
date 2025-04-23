# -*- coding: utf-8 -*-

import pandas as pd

##################################################################
###################Introduction ##################################
##################################################################
"""
In this lesson, we will learn how to group and sort data in a dataframe

"""

csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
df=pd.read_csv(csvFPath,index_col=0)

######################################################################
##################### Grouping    ###############################
#####################################################################


"""
In lesson 3 , we have seen Maps, which allow us to transform data in a DataFrame or Series one value at a time for an 
entire column. However, often we want to group our data, and then do something specific to the group the data is in. 
To do this, we can use the groupby operation.

For example, one function we've been using heavily thus far is the value_counts function. We can replicate what 
value_counts does using groupby by doing the following:
"""

# print(df.groupby('points').points.count())

"""
groupby created a group of points which allotted the same point values to the given wines. Then, for each of these 
groups, we grabbed the points column and counted how many times it appeared.

We could image groupby divide the full dataframe into sub dataframe. In our example, the dataframe has been divided into 
21 dataframe (80~100). The first data frame includes all the rows where points=80, the second dataframe is where 
points=81, ... the last dataframe is where points=100.


value_counts is just a shortcut to this groupby operation. We can use any of the summary functions we've used before 
with this data. For example, to get the cheapest wine in each point value category, we can do the following:
"""

#print(df.groupby('points').price.min())

"""
As the groupby function returns a sub dataframe, we could use all operation of dataframe such as apply,
For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:
"""

#print(df.groupby('winery').apply(lambda df: df.title.iloc[0]))

"""
For even more fine-grained control, you can also group by more than one column. For an example, here's how we would 
pick out the first best wine (if there are two vins with the same points, take the first) by country and province:
"""

# the argmax function will return the row label of which contains the max value of the chosen column
# if there are many rows which contains the max value, it will return the first row label
# print(df.points.argmax())
# print(df[df['points']==100].points.head(10))

# best_vin_of_each_province=df.groupby(['country', 'province']).apply(lambda sub_df:sub_df.loc[sub_df.points.argmax()])
# print(best_vin_of_each_province.head(1))
# print(best_vin_of_each_province.count())

"""
Another groupby method worth mentioning is agg, which lets you run a bunch of different functions on your DataFrame 
simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:
"""

# price_stat_groupby_country=df.groupby(['country']).price.agg([len,min,max])
# print(price_stat_groupby_country)

# we can use different agg on different column
# points_price_stats=df.groupby(['country']).agg({'points':['min','max'],'price':[len,min]})
# print(points_price_stats)

#########################################################################################
################################ Multi- indexes #########################################
#######################################################################################

"""
In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index.  
groupby is slightly different in the fact that, depending on the operation we run, it will sometimes result in what is 
called a multi-index.

A multi-index differs from a regular index in that it has multiple levels. For example:
"""

country_province_reviews=df.groupby(['country','province']).description.agg([len])
# print(country_province_reviews)

"""
The type of multi index in pandas
mi = _.index
type(mi)
pandas.core.indexes.multi.MultiIndex


Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. 
They also require two levels of labels to retrieve a value, an operation that looks something like this. Dealing with 
multi-index output is a common "gotcha" for users new to pandas.

The use cases for a MultiIndex are detailed alongside detailed instructions on using them in the MultiIndex / Advanced 
(https://pandas.pydata.org/pandas-docs/stable/advanced.html). Selection (https://pandas.pydata.org/pandas-docs/stable/advanced.html) 
section of the pandas documentation.

However, in general the MultiIndex method you will use most often is the one for converting back to a regular index, 
the reset_index method:
"""

country_province_reviews.reset_index(inplace=True)
# print(country_province_reviews)



###########################################################################################
#################################### Sorting ############################################
#######################################################################################

"""
Looking again at country_province_reviews we can see that grouping returns data in index order, not in value order. 
That is to say, when outputting the result of a groupby, the order of the rows is dependent on the values in the index, 
not in the data.

To get data in the order want it in we can sort it ourselves. The sort_values method is handy for this.
"""

# print(country_province_reviews.sort_values(by='len'))

# by default the sort order is ascending, if you want decending, use the following option
# print(country_province_reviews.sort_values(by='len', ascending=False))

# we could also sort the index

print(country_province_reviews.sort_index())

# Finally, know that you can sort by more than one column at a time:

print(country_province_reviews.sort_values(by=['country','len']))
