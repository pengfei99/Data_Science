# -*- coding: utf-8 -*-

import pandas as pd

##################################################################
###################Introduction ##################################
##################################################################
"""
In this lesson, we will look at two inter-related concepts, data types and missing data. This lesson draws from the Intro to 
data structures (https://pandas.pydata.org/pandas-docs/stable/dsintro.html) and Working with missing data 
(https://pandas.pydata.org/pandas-docs/stable/missing_data.html) sections of the comprehensive official tutorial.

"""

csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
df=pd.read_csv(csvFPath,index_col=0)


########################################################################
####################### Data types ####################################
######################################################################

# return the type of single column
# print(df.price.dtype)

# return the type of all columns
# print(df.dtypes)

"""
Data types tell us something about how pandas is storing the data internally. float64 means that it's using a 64-bit 
floating point number; int64 means a similarly sized integer instead, and so on.

One peculiarity to keep in mind (and on display very clearly here) is that columns consisting entirely of strings do 
not get their own type; they are instead given the object type.

It's possible to convert a column of one type into another wherever such a conversion makes sense by using the 
astype function. For example, we may transform the points column from its existing int64 data type into a float64 
data type:
"""

# print(df.points.dtype)
# the following command returns a series, it does not modify the origin data frame
# new_points=df.points.astype('float64')
# we could create a new column
# df['float_points']=new_points
# print(df.float_points.dtype)

"""
A DataFrame or Series index has its own dtype, too
"""

# print(df.index.dtype)

#########################################################################################
############################### Missing data ##########################################
########################################################################################

"""
Entries missing values are given the value NaN, short for "Not a Number". For technical reasons these NaN values are 
always of the float64 dtype.

pandas provides some methods specific to missing data. To select NaN entreis you can use pd.isnull (or its companion  
pd.notnull). This is meant to be used thusly:
"""
# empty_country=df[df.country.isnull()]
# print(empty_country.head())

"""
Replacing missing values is a common operation.  pandas provides a really handy method for this problem: fillna. 
fillna provides a few different strategies for mitigating such data. For example, we can simply replace each NaN with 
an  "Unknown":
"""
# fillna by default do not change the origin data frame, it returns a series, to do the chang in place
# you need to add the following option inplace=True
unknow_region_2=df.region_2.fillna("Unknow",inplace=True)
print(unknow_region_2)
print(df.region_2)

"""
Or we could fill each missing value with the first non-null value that appears sometime after the given record in the 
database. This is known as the backfill strategy:

fillna supports a few strategies for imputing missing values. For more on that read the official function documentation
(https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html).

Alternatively, we may have a non-null value that we would like to replace. For example, suppose that since this dataset 
was published, reviewer Kerin O'Keefe has changed her Twitter handle from @kerinokeefe to @kerino. One way to reflect 
this in the dataset is using the replace method:
"""

# like the fillna, by default replace do not change the origin data frame, it returns a series, to do the chang in place
# you need to add the following option
df.taster_twitter_handle.replace("@kerinokeefe", "@kerino",inplace=True)
print(df[df.taster_twitter_handle == '@kerino'].taster_twitter_handle)

"""
The replace method is worth mentioning here because it's handy for replacing missing data which is given some kind of 
sentinel value in the dataset: things like "Unknown", "Undisclosed", "Invalid", and so on.
"""

