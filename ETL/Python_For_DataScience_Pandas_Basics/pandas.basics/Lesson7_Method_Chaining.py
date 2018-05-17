# -*- coding: utf-8 -*-

import pandas as pd
from utils import name_index

##################################################################
###################Introduction ##################################
##################################################################

"""
In this lesson, we will see how to chain method for transform data in dataframe. You can find more detail in 
"Method chaining" section of the Advanced Pandas tutorial.

There is a great tutorial (https://tomaugspurger.github.io/method-chaining.html) for more details

Why method chaining?

Method chaining is the last topic we will cover in this first track of the Advanced Pandas tutorial. It is also the only 
section of this tutorial which is a technique or a pattern, not a function or variable.

Method chaining is a methodology for performing operations on a DataFrame or Series that emphasizes continuity. 

"""

vin_path='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
vin_df=pd.read_csv(vin_path,index_col=0)

ramen_path='/home/pliu/Downloads/data_set/pandas_basic/ramen-ratings.csv'
ramen_df=pd.read_csv(ramen_path)

# print(ramen_df.head(1))
# print(vin_df.head(1))

"""
To demonstrate what I mean, here's a data cleaning and dropping operation (which you should be familiar with from the 
last section) done two different ways:
"""

# get the ramen score column, starts value here is object, we can't do stats on object
# so we need to convert the object column to float64 type
# stars = ramen_df['Stars']

# replace unrated value by none, then drop all rows which is none
# na_stars = stars.replace('Unrated', None).dropna()
# convert the column to float64
# float_stars = na_stars.astype('float64')
# show the stats, we cloud notice that the stars value is between 0 and 5
# print(float_stars.describe())
# print(float_stars.max())


"""
The above code has many variables and assigning values, we could skip all that by using the method chaining
"""

# result=ramen_df['Stars'].replace('Unrated', None).dropna().astype('float64')
# print(result.describe())
# print(result.max())

"""
Most pandas operations can written in a method chaining style, and in the last couple years or so pandas has added more 
and more tools for making these sorts of statements easier to write. This paradigm comes to us from the R programming 
language—specifically, the dpyler module, part of the "Tidyverse".

Method chaining is advantageous for several reasons. One is that it lessens the need for creating and mentally tracking 
temporary variables. Another is that it emphasizes a correctly structured interative approach to working with data, 
where each operation is a "next step" after the last. Debugging is easy: just comment out operations that don't work 
until you get to one that does, and then start stepping forward again. And it looks kind of cool.
"""

############################################################################################
########################## Assign and pipe #################################################
#########################################################################################

######################### Assign ##############################################
"""
Now that we've learned all these ways of manipulating data with pandas, we're ready to take advantage of method 
chaining to write clear, clean data manipulation code. Now I'll introduce three additional methods useful for 
coding in this style.

The first of these is assign. The assign method lets you create new columns or modify old ones inside of a DataFrame 
inline. For example, to fill the region_1 field with the province field wherever the region_1 is null (useful if we're 
mixing in our own categories), we would do:
"""
# beware that, in the province column, we still have 63 nan cell, so after the apply, we will still
# have 63 nan cell, but before we have 21247 which is much worse.

# the assign do not change the origin dataframe, it returns a new data frame
# na_df=vin_df.assign(
#     region_1=vin_df.apply(lambda srs: srs.region_1 if pd.notnull(srs.region_1) else srs.province, axis='columns'))
#
# print(vin_df['province'].isnull().sum())
# print(vin_df['region_1'].isnull().sum())
# print(na_df['region_1'].isnull().sum())

"""
The above assign command equals the following 
"""

# vin_df['region_1'] = vin_df.apply(lambda srs: srs.region_1 if pd.notnull(srs.region_1) else srs.province, axis='columns')
#
# print(vin_df['region_1'].isnull().sum())

"""
You can modify as many old columns and create as many new ones as you'd like with assign, but it does have the 
limitation that the column being modified must not have any reserved characters like periods (.) or spaces () in the 
name.

"""


################################# Pipe ##############################################################

"""
The next method to know is pipe. pipe is a little mind-bending: it lets you perform an operation on the entire 
DataFrame at once, and replaces the current DataFrame which the output of your pipe.

For example, one way to change the give the DataFrame index a new name would be to do:
"""

vin_df.pipe(name_index)

print(vin_df.head(1))

"""
pipe is a power tool: it comes in handy when you're performing very intricate operations on your DataFrame. 
You won't need it often, but it'll be super useful when you do.
"""