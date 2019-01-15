# -*- coding: utf-8 -*-
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import outliers_z_score
from utils import outliers_modified_z_score
from utils import outliers_iqr
import seaborn as sns



#########################################################################################################
########################## Introduction ################################################################
########################################################################################################

"""
What's an outlier?

In plain English, it's a data point that's very far from the other data points.
Mathematically, an outlier is usually defined as an observation more than three standard deviations from the
mean (although sometimes you'll see 2.5 or 2 as well). It's not the only mathmatical definition, but since
it's the one that's used the most often that's what we'll use here as well.
"""
"""
Why care about outliers? There are a couple of reasons:

1. Outliers distort the picture of the data we obtain using descriptive statitics and data visualization. 
When our goal is to understand the data, it is often worthwhile to disregard outliers.
2. Outliers play havoc with many machine learning algorithms and statistical models. When our goal is to 
predict, our models are often improved by ignoring outliers.
3. Outliers can be exactly what we want to learn about, especially for tasks like anomaly detection.
"""
"""
What is Z-score?

The number of standard deviations away from the mean that a particular observation is. 
(You can see the mathematical definition http://mathworld.wolfram.com/z-Score.html, if you're curious.) A negative 
Z-score means an observation is below the mean, while a positive one means it above it. The further away from 0 the 
Z-Score is, the further away from the mean your observation is.
"""

"""
How to detect the outliers?

One of the easiest ways to see if your dataset has outliers is to plot your data. The type of plot you pick will 
depend on how many and what kind of variable you're interesting in looking at. In this lesson , we're going to 
look at three different plots.
"""

"""
A word of warning

None of the below methods will deliver the objective truth about which of a dataset’s observations are outliers, 
simply because there is no objective way of knowing whether something is truly an outlier or an honest-to-goodness 
data point your model should account for. It is a decision you must make subjectively, depending on the goals of 
uour analysis. Nevertheless, there is some guidance to be found in the accumulated wisdom of the field: these 
functions are a great way to start wondering about which points in your data should be treated as outliers.

"""

deputies_input_file="/home/pliu/Downloads/data_set/python_data_cleaning/Lesson6_outliers/deputies_dataset.csv"
dirty_deputies_input_file="/home/pliu/Downloads/data_set/python_data_cleaning/Lesson6_outliers/dirty_deputies_v2.csv"
# data_v1=pd.read_csv(deputies_input_file,index_col=0)
data_v2=pd.read_csv(dirty_deputies_input_file,index_col=0)

# print(data_v1.shape)
# print(data_v2.shape)

#print(data_v2.head())
#print(data_v2.dtypes)

################################################################################################
############################ Indentify outliers by using visualization ########################
#############################################################################################

"""
Univariate just means "one variable". When I'm looking for variation in just one variable, I like to use boxplots. 
This is because outliers are shown as points on a box plot, so it's easy to see if we have any outliers in our data. 
In this Lesson, I'll be using the pandas scatter plot.
"""

# print(data_v2['refund_value'].describe())

"""
The following data describes the refund_value stats, we could notice that 75% value is under 506.
And the standard deviation is 2076, we could set our outliers at 4200 which is 2*std.
count    339089.000000
mean        625.747421
std        2076.983477
min       -6100.000000
25%          50.000000
50%         150.000000
75%         506.190000
max      184500.000000

"""

######### Visu of all data set refund_value###############
# data_v2['refund_value'].plot.line()
# plt.show()

# does not work with screwed data
# data_v2['refund_value'].plot.hist()
# plt.show()

# The scatter plot shows better the result
# The thick line shows where the refund_value consentrate
# The dot line and point shows the outliers
# data_v2.plot.scatter(x='refund_value',y='refund_value')
# plt.show()

######### visu of supposed outlier refund_value #######
# We use the definion of the introduction, try to find out a outlier division
# rare_refund_value=data_v2[data_v2['refund_value']>4200.00]
# rare_refund_value['refund_value'].plot.line()
# plt.show()
# rare_refund_value['refund_value'].plot.hist()
# plt.show()

# we could see there are outliers inside outliers
# rare_refund_value.plot.scatter(x='refund_value', y='refund_value')
# plt.show()

############ visu with 2 variable

"""
In  the following example ,we use a group box plot to show the refund value group by the refund description
As the refund description is a categorical value. So the box plot is very easy to read and find out the outliers
"""
# sns.boxplot(x=data_v2['refund_value'],y=data_v2['refund_description'])
# plt.show()

"""
In the following example, we use a scatter plot to show the refund value group by party_nmembers, They are both numeric 
values. But the party_nmembers column need to be cleaned first. Because the datatype is object, not int 
"""
# We check the column datatype
#print(data_v2['party_nmembers'].dtypes)

"""
We check all the unique values in the column, and we find 3 value '15606','161914','Nan'. We need to remove the ', 
because it will cause problems when we convert the datatype from object to int. 
"""

# print(data_v2['party_nmembers'].unique())
# data_v2['party_nmembers']=data_v2['party_nmembers'].replace('15606',15606).replace('161914',161914).replace('Nan',None)
# print(data_v2['party_nmembers'].unique())

"""
Now all the values in party_nmembers is in good shape, we could start the eliminate the nan and convert it to int
In the following example, we just choose the row which party_nmembers is not null. There is another solution, replace 
the NaN by 0. But after think about it, party_nmembers can never be 0. It must be some kind errors when the value of 
party_nmembers is NaN. So replace it with 0 will make the stats not correspond the reality. So we will drop all the NaN 
rows 
"""
# refund_value_of_party_nmembers=data_v2[data_v2['party_nmembers'].notnull()].loc[:,['refund_value','party_nmembers']]
#
# refund_value_of_party_nmembers['party_nmembers']=refund_value_of_party_nmembers['party_nmembers'].astype(int)
# print(refund_value_of_party_nmembers.dtypes)

"""
Now the dataframe is ready, we can do our visulization
"""

# refund_value_of_party_nmembers.plot.scatter(x='party_nmembers', y='refund_value')
# plt.show()

"""
This plot shows us that there are outliers in refund value all along the different number of members in each party. 
The reason we see strong vertical lines here is because all the politicians from the same political party, and thus 
all their expenses, are associated with a specific number of party members.
"""


############################################################################################
################### Indentify outliers by using Z score method  ###########################
###########################################################################################
"""
The Z-score, or standard score, is a way of describing a data point in terms of its relationship to the mean 
and standard deviation of a group of points. Taking a Z-score is simply mapping the data onto a distribution 
whose mean is defined as 0 and whose standard deviation is defined as 1.

The goal of taking Z-scores is to remove the effects of the location and scale of the data, allowing different 
datasets to be compared directly. The intuition behind the Z-score method of outlier detection is that, once we’ve 
centred and rescaled the data, anything that is too far from zero (the threshold is usually a Z-score of 3 or -3) 
should be considered an outlier.
"""

# refund_value=data_v2['refund_value']
#
# outliers=outliers_z_score(refund_value)
# print(outliers)
#
#
#
# mod_outliers=outliers_modified_z_score(refund_value)
# print(mod_outliers)
#
# iqr_outliers=outliers_iqr(refund_value)
#
# print(iqr_outliers)
###########################

#####################Conclusion ##########################################
"""
It's a subjective desicion, 
"""



"""
Dataframe column name and type

deputy_state           object
political_party        object
refund_description     object
company_name           object
company_id            float64
refund_date            object
refund_value          float64
party_pg               object
party_en               object
party_tse              object
party_regdate          object
party_nmembers         object
party_ideology1        object
party_ideology2        object
party_ideology3        object
party_ideology4        object
party_position         object
"""


###################################################################
######### Documentation########################################
#############################################################

"""
1. find outliers
http://ocefpaf.github.io/python4oceanographers/blog/2013/05/20/spikes/

2. Remove outliers 
https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/


3. Kaggle remove outliers with R
https://www.kaggle.com/rtatman/data-cleaning-challenge-outliers/?utm_medium=email&utm_source=mailchimp&utm_campaign=5DDC-data-cleaning-R
"""
