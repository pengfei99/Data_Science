import pandas as pd
import numpy as np


#############################################################
############ Introduction ###################################
#############################################################

"""
In almost all data sets, you will encouter the missing value problems, you only have two choice if you want
to feed the data to your machine learning model, 

1. drop the column or row which contains the missing value
2. Fill the cell with some estimate value (mean). 

In this Lesson, we will see 

1. How to identify missing values
2. Determine why the values is missing (is it random)?
3. Guess(Impute) the values that are misssing
"""

##########################################
# Prepare data set #######################
##########################################

nfl_data = pd.read_csv("/home/pliu/data_set/python_data_set/python_data_cleaning/day1_missing_value/NFL_Play_by_Play_2009-2017.csv")
sf_permits = pd.read_csv("/home/pliu/data_set/python_data_set/python_data_cleaning/day1_missing_value/Building_Permits.csv")

#set random number generator seed, to have the same sample in every execution of the code
np.random.seed(0)

# get 5 sample of the nfl data set
print(nfl_data.sample(5))



##########################################
# Get missing value number ###############
##########################################

nfl_missing_value_count=nfl_data.isnull().sum()

# look at the number of missing value in the first ten columns
print(nfl_missing_value_count[0:10])

# get the total missing value percentage
# nfl_data.shape => (407688, 102) 407688 is rwo number, 102 is the column number
nfl_total_cells=np.product(nfl_data.shape)
# print(nfl_total_cells)
nfl_total_missing=nfl_missing_value_count.sum()
# print(nfl_total_missing)

nfl_miss_percent=float(nfl_total_missing)/float(nfl_total_cells)

print("Total missing percentage : "+str(nfl_miss_percent*100))

########################################
####Find out why the value is missing ##
#######################################

"""
Value missing because of :

1. not recorded -> need to estimate a value and replace it
2. not exist -> no need to guess the value. Keep it as NaN.

For example, in nfl data set 

By looking at (https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016), I can see that this column has information
on the number of seconds left in the game when the play was made. This means that these values are probably missing 
because they were not recorded, rather than because they don't exist. So, it would make sense for us to try and guess 
what they should be rather than just leaving them as NA's.

On the other hand, there are other fields, like `PenalizedTeam` that also have lot of missing fields. In this case, 
though, the field is missing because if there was no penalty then it doesn't make sense to say *which* team was 
penalized. For this column, it would make more sense to either leave it empty or to add a third value like "neither" 
and use that to replace the NA's.
"""

"""
Is your data MAR (Missing At Random)?

For the values that does not exist in real world (e.g. 3 room column of a two room appartment), they are 
not missing at Random.

For the values that does not recorded in the data set, but it exits in real world. For example, the
age or sex of a patient, we know it must exist. If these kind of values is missing, we can say that our
data is missing at random.

But, most of times, it's hard to determine a value is missing because of not recorded or not exist.
There is a good paper(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4121561/) which explains the difficulties.
"""

#####################################
#######Drop missing value ###########
####################################

# remove all the rows that contain a missing value
nfl_data.dropna()

# remove all columns with at least one missing value
nfl_columns_with_na_dropped = nfl_data.dropna(axis=1)
nfl_columns_with_na_dropped.head()

# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % nfl_columns_with_na_dropped.shape[1])

##############################################
### Filling in missing value automatically ##
#############################################

# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
print(subset_nfl_data.sample(5))

# replace all NA's with 0
subset_nfl_data.fillna(0)

# replace all NA's the value with the value that comes directly after it (next row) in the same column,
# then replace all the reamining na's with 0
subset_nfl_data.head()
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)