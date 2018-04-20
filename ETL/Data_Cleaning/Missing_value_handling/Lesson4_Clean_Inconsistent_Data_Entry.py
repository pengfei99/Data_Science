import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process
from utils import *

np.random.seed(0)

# read in our data
suicide_attacks = pd.read_csv("/home/pliu/Downloads/python_data_cleaning/PakistanSuicideAttacks_v11.csv",encoding='Windows-1252')


# print (suicide_attacks.sample(5))
#######################################
####Clean column city #################
#####################################
cities = suicide_attacks['City'].unique()
cities.sort()
print(cities.size)

# convert all to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()

# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()

get_column_unique_entry_size(suicide_attacks,'City')
# cities = suicide_attacks['City'].unique()
# cities.sort()


# find 10 first matches which looks like d.i khan
#matches = fuzzywuzzy.process.extract("d.i khan",cities,limit=10,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

#print(matches)

# use the following function to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")

get_column_unique_entry_size(suicide_attacks,'City')

# use the following function to replace close matches to "Kuram agency" with "Kuram agency"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="Kuram agency")

cities=get_column_unique_entry_size(suicide_attacks, 'City')

####################################################################
#### clean the province column #####################################
###################################################################
province=get_column_unique_entry_size(suicide_attacks, 'Province')
print(province)

# convert all to lower case
suicide_attacks['Province']= suicide_attacks['Province'].str.lower()

# remove trailling space
suicide_attacks['Province']= suicide_attacks['Province'].str.strip()

# replace all matches "Baluchistan" to "Baluchistan"
replace_matches_in_column(df=suicide_attacks,column='Province', string_to_match="baluchistan")

province=get_column_unique_entry_size(suicide_attacks, 'Province')
print(province)