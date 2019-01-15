# -*- coding: utf-8 -*-
import string
import pandas as pd
import matplotlib.pyplot as plt

########################################################################################
########################## Introduction ###############################################
######################################################################################

"""
Remove duplication (Deduplication)

"Duplication" just means that you have repeated data in your dataset. This could be due to things like data entry errors 
or data collection methods. For example, if you're using a web scraper you may happen to scrape the same webpage more 
than once, or the same information from two different pages. Whatever the reason, duplication can lead you to make 
incorrect conclusions by leading you to believe that some observations are more common than they really are.

In this lesson we're going to learn how to find and remove duplicate records. (Removing duplicates is called 
"deduplication".) Here's a quick overview of what we'll be doing today.

- Visualizing duplication
- Finding & removing exact duplicates
- Finding & removing partial duplicates
"""

"""
The game dataset contains information on what games some Steam users bought and how many hours they spent playing them. 
The ign_data dataset contains information on reviews of different video games.
"""
game_input_file = "/home/pliu/Downloads/data_set/python_data_cleaning/Lesson7_remove_duplication/steam-200k.csv"
# the game csv does not have column names, so we add column names by using option names
game_df=pd.read_csv(game_input_file, names=["user_id", "game_title", "behavior_name", "value", "x"])
# remove the last column which is 0 for all rows
game_df=game_df.iloc[:,0:4]

ign_input_file = "/home/pliu/Downloads/data_set/python_data_cleaning/Lesson7_remove_duplication/ign.csv"
ign_df=pd.read_csv(ign_input_file,index_col=0)
# print(game_df.head())
# print(ign_df.head())

"""
What is duplication?

To start off, let's define what we mean by "duplication". Duplication can mean two slightly different things:

- More than one record that is exactly the same. This is what I call "exact duplication".
- More than one record associated with the same observation, but the values in the rows are not exactly the same. 
  This is what I call "partial duplication", but removing these types of duplicated records is also called 
  "record linkage".
  
In this lesson, we'll be talking about methods to identify and remove both of these types of duplication.
"""

#################################################################################################################
############################ Visualizing duplication ###########################################################
################################################################################################################

"""
You may have noticed that I tend to do a lot of visualizing as part of different data cleaning tasks, and deduplication 
is no exception! Let's take a look at how many rows are duplicated in our dataset and where they are. There are a couple 
reasons you may want to do this:

- See how much duplicated data you have. If you only have a couple duplicates, or even none, you can just move on 
  without worrying about them.

- See if there are any patterns in duplication. One fairly common pattern is that you'll see big blocks of duplicates 
  due to data being copy and pasted at the end of existing data. If that's the case, you can just remove those rows 
  from your dataframe and call it a day.

To plot duplicates, I'm first going to create a dataframe with 
1. a logical vector indicating whether or not a specific row is duplicated elsewhere in the dataset and 
2. a numeric vector of the index of each row. (I'm not using row numbers because if you're using the Tidyverse version 
   of a dataframe, they get removed whenever you manipulate the dataframe.) Then, I'm going to plot that information 
   so that each duplicated row shows up as a black line. Like so:
"""

# Create a new column called is_duplicate and add
game_df['is_duplicate'] = game_df.duplicated()
# print(game_df.head())
# get only duplicate rows
duplicate_game_df=game_df[game_df.is_duplicate==True]
print(duplicate_game_df.head())
duplicate_game_df.plot.box()
plt.show()


