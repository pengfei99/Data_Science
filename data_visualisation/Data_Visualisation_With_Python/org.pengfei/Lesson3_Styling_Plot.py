import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data_first150k.csv'
reviews=pd.read_csv(input_file,index_col=0)
reviews.head(3)

######## the default plot on points count
vin_points=reviews['points'].value_counts().sort_index()
#vin_points.plot.bar()
#plt.show()

######## It's too small to see, we want to make it big
# we use figsize(width, height) to define the size of plot
#vin_points.plot.bar(figsize=(12,6))
#plt.show()

######## Now, we want to change the color of bars
# we use color='color_value'
# color_value could be (mediumvioletred, red, yellow, etc)
# vin_points.plot.bar(figsize=(12,6),color='yellow')
# plt.show()

####### The text label are very hard to read, we can change the fontsize
#vin_points.plot.bar(figsize=(12,6),color='yellow',fontsize=16)
#plt.show()

###### Now we need to add a title to this plot
# we use title
#vin_points.plot.bar(figsize=(12, 6), color='mediumvioletred', fontsize=16, title='Rankings Given by Wine Magazine')
#plt.show()

"""
Pandas data visu is built based on matplotlib, so all the plot build with pandas can use also the function of matplotlib

The following example we will use matplotlib function to set title with a bigger font size
"""

#my_plot=vin_points.plot.bar(figsize=(12, 6), color='mediumvioletred', fontsize=16)
#my_plot.set_title("Rankings Given by Wine Magazine",fontsize=20)
#plt.show()

""" 
We can use another lib called seaborn which is also built based on matplotlib

"""

#my_plot=vin_points.plot.bar(figsize=(12, 6), color='mediumvioletred', fontsize=16)
#my_plot.set_title("Rankings Given by Wine Magazine",fontsize=20)
#sns.despine(bottom=True,left=True)
# sns.despine method to turn off the ugly black border.
#plt.show()


################################################################################
################ Exercises on pokemon #########################################
##############################################################################

pokemon_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon=pd.read_csv(pokemon_input_file,index_col=0)

#### Add title to plot
#attack_defense_plot=pokemon.plot.scatter(x='Attack',y='Defense',figsize=(16,9),title='Pokemon by Attack and Defense')
#plt.show()

#### change the histogram color
#print(pokemon.dtypes)
#print(pokemon['Total'].head())

#ax = pokemon['Total'].plot.hist(figsize=(12, 6),fontsize=14,bins=50,color='gray')
#ax.set_title('Pokemon by Stat Total', fontsize=20)
#plt.show()

#### bar plot of pokemon primary type without border
pokemon_type_plot = pokemon['Type 1'].value_counts().plot.bar(figsize=(12,6),fontsize=14)
pokemon_type_plot.set_title("Rankings Given by Wine Magazine",fontsize=20)
sns.despine(bottom=True, left=True)
plt.show()


