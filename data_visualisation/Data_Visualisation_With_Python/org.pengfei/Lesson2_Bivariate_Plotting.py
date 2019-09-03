# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data_first150k.csv'
reviews=pd.read_csv(input_file,index_col=0)

#print(reviews.head())


#################################################
################Scatter plot ####################
###################################################

"""
The simplest bivariate plot is the lowly scatter plot. A simple scatter plot simply maps each variable of 
interest to a point in two-dimensional space. This is the result
"""
# the following line first filter rows which has price < 100, then sample 100 row
# at last show a scatter plot with x as vin price , y as vin score
#reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')
#plt.show()

"""
Note that in order to make effective use of this plot, we had to downsample our data, taking just 100 points 
from the full set. This is because naive scatter plots do not effectively treat points which map to the same place.

For example, if two wines, both costing 100 dollars, get a rating of 90, then the second one is overplotted 
onto the first one, and we add just one point to the plot.

This isn't a problem if it happens just a few times. But with enough points the distribution starts to look 
like a shapeless blob, and you lose the forest for the trees
"""

# without sampling
#reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')
#plt.show()

###########################################################
###################Hex plot#################################
###########################################################

"""
A hexplot aggregates points in space into hexagons, and then colorize those hexagons
"""

#reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
#plt.show()

"""
The data in this plot is directly comprable to the scatter plot from earlier, but the story it tells us 
is very different. The hexplot provides us with a much more useful view on the dataset, showing that the 
bottles of wine reviewed by Wine Magazine cluster around 87.5 points and around $20.

Hexplots and scatter plots can by applied to combinations of interval variables or ordinal categorical variables. 
To help aleviate overplotting, scatter plots (and, to a lesser extent, hexplots) benefit from variables which can 
take on a wide range of unique values.
"""

################################################################
######################Stacked plots#############################
################################################################

"""
Scatter plots and hex plots are new. But we can also use the simpler plots we saw in the last notebook.

The easiest way to modify them to support another visual variable is by using stacking. A stacked chart 
is one which plots the variables one on top of the other.

We'll use a supplemental selection of the five most common wines for this next section.
"""

# wine_counts_file='/home/pliu/data_set/pandas_data_visu/top-five-wine-score-counts.csv'
# wine_counts = pd.read_csv(wine_counts_file,index_col=0)
#
# print(wine_counts.head())
#
# wine_counts.plot.bar(stacked=True)
#
# plt.show()

"""
Stacked bar plots share the strengths and weaknesses of univariate bar charts. They work best for nominal 
categorical or small ordinal categorical variables.

Another simple example is the area plot, which lends itself very naturally to this form of manipulation:
"""

#wine_counts.plot.area()
#plt.show()

#wine_counts.plot.line()
#plt.show()

#####################################################################################
#######################Excercise on pokemen#######################################
#################################################################################


input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon=pd.read_csv(input_file,index_col=0)

##### Q1. scatter plot on attack and defense
# scatter plot for one hundred row
#pokemon.sample(100).plot.scatter(x='Attack',y='Defense')
# scatter plot for all row
#pokemon.plot.scatter(x='Attack',y='Defense')
#plt.show()

##### Q2. hex plot on attack and defense

#pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=20)
#plt.show()


##### Q3. bar stacked plot

#pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]

#print(pokemon_stats_legendary.head())

#pokemon_stats_legendary.plot.bar(stacked=True)

#plt.show()

##### Q4. line stacked plot

pokemon_stats_by_generation = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

pokemon_stats_by_generation.plot.line()

plt.show()




############################################################################
#################  Conclousion #########################################
#####################################################################

"""
1.Scatter plots and hex plots work best with a mixture of ordinal categorical and interval data.

2.Nominal categorical data makes sense in a (stacked) bar chart, but not in a (bivariate) line chart.

3.Interval data makes sense in a bivariate line chart, but not in a stacked bar chart.
"""
