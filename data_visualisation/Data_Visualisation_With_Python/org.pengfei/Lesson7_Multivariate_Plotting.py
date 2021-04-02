# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
# from utils import isDecimal

footballer_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/footballer.csv'
df=pd.read_csv(footballer_input_file,index_col=0,encoding='utf8')
# print(df['Value'].head(5))
# break the Value column(€95.5M) into two columns, Unit column can be M (million) or 0
# The Value (M) column represents the player's value in Million,
# example €95.5M -> M, 95.5
# €0 -> 0, 0
#print(df.dtypes)
footballers= df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0,
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
#print(footballers[footballers['Unit']=='0']['Value'].head(5))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M',
                                    footballers['Value (M)'],
                                    footballers['Value (M)']/1000)
# replace the old value column by Value (M) column
# replace the old position column by the first element in Preferred Position
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])
# print(footballers[footballers['Unit']=='M']['Value (M)'].head(5))
# print(footballers[footballers['Unit']=='0']['Value (M)'].head(5))

#print(footballers['Unit'].head(5))

#print(footballers.head())

"""
Adding more visual variables

The most obvious way to plot lots of variables is to augement the visualizations we've been using thus far with even 
more visual variables. A visual variable is any visual dimension or marker that we can use to perceptually distinguish 
two data elements from one another. Examples include size, color, shape, and one, two, and even three dimensional position.

"Good" multivariate data displays are ones that make efficient, easily-interpretable use of these parameters.
"""
##########################################################################
######### Multivariate scatter plots  #####################################
#########################################################################
"""
Let's look at some examples. We'll start with the scatter plot. Supose that we are interested in seeing which 
type of offensive players tends to get paid the most: the striker, the right-winger, or the left-winger.

"""
# use color for hue distinction, the default setting
#sns.lmplot(x='Value',y='Overall',hue='Position',data=footballers.loc[footballers['Position'].isin(['ST','RW','LW'])],fit_reg=False)
#plt.show()

"""
This scatterplot uses three visual variables. The horizontal position (x-value) tracks the Value of the player 
(how well they are paid). The vertical position (y-value) tracks the Overall score of the player across all attributes. 
And the color (the hue parameter) tracks which of the three categories of interest the player the point represents is in.

The new variable in this chart is color. Color provides an aesthetically pleasing visual, but it's tricky to use. 
Looking at this scatter plot we see the same overplotting issue we saw in previous sections. But we no longer have an 
easy solution, like using a hex plot, because color doesn't make sense in that setting.

Another example visual variable is shape. Shape controls, well, the shape of the marker:
"""

# use shape for hue distinction, use markers to specifies which marker for which position
# o-> ST, x->RW, *->LW
# sns.lmplot(x='Value', y='Overall', markers=['o', 'x', '*'], hue='Position',
#            data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],
#            fit_reg=False
#           )
#
# plt.show()

#########################################################################################
####################Grouped box plot ###################################################
#######################################################################################

"""
Another demonstrative plot is the grouped box plot. This plot takes advantage of grouping. 
Suppose we're interested in the following question: do Strikers score higher on "Aggression" than Goalkeepers do?
"""
# we filter the data set with only ST and GK
# we keep only 4 columns
f = (footballers
          .loc[footballers['Position'].isin(['ST', 'GK'])]
          .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
     )
print(f.head(5))
# we filter the data set with only overall score between 80 and 85
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)
#
sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
plt.show()

"""
As you can see, this plot demonstrates conclusively that within our datasets goalkeepers (at least, those with an 
overall score between 80 and 85) have much lower Aggression scores than Strikers do.

In this plot, the horizontal axis encodes the Overall score, the vertical axis encodes the Aggression score, and the 
grouping encodes the Position.

Grouping is an extremely communicative visual variable: it makes this chart very easy to interpret. However, it 
has very low cardinality: it's very hard to use groups to fit more than a handful of categorical values. In this 
plot we've chosen just two player positions and five Overall player scores and the visualization is already rather 
crowded. Overall, grouping is very similar to faceting in terms of what it can and can't do.
"""

"""
Summarization

It is difficult to squeeze enough dimensions onto a plot without hurting its interpretability. Very busy plots are 
naturally very hard to interpret. Hence highly multivariate can be difficult to use.

Another way to plot many dataset features while circumnavigating this problem is to use summarization. Summarization 
is the creation and addition of new variables by mixing and matching the information provided in the old ones.

Summarization is a useful technique in data visualization because it allows us to "boil down" potentially very 
complicated relationships into simpler ones.
"""

##########################################################################
#################### Heatmap ############################################
#########################################################################

# 1.filter the data set with only five columns (e.g. 'Acceleration', 'Aggression', 'Agility', 'Balance',
# 'Ball control').
# 2. Use applymap to convert all cell values from string to int, if can't convert, put null instead
# lambda v is a function declared by using a lambda operator
# 3. drop all null cell
# 4. do the correlation

# f = (
#     footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
#         .applymap(lambda v: int(v) if isDecimal(v) else np.nan)
#         .dropna()
# ).corr()
#
# # generate the heat map
# sns.heatmap(f, annot=True)
# plt.show()

"""
Each cell in this plot is the intersection of two variables; its color and label together indicate the amount of 
correlation between the two variables (how likely both variables are the increase or decrease at the same time). 
For example, in this dataset Agility and Acceleration are highly correlated, while Aggression and Balanced are very 
uncorrelated.

A correlation plot is a specific kind of heatmap. A heatmap maps one particular fact (in this case, correlation) 
about every pair of variables you chose from a dataset.

The number of correlation is between 0 and 1, 1 means perfect correlated, 0 means not correlated at all. 
"""

############################################################################
########## Parallel Coordinates ###########################################
#########################################################################

"""
A parallel coordinates plot provides another way of visualizing data across many variables.
"""
"""
1. loc gets rows (or columns) with particular labels from the index.
2. iloc gets rows (or columns) at particular positions in the index (so it only takes integers).

>>> s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
>>> s
49   NaN
48   NaN
47   NaN
46   NaN
45   NaN
1    NaN
2    NaN
3    NaN
4    NaN
5    NaN

>>> s.iloc[:3] # slice from the begining to three rows, so 3 is not the index of the row, it's the position in the row
49   NaN
48   NaN
47   NaN

>>> s.loc[:3] # slice up to and including label 3, so 3 is the index of the row
49   NaN
48   NaN
47   NaN
46   NaN
45   NaN
1    NaN
2    NaN
3    NaN

>>> s.iloc[:6]
49   NaN
48   NaN
47   NaN
46   NaN
45   NaN
1    NaN

>>> s.loc[:6] # because no index 6 in the dataframe
KeyError: 6
"""
#1. we slice the column between position 12 and 17 (e.g. Acceleration, Aggression, Agility, Balance Ball control)
#2. we filter the data set where poisition is ST or GK
#3. convert all cell value into int, if can't replace by null
#4. drop all null cell
# f = (
#     footballers.iloc[:, 12:17]
#         .loc[footballers['Position'].isin(['ST', 'GK'])]
#         .applymap(lambda v: int(v) if isDecimal(v) else np.nan)
#         .dropna()
# )
# f['Position'] = footballers['Position']
# f = f.sample(200)
#
# parallel_coordinates(f, 'Position')
#
# plt.show()

"""
In the visualization above we've plotted a sample of 200 goalkeepers (in dark green) and strikers (in light green) 
across our five variables of interest.

Parallel coordinates plots are great for determining how distinguishable different classes are in the data. 
They standardize the variables from top to bottom... In this case, we see that strikers are almost uniformally 
higher rated on all of the variables we've chosen, meaning these two classes of players are very easy to distinguish.
"""
#########################################################################################
####################Pokemon Exercises##################################################
###################################################################################

"""
1. What are three techniques for creating multivariate data visualziations? 
 # My Answers (1. multivariate scatter plot, 2. grouped box plot, 3. heatmap, 4. parallel coordiantes)
 The three techniques we have covered in this tutorial are faceting, using more visual variables, and summarization
2. Name three examples of visual variables. (HP, Attack, Defense)
 Some examples of visual variables are shape, color, size, x-position, y-position, and grouping. However there are many more that are possible!
3. How does summarization in data visualization work?
In data visualization, summarization works by compressing complex data into simpler, easier-to-plot indicators.
"""

pokemon_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon=pd.read_csv(pokemon_input_file,index_col=0)

# Q1 create a multivariate scatter plot with x= attack, y= defense, hue=legendary
# sns.lmplot(x='Attack', y='Defense', markers=['o', 'x'], hue='Legendary',
#            data=pokemon,
#            fit_reg=False
#           )
#
# plt.show()

# Q2 create a grouped box plot, with x= generation, y= total, hue=legendary
# sns.boxplot(x='Generation',y='Total', hue='Legendary',data=pokemon)
# plt.show()

# Q3 create a heatmap with HP,Attack, Sp. Atk, Defense, Sp. Def, speed
# f = (
#     pokemon.loc[:,['HP', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Speed']]
#             .applymap(lambda v: int(v) if isDecimal(v) else np.nan)
#             .dropna()
# ).corr()
#
# sns.heatmap(data=f,annot=True)
# plt.show()

# Q4 create a parallel coordinates with pokemon type in Fighting, Psychic
# f = (
#     pokemon[pokemon['Type 1'].isin(["Psychic","Fighting"])]
#         .loc[:, ['Type 1', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def']]
# )
#
# parallel_coordinates(f, 'Type 1')
# plt.show()

"""
Conclusion

In this tutorial we followed up on faceting, covered in the last section, by diving into two other multivariate data 
visualization techniques.

The first technique, adding more visual variables, results in more complicated but potentially more detailed plots. 
The second technique, summarization, compresses variable information to a summary statistic, resulting in a simple 
output—albeit at the cost of expressiveness.

Faceting, adding visual variables, and summarization are the three multivariate techniques that we will 
cover in this tutorial.

The rest of the material in this tutorial is optional. In the next section we will learn to use plotly, 
a very popular interactive visualization library that builds on these libraries.
"""



