# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



"""
So far in this tutorial we've been plotting data in one (univariate) or two (bivariate) dimensions, 
and we've learned how plotting in seaborn works. In this section we'll dive deeper into seaborn by exploring faceting.

Faceting is the act of breaking data variables up across multiple subplots, and combining those subplots into a 
single figure. So instead of one bar chart, we might have, say, four, arranged together in a grid.

In this notebook we'll put this technique in action, and see why it's so useful.
"""

footballer_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/footballer.csv'
footballer=pd.read_csv(footballer_input_file,index_col=0,encoding='utf8')

#print(footballer.shape)
#print(footballer.isnull().sum())
#print(footballer.head(5))

#print(footballer.Nationality.unique())


##########################################################################################################
######################Facet Grid########################################################################
##############################################################################################

"""
The FacetGrid

The core seaborn utility for faceting is the FacetGrid. A FacetGrid is an object which stores some information on 
how you want to break up your data visualization.

For example, suppose that we're interested in (as in the previous notebook) comparing strikers and goalkeepers in 
some way. To do this, we can create a FacetGrid with our data, telling it that we want to break the Position variable 
down by col (column).

Since we're zeroing in on just two positions in particular, this results in a pair of grids ready for us to "do" 
something with them:
"""
# compare the overall score of players from Portugal and Brazil
#df = footballer[footballer['Nationality'].isin(['Portugal','Brazil'])]
#g = sns.FacetGrid(df,col="Nationality")
#g.map(sns.kdeplot,"Overall")
#plt.show()


#chinese_foot = footballer[footballer['Nationality'].isin(['Hong Kong'])]
#print(chinese_foot.Overall.describe())

# we can do it for more nationalities
# df6 = footballer[footballer['Nationality'].isin(['Portugal','Brazil','France','England','Germany','Spain'])]
# gAll=sns.FacetGrid(df6,col="Nationality",col_wrap=3)
# gAll.map(sns.kdeplot,"Overall")
# plt.show()

"""
So far we've been dealing exclusively with one col (column) of data. The "grid" in FacetGrid, however, refers to the 
ability to lay data out by row and column.

For example, suppose we're interested in comparing the talent distribution for across rival clubs Real Madrid, 
Atletico Madrid, and FC Barcelona.

As the plot below demonstrates, we can achieve this by passing row=Nationality and col=Club parameters into the plot.
"""
# bivariate facet grid without order
df6 = footballer[footballer['Nationality'].isin(['France','Spain'])]
dfclub = df6[df6['Club'].isin(['Real Madrid CF', 'FC Barcelona'])]
#
gclub= sns.FacetGrid(dfclub,row="Nationality",col="Club")
gclub.map(sns.violinplot,"Overall")
plt.show()

# bivariate facet grid with order
"""
FacetGrid orders the subplots effectively arbitrarily by default. To specify your own ordering explicitly, 
pass the appropriate argument to the row_order and col_order parameters.
"""
# df6 = footballer[footballer['Nationality'].isin(['France','Spain'])]
# dfclub = df6[df6['Club'].isin(['Real Madrid CF', 'FC Barcelona'])]
# gclub_order=sns.FacetGrid(dfclub,row="Nationality",col="Club", row_order=['France','Spain'],
#                   col_order=['FC Barcelona', 'Real Madrid CF'])
# gclub_order.map(sns.violinplot,"Overall")
# plt.show()

"""
Why facet?

In a nutshell, faceting is the easiest way to make your data visualization multivariate.

Faceting is multivariate because after laying out one (categorical) variable in the rows and another (categorical) 
variable in the columns, we are already at two variables accounted for before regular plotting has even begun.

And faceting is easy because transitioning from plotting a kdeplot to gridding them out, as here, is very simple. 
It doesn't require learning any new visualization techniques. The limitations are the same ones that held for the 
plots you use inside.

Faceting does have some important limitations however. It can only be used to break data out across singular or 
paired categorical variables with very low numeracy—any more than five or so dimensions in the grid, and the plots 
become too small (or involve a lot of scrolling). Additionally it involves choosing (or letting Python) an order 
to plot in, but with nominal categorical variables that choice is distractingly arbitrary.

Nevertheless, faceting is an extremely useful and applicable tool to have in your toolbox.
"""

#################################################################################################
#################### Pair plot ################################################################
#############################################################################################

"""
Now that we understand faceting, it's worth taking a quick once-over of the seaborn pairplot function.

pairplot is a very useful and widely used seaborn method for faceting variables (as opposed to variable values). 
You pass it a pandas DataFrame in the right shape, and it returns you a gridded result of your variable values:
"""

# sns.pairplot(footballer[['Overall','Potential','Age']])
# plt.show()


"""
By default pairplot will return scatter plots in the main entries and a histogram in the diagonal. 
pairplot is oftentimes the first thing that a data scientist will throw at their data, and it works 
fantastically well in that capacity, even if sometimes the scatter-and-histogram approach isn't quite 
appropriate, given the data types.
"""


###############################################################################
########### Pokemon Examples ########################################################
#########################################################################

"""
As in previous notebooks, let's now test ourselves by answering some questions about the plots we've used in this 
section. 

1. Suppose that we create an n by n FacetGrid. How big can n get?
You should try to keep your grid variables down to five or so. Otherwise the plots get too small.

2. What are the two things about faceting which make it appealing?
It's (1) a multivariate technique which (2) is very easy to use.

3. When is pairplot most useful?
Pair plots are most useful when just starting out with a dataset, because they help contextualize relationships within it.
"""

pokemon_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon=pd.read_csv(pokemon_input_file,index_col=0)


# Q1. facet grid with row as Legendary . Compare attack values between legendary and non legendary
# gPokemon=sns.FacetGrid(pokemon,row="Legendary")
# gPokemon.map(sns.kdeplot,"Attack")
# plt.show()

# Q2. Compare attack values group by legendary and generation
gLG = sns.FacetGrid(pokemon,row="Generation",col="Legendary")
gLG.map(sns.kdeplot,"Attack")
plt.show()

# Q3. pairplot on HP, attack, defense
sns.pairplot(pokemon[["HP", "Attack", "Defense"]])
plt.show()

"""
Conclusion
In this notebook we explored FacetGrid and pairplot, two seaborn facilities for faceting your data, and discussed why 
faceting is so useful in a broad range of cases.

This technique is our first dip into multivariate plotting, an idea that we will explore in more depth with two 
other approaches in the next section.
"""

