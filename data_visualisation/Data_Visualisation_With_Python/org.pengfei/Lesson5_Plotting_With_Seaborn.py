import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

vin_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data_first150k.csv'
reviews = pd.read_csv(vin_input_file, index_col=0)
print(reviews.head())
########################################################
########bar chart (pandas) = countplot (seaborn) ######
######################################################

"""
Comparing this chart with the bar chart from two notebooks ago, we find that, unlike pandas, 
seaborn doesn't require us to shape the data for it via value_counts; the countplot (true to its name) aggregates 
the data for us!
"""
# countplot take column points as argument

sns.countplot(reviews['points'])
plt.show()


########### KDE Plot

"""
seaborn doesn't have a direct analogue to the line or area chart. Instead, the package provides a kdeplot:
"""

# sns.kdeplot(reviews.query('price < 200').price)
# plt.show()

"""
KDE, short for "kernel density estimate", is a statistical technique for smoothing out data noise. 
It addresses an important fundamental weakness of a line chart: it will buff out outlier or "in-betweener" 
values which would cause a line chart to suddenly dip.

For example, suppose that there was just one wine priced 19.93$, but several hundred prices 20.00$. 
If we were to plot the value counts in a line chart, our line would dip very suddenly down to 1 and 
then back up to around 1000 again, creating a strangely "jagged" line. The line chart with the same data, 
shown below for the purposes of comparison, has exactly this problem!
"""

#reviews.query('price < 200').price.value_counts().sort_index().plot.line()
#plt.show()

#reviews[reviews['price']<200]['price'].value_counts().sort_index().plot.line()
#plt.show()

"""
A KDE plot is better than a line chart for getting the "true shape" of interval data. In fact, 
I recommend always using it instead of a line chart for such data.

However, it's a worse choice for ordinal categorical data. A KDE plot expects that if there are 200 wine rated 85 
and 400 rated 86, then the values in between, like 85.5, should smooth out to somewhere in between (say, 300). 
However, if the value in between can't occur (wine ratings of 85.5 are not allowed), then the KDE plot is fitting to 
something that doesn't exist. In these cases, use a line chart instead.

"""

# KDE plots can also be used in two dimensions.
# : is assumed to represent all columns

#sns.kdeplot(reviews[reviews['price']<200].loc[:,['price','points']].dropna().sample(5000))
#plt.show()

"""
Bivariate KDE plots like this one are a great alternative to scatter plots and hex plots. 
They solve the same data overplotting issue that scatter plots suffer from and hex plots address, 
in a different but similarly visually appealing. However, note that bivariate KDE plots are very 
computationally intensive. We took a sample of 5000 points in this example to keep compute time reasonable.
"""

#################################################################################
####################Histogram (pandas) = Distplot (seaborn) #####################
################################################################################

# bins become bigger , the bar size become smaller
#sns.distplot(reviews['points'],bins=10,kde=False)
#plt.show()

"""
The distplot is a composite plot type. In the example above we've turned off the kde that's included by default, 
and manually set the number of bins to 10 (two possible ratings per bin), to get a clearer picture.
"""

"""
Notice that this plot comes with some bells and whistles: a correlation coefficient is provided, along with 
histograms on the sides. These kinds of composite plots are a recurring theme in seaborn. Other than that, 
the jointplot is just like the pandas scatter plot.

As in pandas, we can use a hex plot (by simply passing kind='hex') to deal with overplotting:
"""
##################################################################################
##############Scatterplot/hexplot (pandas) = jointplot (seaborn)#################
##################################################################################

#sns.jointplot(x='price',y='points',data=reviews[reviews['price']<100])
#plt.show()


#sns.jointplot(x='price',y='points',data=reviews[reviews['price']<100],kind='hex',gridsize=20)
#plt.show()


##################################################################################
############### Boxplot and violin plot ##########################################
###############################################################################

# seaborn provides a boxplot function. It creates a statistically useful plot that looks like this:

#print(reviews.sample(5))
# get the top 5 five vin variety in vin number counts
#chosen_variety_list=reviews.variety.value_counts().head(5).index
#print(chosen_variety_list)
# filter the data set with the top 5 vin variety
#vin_variety_df=reviews[reviews.variety.isin(chosen_variety_list)]

# get the stats of Bordeaux
#bordeaux=vin_variety_df[vin_variety_df['variety']=="Bordeaux-style Red Blend"].points
#print(bordeaux.describe())

# get the stats of Red Blend
#red_blend=vin_variety_df[vin_variety_df['variety']=="Red Blend"]

# get the stats of Red Blend where its score is greater than 97
#red_blend_with_best_score=red_blend[red_blend['points'] > 97.00]

#print(red_blend.points.describe())
#print(red_blend_with_best_score.points.describe())

# We could notice that, the total num in Red blend is 10062, but total num in Red blend where it's score >97 is only 13
# in the box plot, when the num is so little it will not use the "whisker" (line), it will use "outliers" (circles) to
# represent these data.
#vin_variety_df.groupby(['variety']).max()[['points']].plot.line(figsize=(12, 6))
#plt.show()

########## Box plot for vin variety and points
df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]
fig, ax=plt.subplots(figsize=(12, 6))
#my_boxplot=sns.boxplot(x='variety', y='points', data=df,ax=ax)

#plt.show()

"""
The center of the distributions shown above is the "box" in boxplot. The top of the box is the 75th percentile, 
while the bottom is the 25th percentile. In other words, half of the data is distributed within the box! The green line 
in the middle is the median.

The other part of the plot, the "whiskers", shows the extent of the points beyond the center of the distribution. 
Individual circles beyond that are outliers.

This boxplot shows us that although all five wines recieve broadly similar ratings, Bordeaux-style wines tend to be 
rated a little higher than a Chardonnay.

Boxplots are great for summarizing the shape of many datasets. They also don't have a limit in terms of numeracy: you 
can place as many boxes in the plot as you feel comfortable squeezing onto the page.

However, they only work for interval variables and nominal variables with a large number of possible values; 
they assume your data is roughly normally distributed (otherwise their design doesn't make much sense); and they don't 
carry any information about individual values, only treating the distribution as a whole.

I find the slightly more advanced violinplot to be more visually enticing, in most cases:

"""

################ Violinplot

#sns.violinplot(x='variety', y='points', data=df, ax=ax)
#plt.show()

"""
A violinplot cleverly replaces the box in the boxplot with a kernel density estimate for the data. 
It shows basically the same data, but is harder to misinterpret and much prettier than the utilitarian boxplot.
"""


#####################################################################################################################
###################################  Pokemon Excercise ############################################################
##################################################################################################################

pokemon_input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon=pd.read_csv(pokemon_input_file,index_col=0)

#print(pokemon.shape)
#print(pokemon.dtypes)

# Q1. count plot on pokemon num count group by generation
#sns.countplot(pokemon['Generation'],ax=ax)
#plt.show()

# Q2. kde plot on pokemon num group by HP
# a dist plot includes the kde (why)
# sns.distplot(pokemon['HP'])
# plt.show()

# Q3. jointplot on defence and attack
#
# sns.jointplot(x='Attack', y='Defense', data=pokemon)
# plt.show()

# Q4. hexplot on defence and attack
#sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex',gridsize=20)
#plt.show()

# Q5. kde plot on HP and attack
#sns.kdeplot(pokemon.loc[:,['HP','Attack']])
#plt.show()

# Q6. box plot on attack group by legendary
# sns.boxplot(x='Legendary',y='Attack',data=pokemon,ax=ax)
# plt.show()

# Q7. violinplot on attack group by legendary
sns.violinplot(x='Legendary',y='Attack',data=pokemon,ax=ax)
plt.show()

"""
Conclusion

seaborn is one of the most important, if not the most important, data visualization tool in the Python data 
viz ecosystem. In this notebook we looked at what features and capacities seaborn brings to the table. 
There's plenty more that you can do with the library that we won't cover here or elsewhere in the tutorial; 
I highly recommend browsing the terrific seaborn Gallery page (https://seaborn.pydata.org/examples/index.html) 
to see more beautiful examples of the library in action.
"""