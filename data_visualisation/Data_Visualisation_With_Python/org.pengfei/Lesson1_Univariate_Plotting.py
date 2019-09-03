# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data_first150k.csv'
reviews=pd.read_csv(input_file,index_col=0)

#print(reviews.shape)
#print(len(reviews))
#print(reviews.head(3))

######################################################
#####  Categorical data #############################
#####################################################


#########################################
##### Bar charts ######################
########################################


# show top 10 vin produce province
#top10Province=reviews['province'].value_counts().head(10).plot.bar()
#plt.show(top10Province)

#reviews['country'].value_counts().head(10).plot.bar()
#plt.show()

"""
Top10Province plot says California produces far more wine than any other province of the world! 
We might ask what percent of the total is Californian vintage? This bar chart tells us absolute numbers, 
but it's more useful to know relative proportions. 
"""

# top10ProvincePercent=(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()
# plt.show(top10ProvincePercent)

"""
Bar charts are very flexible: The height can represent anything, as long as it is a number. 
And each bar can represent anything, as long as it is a category.
"""

"""
If the categories are not nominal, like earthquake magnitudes, housing complexes with certain number of apartment
(in our words, numeric) we may want to order the bar name, the following example shows the number of reviews of a 
certain score allotted by Wine Magazine
"""

# vinScoreCounts=reviews['points'].value_counts().sort_index()
# vinScoreCountsBarPlot=vinScoreCounts.plot.bar()

#plt.show(vinScoreCountsBarPlot)


###############################################################
#####################Line charts###############################
##############################################################

"""
The wine review scorecard has 20 different unique values to fill, for which our bar chart is just barely enough. 
What would we do if the magazine rated things 0-100? We'd have 100 different categories; simply too many to fit 
a bar in for each one!

In that case, instead of bar chart, we could use a line chart
In the following example, We can view the vin score with a line charts
"""

# vinScoreCountsLineCharts= vinScoreCounts.plot.line()
# plt.show(vinScoreCountsLineCharts)

"""
A line chart can pass over any number of many individual values, making it the tool of first choice 
for distributions with many unique values or categories.
"""

##########################################################################

"""
Quick break: bar or line
Let's do a quick exercise. Suppose that we're interested in counting the following variables:

1.The number of tubs of ice cream purchased by flavor, given that there are 5 different flavors. (bar)
2.The average number of cars purchased from American car manufacturers in Michigan. (bar)
3.Test scores given to students by teachers at a college, on a 0-100 scale.(line)
4.The number of restaurants located on the street by the name of the street in Lower Manhattan.

For which of these would a bar chart be better? Which ones would be better off with a line?

Number 4 is a lot harder. City streets are obviously ordinary categorical variables, so we *ought* to use a bar chart; 
but there are a lot of streets out there! We couldn't possibly fit all of them into a display.

Sometimes, your data will have too many points to do something "neatly", and that's OK. 
If you organize the data by value count and plot a line chart over that, you'll learn 
valuable information about *percentiles*: that a street in the 90th percentile has 20 restaurants, 
for example, or one in the 50th just 6. This is basically a form of aggregation: we've turned streets 
into percentiles!
"""

#######################################################
####################Area chart#########################
#######################################################

"""
Area charts are just line charts, but with the bottom shaded in. That's it!
"""
#vinScoreCountsAreaCharts= vinScoreCounts.plot.area()
#plt.show(vinScoreCountsAreaCharts)


############################################################################
##############################Interval data################################
###########################################################################

"""
Examples of interval variables are the wind speed in a hurricane, shear strength in concrete, and the temperature 
of the sun. An interval variable goes beyond an ordinal categorical variable: it has a meaningful order, in the sense 
that we can quantify what the difference between two entries is itself an interval variable.

For example, if I say that this sample of water is -20 degrees Celcius, and this other sample is 120 degrees Celcius, 
then I can quantify the difference between them: 140 degrees "worth" of heat, or such-and-such many joules of energy.

The difference can be qualitative sometimes. At a minimum, being able to state something so clearly feels a lot more 
"measured" than, say, saying you'll buy this wine and not that one, because this one scored a 92 on some taste test 
and that one only got an 85. More definitively, any variable that has infinitely many possible values is definitely 
an interval variable (why not 120.1 degrees? 120.001? 120.0000000001? Etc).

Line charts work well for interval data. Bar charts don't—unless your ability to measure it is very limited, 
interval data will naturally vary by quite a lot.

Let's apply a new tool, the histogram, to an interval variable in our dataset, 
price (we'll cut price off at 200$ a bottle; more on why shortly).
"""

#reviews[reviews['price']<200]['price'].plot.hist()
#plt.show()

"""
histograms have one major shortcoming (the reason for our 200$ caveat earlier). Because they break space up 
into even intervals, they don't deal very well with skewed data.
"""

#reviews['price'].plot.hist()
#plt.show()

"""
The real reason I excluded the >$200 bottles earlier; some of these vintages are really expensive! And the chart
will "grow" to include them, to the detriment of the rest of the data being shown.
"""

#print(reviews[reviews['price']>1500])

"""
There are only 3 vin cost more than 1500, 

There are many ways of dealing with the skewed data problem; those are outside the scope of this tutorial. 
The easiest is to just do what I did: cut things off at a sensible level.

This phenomenon is known (statistically) as skew, and it's a fairly common occurance among interval variables.
"""

#reviews['points'].plot.hist()
#plt.show()

# Histograms work best for interval variables without skew. They also work really well for ordinal categorical
# variables like points


############################################################
####################Pokeman test############################
############################################################

pd.set_option('max_columns',None)
pokemon_input_file = '/home/pliu/data_set/python_data_set/pandas_data_visu/Pokemon.csv'
pokemon = pd.read_csv(pokemon_input_file)

#0
#print(pokemon.dtypes)
#1
#print(pokemon.shape)
#2
#print(pokemon.isnull().sum())
#3
#print(pokemon.describe)
#4
#print(pokemon.head(3))

##### Q1. The frequency of pokemom type

# sort index using alpha
#pokemon_type_count=pokemon['Type 1'].value_counts().sort_index()
# without sorting
#pokemon_type_count=pokemon['Type 1'].value_counts()
#pokemon_type_count.plot.bar()
#plt.show()

######Q2 The frequency of pokemon by hp

#pokemon_hp_count = pokemon['HP'].value_counts().sort_index()
#pokemon_hp_count.plot.line()
#plt.show()

######Q3 the frequency of pokemon by speed

# be ware of skewed data
pokemon_weight_count = pokemon['Speed'].plot.hist()
plt.show()