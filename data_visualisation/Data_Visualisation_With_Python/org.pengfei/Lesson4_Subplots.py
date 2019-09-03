import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
ubplotting is a technique for creating multiple plots that live side-by-side in one overall figure. 
We can use the subplots method to create a figure with multiple subplots. subplots takes two arguments. 
The first one controls the number of rows, the second one the number of columns.
"""
input_file='/home/pliu/data_set/python_data_set/pandas_data_visu/winemag-data_first150k.csv'
reviews=pd.read_csv(input_file,index_col=0)
reviews.head(3)

##### example of subplot with 2 rows and 1 column

#fig, axarr = plt.subplots(2,1,figsize=(12,8))


"""
Let's break this down a bit. When pandas generates a bar chart, behind the scenes here is what it actually does:

1.Generate a new matplotlib Figure object.
2.Create a new matplotlib AxesSubplot object, and assign it to the Figure.
3.Use AxesSubplot methods to draw the information on the screen.
4.Return the result to the user.

In a similar way, our subplots operation above created one overall Figure with two AxesSubplots vertically nested inside of it.

subplots returns two things, a figure (which we assigned to fig) and an array of the axes contained therein (which we assigned to axarr). 
Here are the axarr contents:
"""

# print(axarr)

# it returns a list of axesSubplot object

"""
To tell pandas which subplot we want a new plot to go in-the first one or the second one-we need to grab the proper 
axis out of the list and pass it into pandas via the ax parameter:
"""
# first plot is the vin review point
#reviews['points'].value_counts().sort_index().plot.bar(ax=axarr[0])
# second plot is the vin province
#reviews['province'].value_counts().head(20).plot.bar(ax=axarr[1])
#plt.show()

##### we have to create a 2*2 subplot

my_fig, my_axarr= plt.subplots(2,2,figsize=(12,8))

# fig score
reviews['points'].value_counts().sort_index().plot.line(ax=my_axarr[0][0], fontsize=12, color='mediumvioletred')
my_axarr[0][0].set_title("Wine Scores", fontsize=18)

# fig variety
reviews['variety'].value_counts().head(20).plot.bar(ax=my_axarr[1][0], fontsize=12, color='mediumvioletred')
my_axarr[1][0].set_title("Wine Varieties", fontsize=18)

# fig origin
reviews['province'].value_counts().head(20).plot.bar(ax=my_axarr[1][1], fontsize=12, color='mediumvioletred')
my_axarr[1][1].set_title("Wine Origins", fontsize=18)

# fig prices
reviews['price'].value_counts().plot.hist(ax=my_axarr[0][1], fontsize=12, color='mediumvioletred')
my_axarr[0][1].set_title("Wine Prices", fontsize=18)

plt.subplots_adjust(hspace=.3)

sns.despine()

plt.show()



