import pandas as pd

########################################################################
######################### Data selection in data frame ######################
########################################################################


csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
df_csv=pd.read_csv(csvFPath,index_col=0)
#print(df_csv.head())



###### Get columns of the dataframe
"""
In Python we can access the property of an object by accessing it as an attribute. A book object, for example, 
might have a title property, which we can access by calling book.title. Columns in a pandas DataFrame work in 
much the same way.

Hence to access the country property, we can use the following syntax
"""

# print("################Simple way#############")
# print(df_csv.country)
#
# print("################Standard way#############")
# print(df_csv['country'])

"""
These are the two ways of selecting a specific columnar Series out of a pandas DataFrame. Neither of them is more or 
less syntactically valid than the other, but the indexing operator [] does have the advantage that it can handle column
names with reserved characters in them (e.g. if we had a country providence column, reviews.country providence wouldn't
work).



Doesn't a pandas Series look kind of like a fancy dict? It pretty much is, so it's no surprise that, to drill down to 
a single specific value, we need only use the indexing operator [] once more:
"""
# print("################Single value#############")
# print(df_csv['country'][0])

####### Index/position based selection

"""
The standard python array slice syntax a[start_posistion:end_position:steps]

a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array
a[start:end:step] # start through not past end, by step, the default step value is 1.

The start end can be negative, which means counting forwards from the end.
"""
# arr=[10,20,30,40,50]
# # it should return [10, 30]
# print(arr[0:3:2])
# # it should return [30, 40]
# print(arr[-3:-1])


"""
The above indexing operator and attribute selection are nice because they work just like they do in the rest of the 
Python ecosystem. As a novice, this makes them easy to pick up and use. However, pandas has its own accessor operators, loc 
and iloc. For more advanced operations, these are the ones you're supposed to be using.

There are types indexing paradigms. 
- iloc : select data based on its numerical position(index) in the data (So it only takes integers). 
- loc : select data based on its label (The label can be string, int, etc.)

Both loc and iloc are row-first, column-second. This is the opposite of what we do in native python, which is column 
first
"""

####### iloc examples

# get first row
# print(df_csv.iloc[0])

# get first column,
# print(df_csv.iloc[:,0])
"""
The : operator comes from native Python, which means everything. When combined with other selectors, however, it can be 
used to indicate a range of values. For example, to select the country column from just the first, second, and third 
row, we would do:
"""

# get the first column of the first 3 row
# print(df_csv.iloc[:3, 0])

"""
The first argument :3 is short of 0:3, which means get row from position 0 to 3
Following the same logic, we cloud get the first column of the second and third row
"""

# print(df_csv.iloc[1:3,0])

"""
We could also use a list as argument
"""

# print(df_csv.iloc[[0,1,2],0])

"""
iloc can also use negative position as the native python slicing system.
"""

# get the last five rows
# print(df_csv.iloc[-5:])

####################### label based selection

"""
The loc operator: label-based selection. In this paradigm it's the data index value(label), not its position, 
which matters.

The loc operator can also take a boolean/ conditional look up , we will see how to use it in section 
Conditional selection

For example, to get the country of first row in data frame, we would now do the following:
"""

# print(df_csv.loc[0,'country'])

"""
iloc is conceptually simpler than loc because it ignores the dataset's indices. When we use iloc we treat the dataset 
like a big matrix (a list of lists), one that we have to index into by position. loc, by contrast, uses the information 
in the indices to do its work. Since your dataset usually has meaningful indices, it's usually easier to do things using 
loc instead. For example, here's one operation that's much easier using loc
"""

# print(df_csv.loc[:,['taster_name','taster_twitter_handle','points']])


"""
iloc vs loc

iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded.
So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.

This is particularly confusing when the DataFrame index is a simple numerical list, e.g. 0,...,1000. In this case df.
iloc[0:1000] will return 1000 entries, while df.loc[0:1000] return 1001 of them! To get 1000 elements using loc, you 
will need to go one lower and ask for df.iloc[0:999]
"""


########################################################################################
#######################Manipulating the index#########################################
######################################################################################

"""
Label-based selection derives its power from the labels in the index. Critically, the index we use is not immutable. 
We can manipulate the index in any way we see fit.

The set_index method can be used to do the job. 

Performing a set_index is useful if you can come up with an index for the dataset which is better than the current one.

Here is what happens when we set_index to the title field:
"""

# for the set_index takes effect on the data frame ,we need to add inplace=True, or we need assign the
# the result to a variable.

# solution 1
# df_csv.set_index("title",inplace=True,drop=True)
#print(df_csv.head())

# solution 2
# new_df=df_csv.set_index('title')
# print(new_df.head())

# try to use title label with loc to find the rows with following titles.
# print(df_csv.loc[["Quinta dos Avidagos 2011 Avidagos Red (Douro)", "Rainstorm 2013 Pinot Gris (Willamette Valley)"],['country','designation']])

# taster_df=df_csv.set_index('taster_name')

# get taster Roger Voss reviewed vin title and points
# print(taster_df.loc[['Roger Voss'],['title','points']])

###########################################################################################
######################## Conditional selection ###########################################
##########################################################################################

"""
So far we've been indexing various strides of data, using structural properties of the DataFrame itself. To do 
interesting things with the data, however, we often need to ask questions based on conditions.

For example, suppose that we're interested specifically in better-than-average wines produced in Italy.
"""

# print(df_csv.country == 'Italy')

"""
This operation produced a Series of True/False booleans based on the country of each record. This result can then 
be used inside of loc to select the relevant data:
"""
# print(df_csv.loc[df_csv.country == 'Italy'])
# we can combine another loc operator to only show the country column
# print(df_csv.loc[df_csv.country == 'Italy'].loc[:,['country']])

"""
The boolean expression can be combined with &, | operator
"""

# res=df_csv.loc[(df_csv.country == 'Italy') & (df_csv.points >= 90)].loc[:,['country','points']]
# print(res.head())
#
# res=df_csv.loc[(df_csv.country == 'Italy') | (df_csv.points >= 90)].loc[:,['country','points']]
# print(res.head())


"""
pandas comes with a few pre-built conditional selectors, two of which we will highlight here. The first is isin. isin 
is lets you select data whose value "is in" a list of values. 

For example, here's how we can use it to select wines only from Italy or France:
"""

# print(df_csv.loc[df_csv.country.isin(['Italy','France'])].loc[:,['country']].head())


"""
The second is isnull (and its companion notnull). These methods let you highlight values which are or are not empty 
(NaN). 

For example, to filter out wines lacking a price tag in the dataset, here's what we would do:
"""

# print(df_csv.loc[df_csv.price.notnull()])


####################################################################################
###########################Assigning data ##########################################
##################################################################################

"""
Assigning data in data frame can be divided into two categories
- add column (concatenation vertical)
- add rows (concatenation horizontal)
"""
##############  Add a column ##################

# df_csv['pengfei'] = 'Great'
# print(df_csv.pengfei.head())
#
# df_csv['index_backwards'] = range(len(df_csv),0,-1)
# print(df_csv.index_backwards.head())
contry_column=df_csv.country
after_col_append=pd.concat([df_csv,contry_column],axis=1)
print(after_col_append.head(1))
############# Add a row ######################

"""
Pandas only support append to insert a new row at the end of the dataframe
In the following example, I will take the first row and insert it at the end of data frame
"""
last_row=df_csv.iloc[:1,:]
#print(df_csv.head(1))
#print(last_row)
# last_row should be = head(1)

# last row befor append
# print(df_csv.iloc[-1:,:])

# without the ignor_index, the last row will still use the old index which is 0,
# after_append=df_csv.append(last_row,ignore_index=True)
# last row after append
# print(after_append.iloc[-1:,:])

############################################################################
####################### Documentations ####################################
##########################################################################

"""
A great tutorial of iloc, loc https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
"""


