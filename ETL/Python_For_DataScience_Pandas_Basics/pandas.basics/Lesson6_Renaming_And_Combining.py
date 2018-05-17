# -*- coding: utf-8 -*-

import pandas as pd

##################################################################
###################Introduction ##################################
##################################################################

"""
In this lesson, we will see how to rename and combine columns in dataframe. You can find more detail in 
"Renaming and combining" section of the Advanced Pandas tutorial.

Renaming is covered in its own section(https://pandas.pydata.org/pandas-docs/stable/basics.html#renaming-mapping-labels)
in the Essential Basic functionality section of the extensive official documentation. Combining is covered by the 
"Merge, join, concatenate" (https://pandas.pydata.org/pandas-docs/stable/merging.html) section
"""

csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
df=pd.read_csv(csvFPath,index_col=0)


#####################################################################################
############################# Rename index/column ###################################
#####################################################################################

#print(df.head())

"""
Oftentimes data will come to us with column names, index names, or other naming conventions that we are not satisfied 
with. In that case, we may use pandas renaming utility functions to change the names of the offending entries to 
something better.

The first function we'll introduce here is rename, which lets you rename index names and/or column names. For example, 
to change the points column in our dataset to score, we would do:
"""
# as always, by default, rename does not change the origin data frame, it returns a new dataframe
# to make it change the origin data frame, add inplace option.
# df.rename(columns={'points':'score'},inplace=True)
# print(df.score)


"""
rename lets you rename index or column values by specifying a index or column keyword parameter, respectively. 
It supports a variety of input formats, but I usually find a Python dict to be the most convenient one. 
Here is an example using it to rename some elements on the index.
"""

# df.rename(index={0:'firstEntry',1:'secondEntry'},inplace=True)
# print(df.loc[['firstEntry','secondEntry'],:])


"""
You'll probably rename columns very often, but rename index values very rarely. For that, set_index is usually more 
convenient.

Both the row index and the column index can have their own name attribute. The complimentary rename_axis method may 
be used to change these names. For example:
"""

# rename_axis can't modify the origin dataframe even with inplace=True option, we have to assign a variable to the
# return value
# new_df=df.rename_axis("wines", axis='rows',inplace=True).rename_axis("fields", axis='columns',inplace=True)
# print(new_df.head(1))


####################################################################################################
############################# Combining ##########################################################
##################################################################################################

"""
When performing operations on a dataset we will sometimes need to combine different DataFrame and/or Series in 
non-trivial ways. pandas has three core methods for doing this. In order of increasing complexity, these are:

- concat 
- join  
- merge 

Most of what merge can do can also be done more simply with join. The simplest combining method is concat. This function 
works just like the list.concat method in core Python: given a list of elements, it will smush those elements together 
along an axis.

This is useful when we have data in different DataFrame or Series objects but having the same fields (columns). 
"""

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                     'D': ['D2', 'D3', 'D6', 'D7'],
                     'F': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])


##############################################################################
######################### Concat #############################################
##############################################################################

frames=[df1,df2,df3]
# result=pd.concat(frames)
# print(result)


"""
Like its sibling function on ndarrays, numpy.concatenate, pandas.concat takes a list or dict of homogeneously-typed 
objects and concatenates them with some configurable handling of “what to do with the other axes”:

pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)
          
- objs : a sequence or mapping of Series, DataFrame, or Panel objects. If a dict is passed, the sorted keys will be used 
         as the keys argument, unless it is passed, in which case the values will be selected (see below). Any None 
         objects will be dropped silently unless they are all None in which case a ValueError will be raised.
         
- axis : {0, 1, ...}, default 0. The axis to concatenate along.

- join : {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for 
         intersection.
         
- ignore_index : boolean, default False. If True, do not use the index values on the concatenation axis. The resulting 
                 axis will be labeled 0, ..., n - 1. This is useful if you are concatenating objects where the 
                 concatenation axis does not have meaningful indexing information. Note the index values on the other 
                 axes are still respected in the join.
                 
- join_axes : list of Index objects. Specific indexes to use for the other n - 1 axes instead of performing inner/outer 
              set logic.
              
- keys : sequence, default None. Construct hierarchical index using the passed keys as the outermost level. If multiple 
         levels passed, should contain tuples.
         
- levels : list of sequences, default None. Specific levels (unique values) to use for constructing a MultiIndex. 
           Otherwise they will be inferred from the keys.
           
- names : list, default None. Names for the levels in the resulting hierarchical index.

- verify_integrity : boolean, default False. Check whether the new concatenated axis contains duplicates. This can be 
                     very expensive relative to the actual data concatenation.
                     
- copy : boolean, default True. If False, do not copy data unnecessarily.
"""

"""
Without a little bit of context and example many of these arguments don’t make much sense. Let’s take the above example. 
Suppose we wanted to associate specific keys with each of the pieces of the chopped up DataFrame. We can do this using 
the keys argument:
"""

# we add a new index into the data frame, df1 will have x as index, etc.
# result = pd.concat(frames, keys=['x', 'y', 'z'])
# print(result)

# get data of df1 by using loc label x
# df1_data=result.loc[['x'],:]
# print(df1_data)

"""
When gluing together multiple DataFrames (or Panels or...), for example, you have a choice of how to handle the other 
axes (other than the one being concatenated). This can be done in three ways:

1. Take the (sorted) union of them all, join='outer'. This is the default option as it results in zero information loss.
2. Take the intersection, join='inner'.
3. Use a specific index (in the case of DataFrame) or indexes (in the case of Panel or future higher dimensional 
   objects), i.e. the join_axes argument
   
Here is a example of each of these methods. First, the default join='outer' behavior:
"""
# result = pd.concat([df1, df4], axis=1)
# print(result)
#
# axis= 1 means join on the row index, the result will be row 2 and 3 (the only intersection possible)
# inner_join = pd.concat([df1,df4], axis=1, join='inner')
# print(inner_join)

# axis=0 (the default value) means join on the column index, the result will be column B and D
# column_inner_join=pd.concat([df1,df4],join='inner')
# print(column_inner_join)


"""
A useful shortcut to concat are the append instance methods on Series and DataFrame. These methods actually predated 
concat. They concatenate along axis=0, namely the index:
"""
# result=df1.append(df2)
# print(result)

"""
If the index and column is not uniform 
- row index have duplicates, append will create a new row for the duplicate index row
- column in df1 not present in df2, append will create a cell for the column with NaN as value.
 
See the following example
"""

# result=df1.append(df4)
# print(result)

"""
For DataFrames which don’t have a meaningful index, you may wish to append them and ignore the fact that they may have 
overlapping indexes:

The following option will ignor the origin index in the origin datafram, it will generate a new index for the result 
data frame. It works for concat and append
"""

# result=pd.concat([df1,df4],ignore_index=True)
# result=df1.append(df4,ignore_index=True)
# print(result)

"""
Append can take multiple objects to concatenate 
Unlike list.append method, which appends to the original list and returns nothing, append here does not modify df1 and 
returns its copy with df2 appended.
"""

# result = df1.append([df2,df3])
# print(result)


###################### Concatenating with mixed ndims ##########################################################

"""
You can concatenate a mix of Series and DataFrames. The Series will be transformed to DataFrames with the column name 
as the name of the Series. if the series are unamed, a number (0..) will be used as column name
"""
# s1= pd.Series(['X0','X1','X2','X3'], name='X')
# result = pd.concat([df1,s1],axis=1)

"""
You can add a higher level label to each element of the concatenation, in our example, we add a dataframe label to all 
columns of the dataframe, and series label to the series column
"""
# result_with_keys=pd.concat([df1,s1],axis=1,keys=['data_frame','series'])
# print(result_with_keys)


######################### Appending series as a rows of a dataframe  ####################################

# create a series with a list of index, the index here will be the column name of the dataframe which we want to append
# s2 = pd.Series(['X0','X1','X2','X3'], index=['A','B','C','D'])
# result= df1.append(s2,ignore_index=True)
# print(result)


######################################################################################################################
################################# Merging ###########################################################################
#################################################################################################################

"""
pandas provides a single function, merge, as the entry point for all standard database join operations between DataFrame 
objects:

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
         
- left: A DataFrame object

- right: Another DataFrame object

- on: Columns (names) to join on. Must be found in both the left and right DataFrame objects. If not passed and 
      left_index and right_index are False, the intersection of the columns in the DataFrames will be inferred 
      to be the join keys

- left_on: Columns from the left DataFrame to use as keys. Can either be column names or arrays with length equal 
           to the length of the DataFrame

- right_on: Columns from the right DataFrame to use as keys. Can either be column names or arrays with length equal 
            to the length of the DataFrame

- left_index: If True, use the index (row labels) from the left DataFrame as its join key(s). In the case of a 
              DataFrame with a MultiIndex (hierarchical), the number of levels must match the number of join keys 
              from the right DataFrame

- right_index: Same usage as left_index for the right DataFrame

- how: One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed description 
       of each method

- sort: Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting to False will 
        improve performance substantially in many cases

- suffixes: A tuple of string suffixes to apply to overlapping columns. Defaults to ('_x', '_y').

- copy: Always copy data (default True) from the passed DataFrame objects, even when reindexing is not necessary. 
        Cannot be avoided in many cases but may improve performance / memory usage. The cases where copying can be 
        avoided are somewhat pathological but this option is provided nonetheless.

- indicator: Add a column to the output DataFrame called _merge with information on the source of each row. 
             _merge is Categorical-type and takes on a value of left_only for observations whose merge key 
             only appears in 'left' DataFrame, right_only for observations whose merge key only appears in 
             'right' DataFrame, and both if the observation’s merge key is found in both.

New in version 0.17.0.

validate : string, default None. If specified, checks if merge is of specified type.

“one_to_one” or “1:1”: checks if merge keys are unique in both left and right datasets.
“one_to_many” or “1:m”: checks if merge keys are unique in left dataset.
“many_to_one” or “m:1”: checks if merge keys are unique in right dataset.
“many_to_many” or “m:m”: allowed, but does not result in checks.

New in version 0.21.0.

The return type will be the same as left. If left is a DataFrame and right is a subclass of DataFrame, the return type 
will still be DataFrame.

merge is a function in the pandas namespace, and it is also available as a DataFrame instance method, with the calling 
DataFrame being implicitly considered the left object in the join.

The related DataFrame.join method, uses merge internally for the index-on-index (by default) and column(s)-on-index join.
If you are joining on index only, you may wish to use DataFrame.join to save yourself some typing.
"""

################################ merge on a single key #########################################
"""
The following example is a one on one join, because the merge keys are unique in both left and right datasets
"""
# left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                          'A': ['A0', 'A1', 'A2', 'A3'],
#                          'B': ['B0', 'B1', 'B2', 'B3']})
#
# right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                           'C': ['C0', 'C1', 'C2', 'C3'],
#                           'D': ['D0', 'D1', 'D2', 'D3']})

# because how='inner' is the default value, so all merge by default is a inner join(intersection of two datasets)
# one_to_one_result = pd.merge(left, right, on='key')
#
# print(one_to_one_result)

"""
We could modify a little right dataset to make the merge is a one to many example, For example, if we changed the k3 key
to K2, so we have multiple K2 key
"""

# multi_right=pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K2'],
#                           'C': ['C0', 'C1', 'C2', 'C3'],
#                           'D': ['D0', 'D1', 'D2', 'D3']})
#
# one_to_many_result = pd.merge(left,multi_right,on='key')
# print(one_to_many_result)

"""
Many to one is the same as one to many, We only need to have duplicate key in left side, and right side is unique key
"""

"""
Many to many means both left and right side has duplicate key values. The result is the cartesian product of all the 
intersection key value column. For example, the merge of the following two dataset will have 4 rows which key value is 
K2.
"""

# multi_left=pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K2'],
#                           'A': ['A0', 'A1', 'A2', 'A3'],
#                           'B': ['B0', 'B1', 'B2', 'B3']})
#
# multi_right=pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K2'],
#                            'C': ['C0', 'C1', 'C2', 'C3'],
#                            'D': ['D0', 'D1', 'D2', 'D3']})
#
# many_to_many_result = pd.merge(multi_left,multi_right,on='key')
# print(many_to_many_result)

################################ Merge on multiple key #######################################

"""
We could merge on multi key, the following example is a one to many inner join on two key: key1,key2
"""

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
#
# inner_result = pd.merge(left, right, on=['key1','key2'])
# print(inner_result)

#################################Different join type, inner, left, right, outer #############

"""
The how argument to merge specifies how to determine which keys are to be included in the resulting table. If a key 
combination does not appear in either the left or right tables, the values in the joined table will be NA. Here is a 
summary of the how options and their SQL equivalent names:

Merge method	SQL Join Name	        Description
left	        LEFT OUTER JOIN	        Use keys from left frame only
right	        RIGHT OUTER JOIN	    Use keys from right frame only
outer	        FULL OUTER JOIN	        Use union of keys from both frames
inner	        INNER JOIN	            Use intersection of keys from both frames
"""

####################### example of left join ############################

"""
Left join only looks keys in the left frame, if there is the same key on the right frame, it will add the value in the 
added column, otherwise it will fill the added column with NaN.

The following example is a one to many left join
"""

# left_result = pd.merge(left,right,how='left',on=['key1','key2'])
# print(left_result)

################### example of right join ##############################

"""
Right join only looks keys in the right frame, if there is the same key on the left frame, it will add the value in the 
added column, otherwise it will fill the added column with NaN

The following example is a many to one right join
"""

# right_result = pd.merge(left,right,how='right',on=['key1','key2'])
# print(right_result)

################### example of outer join ###################################

"""
Outer join will use union of keys from both frames. 

For example, if the left frame keys = [(K0,K0),(K0,K1),(K1,K0),(K2,K1)],right frame keys = [(K0,K0),(K1,K0),(K1,K0),(K2,K0)]
We start from the first key of the left frame, we try to find all keys which has the same value. So, first stpe will be 
l(K0,K0)+r(K0,K0), no more Key on the right frame which has (K0,K0), so we move to next left key (K0,K1), but there is 
no key on right frame, so it will be l(K0,K1)+NaN. Then we move to l(K1,K0), and there are two r(K1,K0), so we will have
2 rows (l(K1,K0)+r(K1,K0)_1, l(K1,K0)+r(K1,K0)_2) instead of one row in the result. Now we move to key l(K2,K1), there is
no key in right frame, so will be l(K2,K1)+NaN. The left frame key is over, there is still one key left in the right frame
we will generate the last row as NaN+r(K2,K0)

Following above algo, result frame key is [(K0,K0),(K0,K1),(K1,K0),(K1,K0),(K2,K1),(K2,K0)]
"""

# outer_result = pd.merge(left,right,how='outer',on=['key1','key2'])
# print(outer_result)


"""
To be more clear, the following example left keys (2,2), right keys(2,2,2).

Following the same algo, it will be l(2)_1+r(2)_1, l(2)_1+r(2)_2, l(2)_1+r(2)_3, l(2)_2+r(2)_1, l(2)_2+r(2)_2, 
l(2)_2+r(2)_3, so the result key will be (2,2,2,2,2,2)
"""

# dup_left = pd.DataFrame({'A' : [1,2], 'B' : [2, 2]})
# dup_right = pd.DataFrame({'A' : [4,5,6], 'B': [2,2,2]})
#
# dup_outer_result=pd.merge(dup_left, dup_right, on='B', how='outer')
# print(dup_outer_result)

"""
Warning 

Joining / merging on duplicate keys can cause a returned frame that is the multiplication of the row dimensions, which 
may result in memory overflow. It is the user’ s responsibility to manage duplicate values in keys before joining large 
DataFrames.
"""

"""
To avoid unexpected duplicate keys, we can use validate option. Key uniqueness is checked before merge operations and 
so should protect against memory overflows. Checking key uniqueness is also a good way to ensure user data structures 
are as expected.

In the following example, there are duplicate values of B in the right DataFrame. As this is not a one-to-one merge – 
as specified in the validate argument – an exception will be raised.
"""

# left = pd.DataFrame({'A' : [1,2], 'B' : [1, 2]})
# right = pd.DataFrame({'A' : [4,5,6], 'B': [2, 2, 2]})
#
# result = pd.merge(left, right, on='B', how='outer', validate="one_to_one")
#
# print(result)


#################################################################################
##################### join ###################################################
##############################################################################

"""
DataFrame.join() is a convenient method for combining the columns of two potentially differently-indexed DataFrames 
into a single result DataFrame. Here is a very basic example:
"""

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                          'B': ['B0', 'B1', 'B2']},
#                          index=['K0', 'K1', 'K2'])
#
# right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
#                           'D': ['D0', 'D2', 'D3']},
#                           index=['K0', 'K2', 'K3'])

# The default behavior is a left join, it only looks the index of left dataframe
# default_result = left.join(right)
# print(default_result)

# to make the joint outer, or inner, we need to specify the option how.
# outer_result = left.join(right,how='outer')
# print(outer_result)
#
# inner_result = left.join(right,how='inner')
# print(inner_result)

################################# joining key columns on an index ######################

"""
join() takes an optional on argument which may be a column or multiple column names, which specifies that the passed 
DataFrame is to be aligned on that column in the DataFrame. These two function calls are completely equivalent:

left.join(right, on=key_or_keys)
pd.merge(left, right, left_on=key_or_keys, right_index=True,
      how='left', sort=False)
"""

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                          'B': ['B0', 'B1', 'B2', 'B3'],
#                        'key': ['K0', 'K1', 'K0', 'K1']})
#
# right = pd.DataFrame({'C': ['C0', 'C1'],
#                           'D': ['D0', 'D1']},
#                           index=['K0', 'K1'])
#
# result = left.join(right,on='key')
# print(result)

"""
To join on multiple keys, the passed DataFrame must have a MultiIndex:
"""

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                          'B': ['B0', 'B1', 'B2', 'B3'],
#                          'key1': ['K0', 'K0', 'K1', 'K2'],
#                          'key2': ['K0', 'K1', 'K0', 'K1']})
#
# index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
#                                    ('K2', 'K0'), ('K2', 'K1')])
#
# right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']},
#                       index=index)
#
# result = left.join(right,on=['key1','key2'])
# print(result)


################## Joining a single index to a Multi-index #######################

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                          'B': ['B0', 'B1', 'B2']},
#                          index=pd.Index(['K0', 'K1', 'K2'], name='key'))
#
# index = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
#                                       ('K2', 'Y2'), ('K2', 'Y3')],
#                                        names=['key', 'Y'])
#
# right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
#                        'D': ['D0', 'D1', 'D2', 'D3']},
#                           index=index)
#
# result = left.join(right, how='inner')
# This below merge is equivalent but less verbose and more memory efficient / faster than the above join.
# result = pd.merge(left.reset_index(), right.reset_index(),
#           on=['key'], how='inner').set_index(['key','Y'])
# print(result)


################### Add suffix to column name to avoid duplicate column names ###############

"""
If the column name in the two dataframe has duplicates, to distinguish them, we can add suffix on the column name of each
data frame.

The following example, left data frame has two column (A,B), the right data frame has also two columns (A,B)
After the join, we want something like this

key  A_left B_left A_right B_right
K0     A0     B0      C0      D0
K1     A1     B1     NaN     NaN
K2     A2     B2      C2      D2
"""

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                         'B': ['B0', 'B1', 'B2']},
                         index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'A': ['C0', 'C2', 'C3'],
                          'B': ['D0', 'D2', 'D3']},
                          index=['K0', 'K2', 'K3'])

result= left.join(right,lsuffix='_left',rsuffix='_right')
print(result)