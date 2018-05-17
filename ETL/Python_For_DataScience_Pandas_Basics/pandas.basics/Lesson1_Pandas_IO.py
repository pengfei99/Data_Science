import pandas as pd
import sqlite3

##############################################################
################Introuduction ###############################
##############################################################

"""
This lesson will show how to use pandas to read file as data frame, create dataframe and
write it to a file
"""



######################################################
######## Creating pandas series and dataframe ########
######################################################

"""
There are two core objects in pandas: the DataFrame and the Series.

A DataFrame is a table. It contains an array of individual entries, each of which has a certain value. Each entry 
corresponds with a row (or record) and a column

For example, consider the following simple DataFrame:
"""

# example1=pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
# print(example1)
#
# example2=pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
# print(example2)

"""
We can also creat dataframe with row indexs (row ids)
"""
# example3=pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
#               'Sue': ['Pretty good.', 'Bland.']},
#              index=['Product A', 'Product B'])
# print(example3)

"""
A Series, by contrast, is a sequence of data values. If a DataFrame is a table, a Series is a list. And in fact you 
can create one with nothing more than a list:
"""

# example4=pd.Series([1, 2, 3, 4, 5])
# print(example4)

"""
A Series is, in essence, a single column of a DataFrame. So you can assign column values to the Series the same 
way as before, using an index parameter. However, a Series do not have a column name, it only has one overall name:
"""

# example5=pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
# print(example5)


###################################################################################################
####################Read csv, excel, .dat file and return a dataframe#############################
#################################################################################################

##### Read csv files

# csvFPath='/home/pliu/Downloads/data_set/pandas_basic/winemag-data-130k-v2.csv'
# df_csv=pd.read_csv(csvFPath,index_col=0)
# print(df_csv.head())


##### Read excel files
"""
An Excel file (XLS or XLST) organizes itself as a sequence of named sheets. Each sheet is basically a table. So to 
load the data into pandas we need one additional parameter: the name of the sheet of interest.

In the following example, the sheet_name is 
"""

# exelFPath='/home/pliu/Downloads/data_set/pandas_basic/WICAgencies2013ytd.xls'
# df_excel=pd.read_excel(exelFPath,sheet_name='Pregnant Women Participating')
# print(df_excel.head())

"""
As you can see in this example, Excel files are often not formatted as well as CSV files are. Spreadsheets allow 
(and encourage) creating notes and fields which are human-readable, but not machine-readable.

So before we can use this particular dataset, we will need to clean it up a bit. We will see how to do so in the 
next Lesson.
"""

###### Read database

"""
Connecting to a SQL database requires a lot more thought than reading from an Excel file. For one, you need to 
create a connector, something that will handle siphoning data from the database.

pandas won't do this for you automatically because there are many, many different types of SQL databases out 
there, each with its own connector. So for a SQLite database (the only kind supported on Kaggle), you would need to 
first do the following (using the sqlite3 library that comes with Python):
"""
# dbPath="/home/pliu/Downloads/data_set/pandas_basic/chinook.db"
# conn = sqlite3.connect(dbPath)

"""
The other thing you need to do is write a SQL statement. Internally, SQL databases all operate very differently. 
Externally, however, they all provide the same API, the "Structured Query Language" (or...SQL...for short).

We (very briefly) need to use SQL to load data into
For the purposes of analysis however we can usually just think of a SQL database as a set of tables with names, 
and SQL as a minor inconvenience in getting that data out of said tables. So, without further ado, here is all the 
SQL you have to know to get the data out of SQLite and into pandas:
"""

# customers= pd.read_sql_query("select * from customers", conn)
# print(customers.head())


#################################################################################
##################### Write dataframe to common files format ####################
################################################################################

example3=pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
              'Sue': ['Pretty good.', 'Bland.']},
              index=['Product A', 'Product B'])
out_dir_path="/home/pliu/Downloads/data_set/pandas_basic"

# write to csv
# csv_out_path=out_dir_path+"/example.csv"
# example3.to_csv(csv_out_path)

# write to excel
# excel_out_path=out_dir_path+"/example.xlsx"
# example3.to_excel(excel_out_path,sheet_name="example")

# write to db
db_out_path=out_dir_path+"/example.db"
db_conn=sqlite3.connect(db_out_path)
"""
option if_exists='append' => if the table exist, append the data frame to the table
       if_exists='replace' 
"""
example3.to_sql("example",db_conn,if_exists='replace')

example4=pd.read_sql_query("select * from example", db_conn)
print(example4.head)



##################################################################################
################### Documentation #######################################
##########################################################################

"""
There are 11 tables in the chinook sample database.

- employees table stores employees data such as employee id, last name, first name, etc. It also has a field named 
  ReportsTo to specify who reports to whom.
- customers table stores customers data.
- invoices & invoice_items tables: these two tables store invoice data. The invoices table stores invoice header data 
  and the invoice_items table stores the invoice line items data.
- artists table stores artists data. It is a simple table that contains only artist id and name.
- albums table stores data about a list of tracks. Each album belongs to one artist. However, one artist may have 
  multiple albums.
- media_types table stores media types such as MPEG audio file, ACC audio file, etc.
- genres table stores music types such as rock, jazz, metal, etc.
- tracks table store the data of songs. Each track belongs to one album.
- playlists & playlist_track tables: playlists table store data about playlists. Each playlist contains a list of 
  tracks. Each track may belong to multiple playlists. The relationship between the playlists table and tracks table 
  is many-to-many. The playlist_track table is used to reflect this relationship.
"""