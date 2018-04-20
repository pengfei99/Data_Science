import pandas as pd
import numpy as np
from utils import getWrongDateFormat
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# read in our data
earthquakes = pd.read_csv("/home/pliu/Downloads/python_data_cleaning/database.csv")
landslides = pd.read_csv("/home/pliu/Downloads/python_data_cleaning/catalog.csv")

np.random.seed(0)

#print(landslides['date'].head())

#print(earthquakes['Date'].head())

# parse the date with the a given date format
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")

#print(landslides['date_parsed'].head())

# parse the date with auto determin format
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

#print(earthquakes['Date_parsed'].head())

# get array of all days in parsed date column
day_of_month_landslides = landslides['date_parsed'].dt.day
# get array of all months in parsed date column
month_of_year_landslides = landslides['date_parsed'].dt.month

day_of_month_landslides = day_of_month_landslides.dropna()


day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day.dropna()

# show the days in a plot
fig, ax=plt.subplots(1,2)
sns.distplot(day_of_month_landslides,kde=False,bins=31, ax=ax[0])
ax[0].set_title("Landslides")
sns.distplot(day_of_month_earthquakes,kde=False,bins=31, ax=ax[1])
ax[1].set_title("Earthquakes")
plt.show()
#print(month_of_year_landslides)


# find out the bad date format in the column
dateList=earthquakes['Date'].tolist()
getWrongDateFormat(dateList)

