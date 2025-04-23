import pandas as pd
import numpy as np

from scipy import stats

from mlxtend.preprocessing import minmax_scaling

import seaborn as sns

import matplotlib.pyplot as plt

kickstarters_2017=pd.read_csv("/home/pliu/Downloads/python_data_cleaning/ks-projects-201801.csv")

np.random.seed(0)


#######################################################################
###########Scalling data #############################################
#####################################################################

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)
#print(original_data[1],original_data[2])
# mix-max scale the data between 0 and 1, it does not change the data distribution shape
scaled_data = minmax_scaling(original_data, columns = [0])
#print(scaled_data[1],original_data[2])
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")



#######################################################################
###########Normalize data #############################################
#####################################################################

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")

#######################################################################
###########Scalling usd_goal_real of kickstarters #####################
#####################################################################

#print(kickstarters_2017.sample(5))

# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


#######################################################################
###########Scalling goal of kickstarters #####################
#####################################################################
goal = kickstarters_2017.goal
scaled_goal_data = minmax_scaling(goal,columns=[0])
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_goal_data, ax=ax[1])
ax[1].set_title("Scaled data")



#######################################################################
###########Normalize the positive_pledges #############################################
#####################################################################
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")

plt.show()