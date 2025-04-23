#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font",size=14)
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import statsmodels.api as sm

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

################################################
###Data injection
###############################################
data = pd.read_csv("/home/pliu/Documents/data_set/sklearn/banking.csv",header=0)
#data.dropna()
#print(data.shape)
#print(list(data.columns))
# see the data frame
#print(data.head())
#print(data['education'].unique())

# Clean the data, change basic.9y,4y,etc to Basic

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

#print(data['education'].unique())


########################################################
# Data exploration, get the label status (how many baught, how many didn't)
#####################################################
#print(data['y'].value_counts())
"""
0    36548
1     4640
Name: y, dtype: int64
"""

#visualize the above result in a figure
#sns.countplot(x='y',data=data,palette='hls')
#plt.show()
#plt.savefig('count_plot')

#visualize the educatino level of all entries
# sns.countplot(x='education',data=data,palette='hls')
# plt.show()

# let's get a sense of the numbers across the two classes

# print(data.groupby('y').mean())

"""
         age    duration  campaign       pdays  previous  emp_var_rate  \
y                                                                        
0  39.911185  220.844807  2.633085  984.113878  0.132374      0.248875   
1  40.913147  553.191164  2.051724  792.035560  0.492672     -1.233448   

   cons_price_idx  cons_conf_idx  euribor3m  nr_employed  
y                                                         
0       93.603757     -40.593097   3.811491  5176.166600  
1       93.354386     -39.789784   2.123135  5095.115991  
"""

"""
Observation

1. The average age of customers who bought the term deposit is higher than that of the customers who didn’t.
2. The pdays (days since the customer was last contacted) is understandably lower for the customers who bought it. The lower the pdays, the better the memory of the last call and hence the better chances of a sale.
3. Surprisingly, campaigns (number of contacts or calls made during the current campaign) are lower for customers who bought the term deposit.
"""


# we can calculate categorical means for other categorical variables such as education and marital status to get a more detailed sense of our data

#print(data.groupby('job').mean())

#print(data.groupby('marital').mean())
#print(data.groupby('education').mean())


#Visualizations
# pd.crosstab(data.job,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.show()

# The frequency of purchase of the deposit depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable.
# table=pd.crosstab(data.marital,data.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Marital Status vs Purchase')
# plt.xlabel('Marital Status')
# plt.ylabel('Proportion of Customers')
# plt.show()
#plt.savefig('mariral_vs_pur_stack')

# The marital status does not seem a strong predictor for the outcome variable.

# table=pd.crosstab(data.education,data.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.show()
#plt.savefig('edu_vs_pur_stack')

# Education seems a good predictor of the outcome variable.

# pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Day of Week')
# plt.xlabel('Day of Week')
# plt.ylabel('Frequency of Purchase')
# plt.show()
#plt.savefig('pur_dayofweek_bar')


# pd.crosstab(data.month,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Month')
# plt.xlabel('Month')
# plt.ylabel('Frequency of Purchase')
# plt.show()
# plt.savefig('pur_fre_month_bar')

# Month might be a good predictor of the outcome variable



# data.age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('hist_age')
# Most of the customers of the bank in this dataset are in the age range of 30–40.

# pd.crosstab(data.poutcome,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Poutcome')
# plt.xlabel('Poutcome')
# plt.ylabel('Frequency of Purchase')
# plt.show()
#plt.savefig('pur_fre_pout_bar')
#Poutcome seems to be a good predictor of the outcome variable.


#plt.savefig('purchase_fre_job')
################################################
##Prepare data for sklearn classifier
#################################################


#To faciliate the classifier job, we need to transform the string enum field
#into boolean type (dummy variables value in 0 or 1)
# for example the education column has Basic, illiterate,etc
# we need to create two new columns education_Basic and education_illiterate
# the value will be 0 or 1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
#print(data_final.columns.values)

# get all column in X
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
#print(X)

####################################################
# Feature selection
#############################################

"""
Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model 
and choose either the best or worst performing feature, setting the feature aside and then 
repeating the process with the rest of the features. This process is applied until all 
features in the dataset are exhausted. The goal of RFE is to select features by recursively 
considering smaller and smaller sets of features.
"""
#logreg = LogisticRegression()
#rfe = RFE(logreg, 18)
#rfe = rfe.fit(data_final[X], data_final[y] )
#print(rfe.support_)
#print(rfe.ranking_)

"""
The RFE has helped us select the following features: 
“previous”, “euribor3m”, “job_blue-collar”, “job_retired”, “job_services”, 
“job_student”, “default_no”, “month_aug”, “month_dec”, “month_jul”, “month_nov”, 
“month_oct”, “month_sep”, “day_of_week_fri”, “day_of_week_wed”, “poutcome_failure”, 
“poutcome_nonexistent”, “poutcome_success”.
"""
#######################################################
# based on the feature selection , build the training data set
#######################################################
cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed",
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
X=data_final[cols]
y=data_final['y']

#######################################################
# Test the feature in the logitistic model to determine if it's significant or not
######################################################
logit_model=sm.Logit(y,X)
result=logit_model.fit()
#print(result.summary())

"""
Optimization terminated successfully.
         Current function value: 0.287116
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                41188
Model:                          Logit   Df Residuals:                    41170
Method:                           MLE   Df Model:                           17
Date:                Thu, 08 Mar 2018   Pseudo R-squ.:                  0.1844
Time:                        16:45:39   Log-Likelihood:                -11826.
converged:                       True   LL-Null:                       -14499.
                                        LLR p-value:                     0.000
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
previous                 0.2385      0.051      4.642      0.000       0.138       0.339
euribor3m               -0.4981      0.012    -40.386      0.000      -0.522      -0.474
job_blue-collar         -0.3222      0.049     -6.549      0.000      -0.419      -0.226
job_retired              0.3821      0.069      5.552      0.000       0.247       0.517
job_services            -0.2423      0.065     -3.701      0.000      -0.371      -0.114
job_student              0.3540      0.086      4.107      0.000       0.185       0.523
default_no               0.3312      0.056      5.943      0.000       0.222       0.440
month_aug                0.4272      0.055      7.770      0.000       0.319       0.535
month_dec                0.8061      0.163      4.948      0.000       0.487       1.125
month_jul                0.7319      0.056     13.094      0.000       0.622       0.841
month_nov                0.2706      0.064      4.249      0.000       0.146       0.395
month_oct                0.8043      0.087      9.258      0.000       0.634       0.975
month_sep                0.5906      0.096      6.160      0.000       0.403       0.778
day_of_week_fri         -0.0044      0.046     -0.097      0.923      -0.094       0.085
day_of_week_wed          0.1226      0.044      2.771      0.006       0.036       0.209
poutcome_failure        -1.8438      0.100    -18.412      0.000      -2.040      -1.647
poutcome_nonexistent    -1.1344      0.070    -16.253      0.000      -1.271      -0.998
poutcome_success         0.0912      0.114      0.803      0.422      -0.131       0.314
========================================================================================

"""

# The p-values for most of the variables are smaller than 0.05,
# therefore, most of them are significant to the model.

####################################################
##Logistic Regression Model Fitting
####################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# Test the trained classifier

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Accuracy is 0.90

#####################################################################
### Cross Validation
###################################################################

"""
Cross validation attempts to avoid overfitting while still producing a prediction for each observation dataset. We are using 10-fold Cross-Validation to train our Logistic Regression model.
"""

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

"""
10-fold cross validation average accuracy: 0.897

The average accuracy remains very close to the Logistic Regression model accuracy; hence, we can conclude that our model generalizes well.
"""

####################################################
### Confusion Matrix
##################################################

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

"""
The result is telling us that we have 10872+254 correct predictions and 1122+109 incorrect predictions.
"""

###################################################
### Compute precision, recall, F-measure and support
################################################

"""
To quote from Scikit Learn:

The precision is the ratio tp / (tp + fp) where tp is the number of true positives and 
fp the number of false positives. The precision is intuitively the ability of the classifier 
to not label a sample as positive if it is negative.


The recall is the ratio tp / (tp + fn) where tp is the number of true positives and 
fn the number of false negatives. The recall is intuitively the ability of the classifier 
to find all the positive samples.

The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
where an F-beta score reaches its best value at 1 and worst score at 0.

The F-beta score weights the recall more than the precision by a factor of beta. 
beta = 1.0 means recall and precision are equally important.

The support is the number of occurrences of each class in y_test.
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Interpretation: Of the entire test set, 88% of the promoted term deposit
# were the term deposit that the customers liked. Of the entire test set,
# 90% of the customer’s preferred term deposits that were promoted.

#################################################
# ROC Curve
#################################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()


"""
The receiver operating characteristic (ROC) curve is another common tool used with binary 
classifiers. The dotted line represents the ROC curve of a purely random classifier; a good classifier 
stays as far away from that line as possible (toward the top-left corner).


"""
#######################################################
##Data set details
##############################################

""" Input variables
age (numeric)
job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
education (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
default: has credit in default? (categorical: “no”, “yes”, “unknown”)
housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
contact: contact communication type (categorical: “cellular”, “telephone”)
month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
previous: number of contacts performed before this campaign and for this client (numeric)
poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
emp.var.rate: employment variation rate — (numeric)
cons.price.idx: consumer price index — (numeric)
cons.conf.idx: consumer confidence index — (numeric)
euribor3m: euribor 3 month rate — (numeric)
nr.employed: number of employees — (numeric)
"""


"""
Predict variable (desired target):

y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)
"""