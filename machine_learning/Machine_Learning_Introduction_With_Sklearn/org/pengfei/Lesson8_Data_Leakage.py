import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


"""
Leaky Predictors

This occurs when your predictors include data that will not be available at the time you make predictions.
For example, imagine you want to predict who will get sick with pneumonia. And data set contains a column 
take antibiotic medicines. People take antibiotic medicines after getting pneumonia in order to recover. 
So the raw data shows a strong relationship between those columns. But took_antibiotic_medicine is frequently 
changed after the value for got_pneumonia is determined. This is target leakage.

Leaky Validation Strategy

A much different type of leak occurs when you aren't careful distinguishing training data from validation data. 
For example, this happens if you run preprocessing (like fitting the Imputer for missing values) before calling 
train_test_split. Validation is meant to be a measure of how the model does on data it hasn't considered before. 
You can corrupt this process in subtle ways if the validation data affects the preprocessing behavoir.. The end result? 
Your model will get very good validation scores, giving you great confidence in it, but perform poorly when you deploy 
it to make decisions.


Preventing Leaky Predictors

There is no single solution that universally prevents leaky predictors. It requires knowledge about your data, 
case-specific inspection and common sense. However, leaky predictors frequently have high statistical correlations 
to the target. So two tactics to keep in mind:

-To screen for possible leaky predictors, look for columns that are statistically correlated to your target.
-If you build a model and find it extremely accurate, you likely have a leakage problem.


Preventing Leaky Validation Strategies

If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, 
including the fitting of preprocessing steps. This is easier if you use scikit-learn Pipelines. When using cross-validation, 
it's even more critical that you use pipelines and do your preprocessing inside the pipeline.
"""



input_file = "/home/pliu/Downloads/data_set/python_ml/AER_credit_card_data.csv"

data = pd.read_csv(input_file, true_values = ['yes'],
                   false_values = ['no'])

#0. get all column types
#print(data.dtypes)

#1. get data frame size
#print(data.shape)

#2. get the empty value cell
#print(data.isnull().sum())

#3. show a data sample
#print(data.sample(1))

#4. basic stats of numeric value
#print(data.describe())

#5. Understand data, choose feature data (predictor)
"""
card is the target/label, True -> means credit card application accepted

card  reports       age  income     share  expenditure  owner selfemp  \
True        0  30.58333     1.8  0.019899     29.51583  False   False   

dependents  months  majorcards  active  
 0       6           1      10 
 
card: Dummy variable, 1 if application for credit card accepted, 0 if not
reports: Number of major derogatory reports
age: Age n years plus twelfths of a year
income: Yearly income (divided by 10,000)
share: Ratio of monthly credit card expenditure to yearly income
expenditure: Average monthly credit card expenditure
owner: 1 if owns their home, 0 if rent
selfempl: 1 if self employed, 0 if not.
dependents: 1 + number of dependents
months: Months living at current address
majorcards: Number of major credit cards held
active: Number of active credit accounts
"""
#######################################################
###############train model without feature selection###
######################################################

y=data['card']
#print(y.head)
X=data.drop(['card'], axis=1)
#print(X.head)

my_pipeline=make_pipeline(RandomForestClassifier())
cv=cross_val_score(my_pipeline,X,y,scoring='accuracy')

print("Cross-val accuracy: "+str(cv))
print("Mean of Cross-val accuracy: %f" %cv.mean())

# Mean of Cross-val accuracy: 0.979526, our model is great, bur really?

# We may have a leaky predictors situation
"""
Expenditure and Share look suspicious. For example, does expenditure mean expenditure on this card or on cards used 
before appying?
"""

# we calculate the expenditure of card holder and non card holder
expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]
#print(expenditures_cardholders.head)
#print(expenditures_noncardholders.head)
print('Fraction of those who received a card with no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who received a card with no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))


# cardHolders_expenditures=data.loc[data['card']==True].expenditure
# nonCardHolders_expenditures=data.loc[data['card']==False].expenditure
# print(cardHolders_expenditures.head)
# print(nonCardHolders_expenditures.head)
# print('Fraction of those who received a card with no expenditures: %.2f' \
#       %(( cardHolders_expenditures == 0).mean()))
# print('Fraction of those who received a card with no expenditures: %.2f' \
#       %((nonCardHolders_expenditures == 0).mean()))

"""
Active, Majorcards are a little less clear, but from the description, they sound concerning. In most situations, 
it's better to be safe than sorry if you can't track down the people who created the data to find out more.
"""
cardHolders_ac=data.loc[data['card']==True].active
nonCardHolders_ac=data.loc[data['card']==False].active
#print(cardHolders_ac.head)
#print(nonCardHolders_ac.head)
print('Fraction of those who received a card with no active account: %.2f' \
       %(( cardHolders_ac == 0).mean()))
print('Fraction of those who received a card with no active account: %.2f' \
       %((nonCardHolders_ac == 0).mean()))

"""
Fraction of those who received a card with no active account: 0.14
Fraction of those who received a card with no active account: 0.26

so active is not really corrolated to label
"""
cardHolders_mc=data.loc[data['card']==True].majorcards
nonCardHolders_mc=data.loc[data['card']==False].majorcards
#print(cardHolders_ac.head)
#print(nonCardHolders_ac.head)
print('Fraction of those who received a card with 1 major card: %.2f' \
       %(( cardHolders_mc == 1).mean()))
print('Fraction of those who received a card with 1 major card: %.2f' \
       %((nonCardHolders_mc == 1).mean()))

"""
Fraction of those who received a card with 1 major card: 0.16
Fraction of those who received a card with 1 major card: 0.26

so majorcard is not really corrolated to label
"""

##########################################################################
##############Train model by removing leaky predicators#################
##########################################################################

leaky_predictors=['expenditure','share']

X2=X.drop(leaky_predictors,axis=1)

lp_cv_score=cross_val_score(my_pipeline,X2,y,scoring='accuracy')

print("Cross-val accuracy after removing leaky predictor: %f" %lp_cv_score.mean())

