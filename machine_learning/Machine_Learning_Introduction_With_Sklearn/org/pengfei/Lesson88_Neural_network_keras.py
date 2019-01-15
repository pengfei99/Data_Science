import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#####################################################################################################
################################Introduction #####################################################
####################################################################################################
"""
In this lesson we will learn how to use pandas, keras with tensor flow backends to build a neural network

The data set is from a bank which customer is going to leave this bank service. Our model (neural netwrok) should help
us to find out who will leave. 

Dataset is small(for learning purpose) and contains 10000 rows with 14 columns.

Example of the data 

RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1


We will follow the below steps to cover this Lesson
1. Load Data(clean data,transform data).
2. Define Model.
3. Compile Model.
4. Fit Model.
5. Evaluate Model.
6. Tie It All Together.


"""
# fix random seed for reproducibility
np.random.seed(7)

################################################################################################
##############################Load data, clean data, transform data ###################################################
################################################################################################

df_input_file_path='/home/pliu/Downloads/data_set/deep_learning/neural_net_keras/Churn_Modelling.csv'

df=pd.read_csv(df_input_file_path,index_col=0)

# print(df.shape)
# print(df.dtypes)
# print(df.head(1))

"""
In this example, the feature engineering will be only select the usefull colmun.

To do so , analysis the column of the dataset

0-CustomerId           int64
1-Surname             object
2-CreditScore          int64
3-Geography           object
4-Gender              object
5-Age                  int64
6-Tenure               int64
7-Balance            float64
8-NumOfProducts        int64
9-HasCrCard            int64
10-IsActiveMember       int64
11-EstimatedSalary    float64
12-Exited               int64

We could notice the column 0 and 1 is customerId and surname. It will not help us in the model, so the first attempt, we
will chose column 2 to 11 as feature column

The column 12 is clearly our label column
"""

X= df.iloc[:,2:12]
y= df.iloc[:,12]

# print(X.head(1))
# print(y.head(1))

"""
Now, we have column 3 Geography, and 4 Gender are string categorical data, we need to transforme them into numeric data
Go to see lesson 4 for more details. Here we use Label Encoder, which will not create extra column, it will 
automatically encode different labels in that column with values between 0 to n_classes-1.

For example, France->0, Spain->2, Germany->1
"""

print(X['Geography'].unique())
print(X['Gender'].unique())


#X['Geography']=LabelEncoder().fit_transform(X['Geography'])
X['Gender']=LabelEncoder().fit_transform(X['Gender'])

# print(X.head(5))

"""
Now everything is numeric, but we create it new problems, LabelEncoder has replaced France with 0, Germany 1 and Spain 2 
but Germany is not higher than France and France is not smaller than Spain so we need to create a dummy variable for 
Country. So we can't use only label encoder on geography, we will use dummy variable
"""

X=pd.get_dummies(X)

"""
Now we have all feature data in numeric, we need to split them into train and test data set

"""
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

"""
The last problem, If you carefully observe data, you will find that data is not scaled properly. Some variable has 
value in (0, 1000000) and some value is in (0,100). We don't want any of our variable to dominate on other, so we need 
to scale the data
"""
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(X_train)
print()
print(X_test)

"""
Now the data is ready, we can start to build our neural net
"""

###################################################################################################
###################################### Define Model ###############################################
###################################################################################################

"""
We import Sequential and Dense from keras,

We need Sequential module for initializing NN and dense module to add Hidden Layers.

As the client will leave or not leave is a classification problem. We will call our model my_classifier
"""

my_classifier = Sequential()

"""
Now we need to add hidden layer one by one using dense function. List of arguments and definitions

1. output_dim -> It's the number of nodes which you want to add to this layer.

2. init -> It's the initialization of Stochastic Gradient Decent. In NN we need to assign weights to each mode
           which is nothing but importance of that node. At the time of initialization, weights should be close to 0 
           and we will randomly initialize weights using uniform function.
         
3. input_dim -> It's needed only for first layer as model doesn't know the number of our input variables. In our case, 
                the total number of input variables are 11. In the second layer model automatically knows the number of
                input variable form the first hidden layer.

4. Activation Function -> VERY IMPORTANT!!! Neuron applies activation function to weighted sum (summation of Wi * Xi where
                        W is weight, X is input variable and i is suffix of W and X). The closer the activation function
                        value to 1 the more activated is the neuron and more the neuron passes the signal. Which 
                        activation function should be used is critical task. Here we are using rectifier(relu) function
                        in our hidden layer and Sigmoid function in our output layer as we want binary result from 
                        output layer but if the number of categories in output layer is more than 2, then use SoftMax 
                        function.
"""

###### Adding the input layer and the first hidden layer
# kernel_initializer="uniform", activation="relu", input_dim=12, units=6
# my_classifier.add(Dense(output_dim = 6, init='uniform',activation='relu',input_dim=12))
my_classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=12, units=6))


###### Adding the second hidden layer
#my_classifier.add(Dense(output_dim = 6, init='uniform',activation='relu'))
my_classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
###### Adding the output layer
#my_classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
my_classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

###########################################################################################################
#######################################Compile model ######################################################
###########################################################################################################

"""
Till now we have added multiple layers to our classifier. Now let's compile them which can be done using compile method
Arguments added in final compilation will control whole neural network so be careful on this step. 

Arguments list:

Optimizer -> This is nothing but the algo which you wanna use to find optimal set of weights (Note in 
             Dense(..., init='uniform'), we use function 'uniform' to initialize the weights). Here, we define 
             a new function(algorithm) which will optimize weights in turn making the NN more powerful. 
             The algo is stochastic gradient descent(SGD). Among several types of SGD algo, We choose the
             "Adam".
             
loss -> If you go in deeper detail of SGD, you will find that SGD depends on loss. Since our dependent variable (label)
        is binary, we will have to use logarithmic loss function called 'binary_crossentropy'

metrics -> We want to improve performance of our neural network based on accuracy
"""

###### Compilling Neural Network
my_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
After this step, our NN model is built. 
"""


##############################################################################################################
####################################Train/fit model ######################################################
#############################################################################################################

"""

batch_size -> is used to specify the number of observation after which you want to update weight.

Epoch -> is nothing but the total number of iterations. Choosing the value of batch size and epoch is trial 
         and error , there is no specific rule for that.
"""
my_classifier.fit(X_train, y_train, batch_size=10,epochs =100)

##########################################################################################################
#################################### Test the model ########################################################
##########################################################################################################

y_pred = my_classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

print(cm)


"""
With epochs = 100, we have 1533+144/2000 = 0.8385

[[1533   56]
 [ 267  144]]
 
With epoch = 10, we have only 0.7796
"""


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################