import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# In this Tutorial, we will use a CNN to classify the handwritten digits image.
# The dataset we will be using in this tutorial is called the MNIST dataset, and it is a classic in the machine
# learning community. This dataset is made up of images of handwritten digits, 28x28 pixels in size

############################ 1 load data ##########################
mnist=tf.keras.datasets.mnist # 28*28 image of handwritten digits of 0-9
(x_train,y_train), (x_test,y_test)=mnist.load_data()

############################ 2 understand data #########################
print(x_train[0])
print(y_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

################################ 3. normalize data #########################
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

# check data after normalization
print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

############################## 4. build the CNN model ##################################
# There are two types of models:
# 1. Sequential :  It contains a linear stack of layers. You can create a Sequential model by passing a list of layers
#                  to the sequential() function
# 2. Functional : It allows you to create an arbitrary graph of layers, as long as they don't have cycles.
model=tf.keras.models.Sequential()
# Build Input layer, the number of neuron must be equal to the number of input 28*28=784
model.add(tf.keras.layers.Flatten())
# Build hidden layer, the activation function here is rectified linear unit y=max(0,x)
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# Build output layer, the number of the neuron here must equal to the number of possible result. In this model it's
# 10. As the output is not binary, we need to use softmax.
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


############################## 5. compile the model with hyper parameter ############################
# adam is the default optimizer, we can use gradient decent, etc.
# loss is how we measure the error of the model
# metrics is how we evaluate the model
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

############################# 6. train the model ################################################
# One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network one time.
# In the below example, we do it three times, because One epoch leads to underfitting of the curve in the graph
# https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9 nice articl explains why
#
# batch_size, We can’t pass the entire dataset into the neural net at once. So, we divide dataset into Number
# of Batches or sets or parts.
#
# Let’s say we have 2000 training examples that we are going to use .
# We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.

model.fit(x_train,y_train,epochs=3,batch_size=100)


########################### 7. validate your model ################################################

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss, val_acc)


########################## 8. Predict on test data ###############################################

prediction=model.predict([x_test])
print(prediction)
print("The CNN model predict the image is a "+ str(np.argmax(prediction[2])))

plt.imshow(x_test[2])
plt.show()
########################## 8. save the model #########################################################
model_path = 'pengfei_cnn_digit_reader.keras'
model.export(model_path)

############################ 9. load model ####################################################
new_model=tf.keras.models.load_model(model_path)

