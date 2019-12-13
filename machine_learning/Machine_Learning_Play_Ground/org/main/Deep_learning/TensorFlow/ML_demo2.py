import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. load data
# 28*28 image of handwritten digits of 0-9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. understand data
print(x_train[0])
print(y_train[0])

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# 3. normalize data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# check data after normalization
print(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# 4. build the CNN model

# set model types as sequential
model = tf.keras.models.Sequential()

# Build Input layer, the number of neuron must be equal to the number of input 28*28=784
model.add(tf.keras.layers.Flatten())
# Build hidden layer, the activation function here is rectified linear unit y=max(0,x)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Build output layer,
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# 5. compile the model with hyper parameter
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 6. train the model
model.fit(x_train, y_train, epochs=3, batch_size=100)

# 7. validate your model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# 8. Predict on test data
prediction = model.predict([x_test])
print(prediction)


# 9. check the prediction with label
print("The CNN model predict the image is a " + str(np.argmax(prediction[1])))
plt.imshow(x_test[1])
plt.show()
