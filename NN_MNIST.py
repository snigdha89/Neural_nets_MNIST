import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)

# fashion_mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()
print("Training data shape - {}".format(X_train_full.shape))
print("Training data type - {}".format(X_train_full.dtype))

# Max normalization and seperating Validation set
X_valid, X_train = X_train_full[:6000]/255.0 , X_train_full[6000:]/255.0
Y_valid, Y_train = Y_train_full[:6000] , Y_train_full[6000:]
X_test = X_test / 255.0
print("Valid data shape - {}".format(X_valid.shape))

class_names = ["T-shirt/top", "Trouser", "pullover", "Dress", "Coat", 
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]

# Creating the model - 1 
model1 = keras.models.Sequential()
model1.add(keras.layers.Flatten(input_shape=[28,28])) # Converts 28x28 images to 1D array
model1.add(keras.layers.Dense(100, activation="relu"))
model1.add(keras.layers.Dense(10, activation="softmax"))

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model1.summary()

# Compiling the model
model1.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])

#Fitting the Model
history = model1.fit(X_train , Y_train , epochs = 10, validation_data=(X_valid,Y_valid))

# Plot the learing curve 
def learning_curve(history):
  print(pd.DataFrame(history.history))
  pd.DataFrame(history.history).plot(figsize=(8, 5)) 
  plt.grid(True)
  plt.gca().set_ylim(0, 1) # Set the virtical range to [0,1]
  plt.show()

learning_curve(history) # learning Curve of Model 1

#Assessing the model on first 3 images
model1.evaluate(X_test, Y_test)
X_new = X_test[:3]
Y_prob = model1.predict(X_new)
Y_prob.round(2)
Y_pred = np.where(model1.predict(X_new) > 0.5, 1, 0)
np.array(class_names)[Y_pred]
Y_test[:3]

##view of pictures also
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[Y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# Creating the model - 2 
model2 = keras.models.Sequential()
model2.add(keras.layers.Flatten(input_shape=[28,28])) # Converts 28x28 images to 1D array 
model2.add(keras.layers.Dense(100, activation="relu"))
model2.add(keras.layers.Dense(100, activation="relu"))
model2.add(keras.layers.Dense(10, activation="softmax"))

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model2.summary()

# Compiling the model
model2.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])

#Fitting the Model
history = model2.fit(X_train , Y_train , epochs = 10, validation_data=(X_valid,Y_valid))

learning_curve(history) # learning Curve of Model 2

#Assessing the model on first 3 images
model2.evaluate(X_test, Y_test)
X_new = X_test[:3]
Y_prob = model2.predict(X_new)
Y_prob.round(2)
Y_pred = np.where(model2.predict(X_new) > 0.5, 1, 0)
np.array(class_names)[Y_pred]
Y_test[:3]

##let us look at the pictures also
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[Y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# Creating the model - 3
model3 = keras.models.Sequential()
model3.add(keras.layers.Flatten(input_shape=[28,28])) # Converts 28x28 images to 1D array (have to). 
model3.add(keras.layers.Dense(100, activation="relu"))
model3.add(keras.layers.Dense(100, activation="relu"))
model3.add(keras.layers.Dense(100, activation="relu"))
model3.add(keras.layers.Dense(10, activation="softmax"))

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model3.summary()

# Compiling the model
model3.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

#Fitting the Model
history = model3.fit(X_train , Y_train , epochs = 10, validation_data=(X_valid,Y_valid))

learning_curve(history) # learning Curve of Model 3

#Assessing the model on first 3 images
model3.evaluate(X_test, Y_test)
X_new = X_test[:3]
Y_prob = model3.predict(X_new)
Y_prob.round(2)
Y_pred = np.where(model3.predict(X_new) > 0.5, 1, 0)
np.array(class_names)[Y_pred]
Y_test[:3]

##let us look at the pictures also
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[Y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

"""**Question** **3**"""

# Mnist Digit dataset
digit_mnist = keras.datasets.mnist
(X_train_full, Y_train_full), (X_test, Y_test) = digit_mnist.load_data()
print("Training data shape - {}".format(X_train_full.shape))
print("Training data type - {}".format(X_train_full.dtype))

X_train= np.delete(X_train_full,np.where(Y_train_full>=5), axis=0)
Y_train= np.delete(Y_train_full,np.where(Y_train_full>=5), axis=0)
X_test_1= np.delete(X_test,np.where(Y_test>=5), axis=0)
Y_test_1= np.delete(Y_test,np.where(Y_test>=5), axis=0)

# Max normalization and seperating Validation set
X_valid, X_train = X_train[:3000]/255.0 , X_train[3000:]/255.0
Y_valid, Y_train = Y_train[:3000] , Y_train[3000:]
X_test_1 = X_test_1 / 255.0
print("Valid data shape - {}".format(X_valid.shape))

# Creating the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # Converts 28x28 images to 1D array
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(5, activation="softmax"))

model_copy= keras.models.clone_model(model)

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model.summary()

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

import os
checkpoint_path = "Assignment5/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Do early stopping if after 3 epochs the loss is not decreasing
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,monitor='loss')
# Create a callback that saves the model after each epoch
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_freq="epoch") # By default saves the model at the end of each Epoch

#Fitting the Model with new call back
history = model.fit(X_train , Y_train , epochs = 30, validation_data=(X_valid,Y_valid),callbacks=[checkpoint_cb,early_stopping_cb]) # Pass callback and early stopping to training
#Evaluate the Model
scores = model.evaluate(X_test_1, Y_test_1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

learning_curve(history)

# save the Final model and architecture
model.save("model.h5")
print("Saved the Final model to disk")

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model_copy.summary()
# Compiling the model
model_copy.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Create a callback for the "best model"
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# Do early stopping if after 3 epochs the loss is not decreasing
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,monitor='loss')
#Fitting the Model with new call back
history = model_copy.fit(X_train , Y_train , epochs = 30, validation_data=(X_valid,Y_valid),callbacks=[checkpoint_cb,early_stopping_cb]) # Pass callback to training
model_copy    = keras.models.load_model("my_keras_model.h5") # rollback to best model
#Evaluate the Model
scores = model_copy.evaluate(X_test_1, Y_test_1)
print("%s: %.2f%%" % (model_copy.metrics_names[1], scores[1]*100))

# save the best model and architecture
model_copy.save("my_keras_model.h5")
print("Saved the best model to disk")

"""**Question 4**

**Part a)** ***Build Best Model with epochs 20 and save its weights***
"""

# Mnist Digit dataset
digit_mnist = keras.datasets.mnist
(X_train_full, Y_train_full), (X_test, Y_test) = digit_mnist.load_data()
print("Training data shape - {}".format(X_train_full.shape))
print("Training data type - {}".format(X_train_full.dtype))

X_train= np.delete(X_train_full,np.where(Y_train_full>=5), axis=0)
Y_train= np.delete(Y_train_full,np.where(Y_train_full>=5), axis=0)
X_test_1= np.delete(X_test,np.where(Y_test>=5), axis=0)
Y_test_1= np.delete(Y_test,np.where(Y_test>=5), axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train,test_size=0.2, stratify=Y_train, random_state=42)

# Max normalization and seperating Validation set
X_train = X_train[:]/255.0
Y_train = Y_train[:]
X_test = X_test/ 255.0
print("Valid data shape - {}".format(X_train.shape))
print("Valid data shape - {}".format(X_test.shape))

# Creating the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # Converts 28x28 images to 1D array
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(100, activation="elu"))
model.add(keras.layers.Dense(5, activation="softmax"))

model_4b= keras.models.clone_model(model)

# start from a known place
keras.backend.clear_session()
tf.random.set_seed(42)
model.summary()

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Create a callback for the "best model"
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_modelQ4.h5", save_best_only=True)
# Do early stopping if after 3 epochs the loss is not decreasing
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,monitor='loss')
#Fitting the Model with new call back
import time
start = time.process_time()
history = model.fit(X_train , Y_train , epochs = 20, validation_data=(X_test,Y_test),callbacks=[checkpoint_cb,early_stopping_cb]) # Pass callback to training
print("Process time is ", time.process_time() - start)
model    = keras.models.load_model("my_keras_modelQ4.h5") # rollback to best model
#Evaluate the Model
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

learning_curve(history)

#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# save the best model and architecture
model.save("BestModel1Q4.h5")
print("Saved the best model to disk")

# save the weights of the best model
model.save_weights("my_keras_weightsQ4.ckpt")
print("Saved the best model weights to disk")

"""**Part b)** ***Use the above saved weights to build a new Model.***"""

mnist_numbers = keras.datasets.mnist
(X_train_full_2, Y_train_full_2), (X_test_2, Y_test_2) = mnist_numbers.load_data()

X_train_new_2 = np.delete(X_train_full_2,np.where(Y_train_full_2<5), axis=0)
Y_train_new_2= np.delete(Y_train_full_2,np.where(Y_train_full_2<5), axis=0)
X_test_2= np.delete(X_test_2,np.where(Y_test_2<5), axis=0)
Y_test_2= np.delete(Y_test_2,np.where(Y_test_2<5), axis=0)

X_train_new22 = X_train_new_2.reshape(X_train_new_2.shape[0], -1)
from imblearn.datasets import make_imbalance
X_train_4b2, Y_train_4b = make_imbalance( X_train_new22,Y_train_new_2,sampling_strategy={5: 100, 6: 100, 7: 100, 8: 100, 9: 100},random_state=42, return_index=True)
X_train_4b = X_train_4b2.reshape(-1,28,28)

from sklearn.model_selection import train_test_split
X_train_fin, X_test_fin, Y_train_fin, Y_test_fin = train_test_split(X_train_4b, Y_train_4b,test_size=0.2, stratify=Y_train_4b, random_state=42)

X_train_fin = X_train_fin[:]/255.0
Y_train_fin = Y_train_fin[:]
X_test_fin = X_test_fin / 255.0

print(" X_train_fin shape - {}".format(X_train_fin.shape))
print(" X_test_fin shape - {}".format(X_test_fin.shape))
print(" Y_train_fin shape - {}".format(Y_train_fin.shape))
print(" Y_test_fin shape - {}".format(Y_test_fin.shape))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y_train_fin)
Y_train_fin = le.transform(Y_train_fin)
Y_test_fin = le.transform(Y_test_fin)

model_4b.summary()

# Compiling the model
model_4b.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model_4b.load_weights("my_keras_weightsQ4.ckpt")

start = time.process_time()

#Fitting the Model with new call back
history = model_4b.fit(X_train_fin , Y_train_fin , epochs = 2, validation_data=(X_test_fin,Y_test_fin), callbacks=[checkpoint_cb,early_stopping_cb]) # Pass callback to training
print("Process time for 2 epochs is ", time.process_time() - start)

#Evaluate the Model
scores = model_4b.evaluate(X_test_fin, Y_test_fin)
print("%s: %.2f%%" % (model_4b.metrics_names[1], scores[1]*100))

start = time.process_time()
#Fitting the Model with new call back
history = model_4b.fit(X_train_fin , Y_train_fin , epochs = 20, validation_data=(X_test_fin,Y_test_fin), callbacks=[checkpoint_cb,early_stopping_cb]) # Pass callback to training
print("Process time for 20 epochs is ", time.process_time() - start)

#Evaluate the Model
scores = model_4b.evaluate(X_test_fin, Y_test_fin)
print("%s: %.2f%%" % (model_4b.metrics_names[1], scores[1]*100))

learning_curve(history)

#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()