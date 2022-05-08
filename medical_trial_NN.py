# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:57:38 2020

@author: Paul
@file: medical_trial_ANN.py
"""
# simple example of an Artifitial Neural Network

# libraries for data initialization
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# libraries for building the neural network model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
# libraries for training the model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# libraries for plotting a confusion matrix
# %matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# library for saving the entire model, or saving weights only
import os.path
# library for loading the entire model
from keras.models import load_model

# library for model reconstruction from JSON
from keras.models import model_from_json
# library for model reconstruction from YAML
from keras.models import model_from_yaml


# I. Creating dummy data from a made-up story
# create two empty lists for training data
# one will hold the input data(samples), the other will hold the target data(labels)
train_samples = []
train_labels = []

# An experimental Covid-19 vaccine was tested on patients ranging from age 13~103 in a clinical 
# trial. The trial had 21000 participants. Half of the participants were under 65 years old, and 
# the other half was 65 years of age or older. The trial showed that around 95% of patients 65 
# or older experienced side effects from the drug, and around 95% of patients under 65 experienced no side effects.
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,103)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,103)
    train_samples.append(random_older)
    train_labels.append(1)
"""
# see what the train_samples looks like
# random numbers are ages ranging anywhere from 13 to 100 years old
for i in train_samples:
    print(i)

# see what the train_labels look like
# 0 indicates that an individual did not experience a side effect
# 1 indicates that an individual did experience a side effect
for i in train_labels:
    print(i)
"""

# convert both lists into numpy arrays(due to the fit() function expects)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
# shuffle the arrays to remove any order that was imposed on the data during the creation process
train_labels, train_samples = shuffle(train_labels, train_samples)

# scale the data down to a range from 0 to 1
# by using scikit-learn’s MinMaxScaler class to scale all of the data down 
# from a scale ranging from 13 to 103 to be on a scale from 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
# reshaping the data as a technical requirement
# since the fit_transform() function doesn’t accept 1D data by default
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

"""
# print out the scaled data
for i in scaled_train_samples:
    print(i)
"""


# II. Creating an Artifitial Neural Network
""" 
(noticed: the following codes activate only when using GPU to run the model, 
 but I'm not using GPU support I think. )

# check to be sure that TensorFlow is able to identify the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# set_memory_growth() allocate only as much GPU memory as needed at a given time, 
# and continues to allocate more when needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
# tf.keras.Sequential model is a linear stack of layers
# it accepts a list, and each element in the list should be a layer

# 1) units specify the number of neurons (or units) the layer has
# 2) input_shape specify the shape of the input data (specified in 
# the first hidden layer(and only this layer)in the model)
# 3) activation specify the activation function to use after this layer

# the third hidden layer use softmax activation function, 
# which give us a probability distribution among the possible outputs
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# call summary() to get a quick visualization
model.summary()

# III. Training
# first, compile the model by using compile() 
# the compile() function configures the model for training and expects a number of parameters
# 1) optimizer = Adam, accepts an optional parameter lr(learning_rate)
# 2) loss = sparse_categorical_crossentropy, given that our labels are in integer format
# 3) metrics = accuracy, a list of metrics evaluated by the model during training and testing
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# then, train the model by using fit()
# 1) x = training set, y = label set
# 2) batch size is the number of samples that are passed to the network at once
# 3) epoch is a single pass of all the data to the network
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, 
          batch_size=10, epochs=30, shuffle=True, verbose=2)


# IV. Prediction
# inference process - the model make predictions for the test_samples based on 
# what it's learned from the train_samples, this process called "inference."

# creting the test set as follow:
# create two empty lists for testing data
# one will hold the input data(samples), the other will hold the target data(labels)
test_samples = []
# normally we would not know the test_labels in the real-world data
# but if we do have the labels, we can draw the confusion matrix later
test_labels =  []

for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,103)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,103)
    test_samples.append(random_older)
    test_labels.append(1)

# convert both lists into numpy arrays(due to the fit() function expects)
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
# shuffle the arrays to remove any order that was imposed on the data during the creation process
test_labels, test_samples = shuffle(test_labels, test_samples)
# reshaping the data as a technical requirement
# since the fit_transform() function doesn’t accept 1D data by default
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

# model.predict() get predictions from the model for the test set
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

"""
# see what the model's predictions look like
# first column: 0, second column: 1
for i in predictions:
    print(i)
"""
# look only at the most probable prediction
rounded_predictions = np.argmax(predictions, axis=-1)
"""
for i in rounded_predictions:
    print(i)
"""


# V. Confusion Matrix
# create the confusion matrix by calling plot_confusion_matrix() function from 
# scikit-learn and assign it to the variable cm(confusion matrix)
# pass the true labels(test_labels) and prediction labels(round_predictions) to cm(confusion matrix)
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# the function plot_confusion_matrix() came directly from scikit-learn’s website
# this is code that they provide in order to plot the confusion matrix.
def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title='Confusion matrix', # default title
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# plot_confusion_matrix() ended

# define the labels for the confusion matrix
# In this case, the labels are titled “no side effects” and “had side effects.”
cm_plot_labels = ['no_side_effects','had_side_effects']

# plot the confusion matrix by using the plot_confusion_matrix() function
# pass in the confusion matrix cm and the labels cm_plot_labels, and a title Confusion Matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# VI. Save and Load a model
# three different ways to save a keras model are demonstrate below.

# 1) saving and loading the model in its entirety - save a model at its 
# current state after it was trained so that we could make use of it later.

# the model.save() function saves:
# 1. the architecture of the model, allowing to re-create the model
# 2. the weights of the model
# 3. the training configuration(loss, optimizer)
# 4. the state of the optimizer, allowing to resume training exactly where we left off

# checks first to see if the file exists already? if not, save the model
if os.path.isfile('D:\computer science lab\ANN_medical trial\model_save_1.h5') is False:
    model.save('D:\computer science lab\ANN_medical trial\model_save_1.h5')

# the load_model() function loads the model by pointing to the saved model's route on disk
model_load_1 = load_model('D:\computer science lab\ANN_medical trial\model_save_1.h5')

"""
# verify that whether the loaded model has the same architecture and weights 
# as the saved model by calling summary() and get_weights()
model_load_1.summary()
print(model_load_1.get_weights())

# we can also inspect attributes about the model, like the optimizer and loss
print(model_load_1.optimizer)  # <keras.optimizers.Adam object at 0x000001FC52FE8978>
print(model_load_1.loss)       # sparse_categorical_crossentropy
"""

# 2) saving ang loading only the architecture of the model - this will not save the weights, 
# configurations, optimizer, loss or anything else, only saves the architecture

# we can save the architecture through JSON string or YAML string:
# save as JSON
model_json_string_2 = model.to_json()
"""
# verify that whether we loaded the architecture correctly
print(model_json_string_2)
"""
# model reconstruction from JSON
model_load_json_2 = model_from_json(model_json_string_2)
"""
# printing the summary to verify that the new model has the same architecture of the original model
# noticed that we only have the architecture in place, so we need to re-train, re-compile later
model_load_json_2.summary()

# =============================================================================
# YAMLLoadWarning: 
#   calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe.
#   Please read https://msg.pyyaml.org/load for full details.
# =============================================================================

# save as YAML
model_yaml_string_2 = model.to_yaml()
print(model_yaml_string_2)
# model reconstruction from YAML
model_load_yaml_2 = model_from_yaml(model_yaml_string_2)
# printing the summary to verify that the new model has the same architecturer of the oiginal model
# noticed that we only have the architecture in place, so we need to re-train, re-compile later
model_load_yaml_2.summary()
"""

# 3) saving and loading only the weights of the model 
# call model.save_weights() and pass in the path and file name to save the weights

# checks first to see if the file exists already? if not, save the weights
if os.path.isfile('D:\computer science lab\ANN_medical trial\model_weights_3.h5') is False:
    model.save_weights('D:\computer science lab\ANN_medical trial\model_weights_3.h5')

# we could then load the saved weights in to a new model, but the new model will need to 
# have the same architecture as the original model before the weights can be saved
# for verification, create model_3 that has the same architecture of the oiginal model
model_3 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# load the previously saved weights to model_3
model_3.load_weights('D:\computer science lab\ANN_medical trial\model_weights_3.h5')
"""
# verify that whether we loaded the weights correctly
print(model_3.get_weights())
"""
