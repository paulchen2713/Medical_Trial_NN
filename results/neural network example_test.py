# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:57:38 2020

@author: Paul
"""
# simple example of an Artifitial Neural Network

# library for data initiation
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# library for building the Neural Network model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
# library for training the model
from keras.optimizer import Adam
from keras.metrics import categorical_crossentropy
# normally tensorflow.keras should be the right command, but it doesn't work, don't know why...

# library for platting a confusion matrix
%matplotlib inline
from sklearn.metrics import confusion_matrix
import intertools
import matplotlib.pyplot as plt

# I. Creating dummy data from a made-up story
train_samples = []
train_labels = []

for i in range(50):
    random_younger = randint(13, 64);
    train_samples.append(random_younger);
    trina_labels.append(1);

for i in range(1000):
    
