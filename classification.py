from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
import cv2
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dropout, Lambda, Dense, Conv2D, Flatten, Input, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import os
from numpy.random import permutation


class Classification:
    def __init__(self, x_train, y_train, x_test, y_test, input_shape, num_classes):
        self.x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        self.y_train = to_categorical(y_train)
        self.x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        self.y_test = to_categorical(y_test)
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        input = Input(shape=(self.input_shape[0], self.input_shape[1], 1))
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input, x)
        return model

    def train(self, epochs):
        model = self.build_model()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=epochs)