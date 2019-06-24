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


class Siamese:
    def __init__(self, x_train, y_train, x_test, y_test, input_shape, num_classes):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = input_shape
        self.num_classes = num_classes

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    def create_pairs(self, x, y):
        pairs = []
        labels = []
        for y_id in y:
            positives = np.where(y == y_id)[0]
            negatives = np.where(y != y_id)[0]

            np.random.shuffle(negatives)
            negatives = negatives[:positives.shape[0] * 2]
            for i in range(positives.shape[0]):
                for j in range(i + 1, positives.shape[0]):
                    pairs += [[x[i], x[j]]]
                    labels += [1]

            for i in range(negatives.shape[0]):
                for j in range(i + 1, negatives.shape[0]):
                    pairs += [[x[i], x[j]]]
                    labels += [0]

        shuffled_pairs = []
        shuffled_labels = []
        permuts = permutation(len(labels)).tolist()
        for permut in permuts:
            shuffled_pairs += [pairs[permut]]
            shuffled_labels += [labels[permut]]

        return np.array(shuffled_pairs), np.array(shuffled_labels)

    def create_base_network(self, input_shape):
        '''Base network to be shared (eq. to feature extraction).'''
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def train(self, epochs):
        tr_pairs, tr_y = self.create_pairs(self.x_train, self.y_train)
        te_pairs, te_y = self.create_pairs(self.x_test, self.y_test)
        base_network = self.create_base_network(self.input_shape)

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss, optimizer=rms, metrics=[self.accuracy])
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))