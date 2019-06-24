from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import cv2
from siamese import Siamese
from classification import Classification
import os
from numpy.random import permutation
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_faces_data(datapath):
    X = []
    y = []
    ids = os.listdir(datapath)
    for id in ids:
        idpath = os.path.join(datapath, id)
        idfaces = os.listdir(idpath)
        for id_train_face in idfaces:
            if id_train_face.split('.')[-1] != 'jpg':
                continue
            img = cv2.imread(os.path.join(idpath, id_train_face))
            img = cv2.resize(img, input_shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img)
            y.append(ids.index(id))

    X = np.asarray(X).astype('float32')
    y = np.asarray(y)
    X /= 255
    return X, y


input_shape = (180, 180)
num_classes = 7
x_train, y_train = load_faces_data('/data_out/siamese_faces/train')
x_test, y_test = load_faces_data('/data_out/siamese_faces/test')


siam = Siamese(x_train, y_train, x_test, y_test, input_shape, num_classes)
siam.train(epochs=5)


classifier = Classification(x_train, y_train, x_test, y_test, input_shape, num_classes)
classifier.train(epochs=5)
