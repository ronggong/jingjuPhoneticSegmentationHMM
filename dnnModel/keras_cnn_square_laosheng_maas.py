# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import cPickle,gzip

# load training and validation data
# filename_train_validation_set = '../trainingData/train_set_all_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
# filename_train_set = '../trainingData/train_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
# filename_validation_set = '../trainingData/validation_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
#
# with gzip.open(filename_train_validation_set, 'rb') as f:
#     X_train_validation, Y_train_validation = cPickle.load(f)
#
# with gzip.open(filename_train_set, 'rb') as f:
#     X_train, Y_train = cPickle.load(f)
#
# with gzip.open(filename_validation_set, 'rb') as f:
#     X_validation, Y_validation = cPickle.load(f)
#
# # X_train = np.transpose(X_train)
# Y_train_validation = to_categorical(Y_train_validation)
# Y_train = to_categorical(Y_train)
# Y_validation = to_categorical(Y_validation)

def squareFilterMaasCNN():

    nlen = 21
    filter_density = 1
    channel_axis = 1
    reshape_dim = (1, 80, nlen)
    input_dim = (80, nlen)

    model_1 = Sequential()
    model_1.add(Reshape(reshape_dim, input_shape=input_dim))
    model_1.add(Convolution2D(48* filter_density, 9, 9, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(Activation("relu"))
    model_1.add(MaxPooling2D(pool_size=(1, 3), border_mode='valid', dim_ordering='th'))

    # print(model_1.output_shape)

    model_1.add(Convolution2D(48 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(Activation("relu"))
    model_1.add(Flatten())
    # print(model_1.output_shape)


    model_1.add(Dense(output_dim=29))
    model_1.add(Activation("softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model_1.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    return model_1

def train_model(file_path_model):
    """
    train final model save to model path
    """
    model_0 = squareFilterMaasCNN()

    print(model_0.count_params())

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]

    hist = model_0.fit(X_train,
              Y_train,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              nb_epoch=500,
              batch_size=128)

    nb_epoch = len(hist.history['acc'])

    model_1 = squareFilterMaasCNN()

    hist = model_1.fit(X_train_validation,
                    Y_train_validation,
                    nb_epoch=nb_epoch,
                    batch_size=128)


    model_1.save(file_path_model)

file_path_model = '/scratch/rgongcnnAcousticModel/out/keras.cnn_maas_laosheng_mfccBands_2D_all_optim.h5'
train_model(file_path_model)

