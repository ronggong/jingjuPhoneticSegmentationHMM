from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle
import gzip


# load training and validation data
filename_train_set = '/scratch/rgongDNNAcousticModel/phonemeSeg/train_set_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'
filename_train_validation_set = '/scratch/rgongDNNAcousticModel/phonemeSeg/train_set_all_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'
filename_validation_set = '/scratch/rgongDNNAcousticModel/phonemeSeg/validation_set_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'

with gzip.open(filename_train_set, 'rb') as f:
    X_train, Y_train = cPickle.load(f)

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_validation_set, 'rb') as f:
    X_validation, Y_validation = cPickle.load(f)

# X_train = np.transpose(X_train)
Y_train = to_categorical(Y_train)
Y_train_validation = to_categorical(Y_train_validation)
Y_validation = to_categorical(Y_validation)

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                      {'layers': 'three',}
                      ]),

            'units1': hp.uniform('units1', 64, 512),

            'dropout1': hp.uniform('dropout1', .25, .75),

            'batch_size' : hp.uniform('batch_size', 28, 128),

            'nb_epochs' :  500,
            'optimizer': hp.choice('optimizer',['adadelta','adam']),
            'activation': 'relu'
        }

def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X_train.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units1'], init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(output_dim=params['units1'], init="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout1']))

    model.add(Dense(29))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics = ['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    model.fit(X_train, Y_train,
              nb_epoch=params['nb_epochs'],
              batch_size=params['batch_size'],
              validation_data = (X_validation, Y_validation),
              callbacks=callbacks,
              verbose = 0)

    score, acc = model.evaluate(X_validation, Y_validation, batch_size = 128, verbose = 0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK}

def f_nn_model(node_size, dropout, i_d):
    model = Sequential()

    from keras.layers import Dense, Activation, Dropout

    model.add(Dense(output_dim=node_size, input_dim=i_d))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(node_size))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(output_dim=29))
    model.add(Activation("softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print 'best: '
    # print best

    model_0 = f_nn_model(511, 0.251, 400)

    print model_0.count_params()

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    hist = model_0.fit(X_train,
                    Y_train,
                    nb_epoch=500,
                    validation_data = (X_validation, Y_validation),
                    callbacks = callbacks,
                    batch_size=58)

    nb_epoch = len(hist.history['acc'])

    model_1 = f_nn_model(511, 0.251, 400)

    hist = model_1.fit(X_train_validation,
                    Y_train_validation,
                    nb_epoch=nb_epoch,
                    batch_size=58)

    # loss_and_metrics = model.evaluate(X_validation, Y_validation, batch_size=128)

    model_1.save('/scratch/rgongDNNAcousticModel/out/keras.dnn_2_optim_danAll_mfccBands_neighbor_all.h5')