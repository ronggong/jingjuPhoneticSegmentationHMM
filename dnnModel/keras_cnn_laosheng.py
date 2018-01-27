from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle, pickle
import gzip


# load training and validation data
filename_train_validation_set = '../trainingData/train_set_all_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
filename_train_set = '../trainingData/train_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
filename_validation_set = '../trainingData/validation_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_train_set, 'rb') as f:
    X_train, Y_train = cPickle.load(f)

with gzip.open(filename_validation_set, 'rb') as f:
    X_validation, Y_validation = cPickle.load(f)

Y_train_validation = to_categorical(Y_train_validation)
Y_train = to_categorical(Y_train)
Y_validation = to_categorical(Y_validation)

space = {
            'filter_density': hp.choice('filter_density', [1, 2, 3, 4]),

            'dropout': hp.uniform('dropout', 0.25, 0.5),

            'pool_n_row': hp.choice('pool_n_row', [2, 4, 6, 'all']),

            'pool_n_col': hp.choice('pool_n_col', [2, 4, 6, 'all'])
        }


from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU
from keras.regularizers import l2


def createModel(model, reshape_dim, input_dim, num_filter, height_filter, width_filter, filter_density, pool_n_row, pool_n_col, dropout):
    model.add(Reshape(reshape_dim, input_shape=input_dim))
    model.add(
        Convolution2D(num_filter * filter_density, height_filter, width_filter, border_mode='valid', input_shape=reshape_dim, dim_ordering='th',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    if pool_n_row == 'all' and pool_n_col == 'all':
        model.add(MaxPooling2D(pool_size=(model.output_shape[2], model.output_shape[3]), border_mode='valid',
                                 dim_ordering='th'))
    elif pool_n_row == 'all' and pool_n_col != 'all':
        model.add(MaxPooling2D(pool_size=(model.output_shape[2], pool_n_col), border_mode='valid',
                                 dim_ordering='th'))
    elif pool_n_row != 'all' and pool_n_col == 'all':
        model.add(MaxPooling2D(pool_size=(pool_n_row, model.output_shape[3]), border_mode='valid',
                                 dim_ordering='th'))
    else:
        model.add(MaxPooling2D(pool_size=(pool_n_row, pool_n_col), border_mode='valid', dim_ordering='th'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    return model


def f_nn_model(filter_density, pool_n_row, pool_n_col, dropout):

    nlen = 21
    reshape_dim = (1, 80, nlen)
    input_dim = (80, nlen)
    
    model_1 = Sequential()
    model_1 = createModel(model_1, reshape_dim, input_dim, 32, 50, 1, filter_density, pool_n_row, pool_n_col, dropout)

    model_2 = Sequential()
    model_2 = createModel(model_2, reshape_dim, input_dim, 16, 50, 5, filter_density, pool_n_row, pool_n_col, dropout)

    model_3 = Sequential()
    model_3 = createModel(model_3, reshape_dim, input_dim, 8, 50, 10, filter_density, pool_n_row, pool_n_col, dropout)


    model_4 = Sequential()
    model_4 = createModel(model_4, reshape_dim, input_dim, 32, 70, 1, filter_density, pool_n_row, pool_n_col, dropout)

    model_5 = Sequential()
    model_5 = createModel(model_5, reshape_dim, input_dim, 16, 70, 5, filter_density, pool_n_row, pool_n_col, dropout)

    model_6 = Sequential()
    model_6 = createModel(model_6, reshape_dim, input_dim, 8, 70, 10, filter_density, pool_n_row, pool_n_col, dropout)


    merged = Merge([model_1, model_2, model_3, model_4, model_5, model_6], mode='concat')

    model_merged = Sequential()
    model_merged.add(merged)
    model_merged.add(Dense(output_dim=29))
    model_merged.add(Activation("softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model_merged.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    return model_merged


def f_nn(params):
    print ('Params testing: ', params)

    model_merged = f_nn_model(params['filter_density'], params['pool_n_row'], params['pool_n_col'], params['dropout'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]

    hist = model_merged.fit([X_train, X_train, X_train, X_train, X_train, X_train],
              Y_train,
              validation_data=([X_validation, X_validation, X_validation, X_validation, X_validation, X_validation], Y_validation),
              callbacks=callbacks,
              nb_epoch=500,
              batch_size=128,
              verbose=0)

    score, acc = model_merged.evaluate([X_validation, X_validation, X_validation, X_validation, X_validation, X_validation], Y_validation, batch_size = 128, verbose = 0)
    print('Test accuracy:', acc, 'nb_epoch:', len(hist.history['acc']))
    return {'loss': -acc, 'status': STATUS_OK}


def train_model(filter_density, pool_n_row, pool_n_col, dropout, nb_epoch, file_path_model):
    """
    train final model save to model path
    """

    model_merged_1 = f_nn_model(filter_density, pool_n_row, pool_n_col, dropout)

    print(model_merged_1.count_params())

    hist = model_merged_1.fit([X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation],
                    Y_train_validation,
                    nb_epoch=nb_epoch,
                    batch_size=128)

    model_merged_1.save(file_path_model)
    

if __name__ == '__main__':

    # parameters search
    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print 'best: '
    # print best

    # train the final model

    file_path_model = '/scratch/rgongcnnAcousticModel/out/keras.cnn_laosheng_mfccBands_2D_all_optim.h5'
    train_model(filter_density=4, pool_n_row=2, pool_n_col='all', dropout=0.32, nb_epoch=54, file_path_model=file_path_model)
