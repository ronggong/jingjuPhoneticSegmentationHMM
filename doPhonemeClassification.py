#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import numpy as np
import pickle
import essentia.standard as ess
from sklearn import preprocessing

from src.parameters import *
from src.phonemeMap import *
from src.textgridParser import syllableTextgridExtraction
from src.trainTestSeparation import getRecordingNames
from phonemeSampleCollection import getFeature,getMFCCBands2D,featureReshape
from src.phonemeClassification import PhonemeClassification
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


def doClassification():
    """
    1. collect features from test set
    2. predict by GMM or DNN models
    3. save the prediction
    :return: prediction of GMM and DNN model
    """

    phone_class = PhonemeClassification()
    phone_class.create_gmm(gmmModel_path)

    mfcc_all = np.array([])
    mfcc_std_all = np.array([])
    mfccBands_all = np.array([])

    y_true = []

    for recording in getRecordingNames('TEST', dataset):
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        # plotAudio(audio,15,16)

        print 'calculating mfcc and mfcc bands ... ', recording
        mfcc                = getFeature(audio, d=False, nbf=True)
        mfccBands           = getMFCCBands2D(audio, nbf=True)
        mfccBands           = np.log(10000*mfccBands+1)
        # mfccBands           = np.log(mfccBands)
        mfcc_std                = preprocessing.StandardScaler().fit_transform(mfcc)

        scaler                  = pickle.load(open(scaler_path, 'rb'))
        mfccBands_std           = scaler.transform(mfccBands)
        # mfccBands_std           = preprocessing.StandardScaler().fit_transform(mfccBands)


        for ii,pho in enumerate(nestedPhonemeLists):

            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))

            # MFCC feature
            sf = round(pho[0][0]*fs/hopsize)
            ef = round(pho[0][1]*fs/hopsize)

            # mfcc syllable
            mfcc_s      = mfcc[sf:ef,:]
            mfcc_s_std  = mfcc_std[sf:ef,:]
            mfccBands_s     = mfccBands[sf:ef,:]
            mfccBands_s_std = mfccBands_std[sf:ef,:]

            fig = plt.figure()
            y = np.arange(0,80)
            x = np.arange(0,21)
            cax = plt.pcolormesh(x,y, np.transpose(mfccBands_s[:21,80*11:80*12]))
            cbar = fig.colorbar(cax)
            plt.xlabel('Frames',fontsize=16)
            plt.ylabel('Mel bands',fontsize=16)
            # plt.title(pho[0][2])
            plt.axis('tight')
            plt.show()
            print(pho[0][2])

            if len(mfcc_all):
                mfcc_all        = np.vstack((mfcc_all,mfcc_s))
                mfcc_std_all    = np.vstack((mfcc_std_all,mfcc_s_std))
                mfccBands_all   = np.vstack((mfccBands_all,mfccBands_s_std))
            else:
                mfcc_all        = mfcc_s
                mfcc_std_all    = mfcc_s_std
                mfccBands_all   = mfccBands_s_std

            # print mfcc_all.shape, mfccBands_all.shape

            ##-- y_true
            y_true_s = []
            for ii_p, p in enumerate(pho[1]):
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]
                index_key = dic_pho_label[key]
                y_true_s += [index_key]*int(round((p[1]-p[0])/hopsize_t))

            print len(y_true_s), mfcc_s.shape[0]

            if len(y_true_s) > mfcc_s.shape[0]:
                y_true_s = y_true_s[:mfcc_s.shape[0]]
            elif len(y_true_s) < mfcc_s.shape[0]:
                y_true_s += [y_true_s[-1]]*(mfcc_s.shape[0]-len(y_true_s))

            y_true += y_true_s

    # phone_class.mapb_gmm(mfcc_all)
    # phone_class.mapb_dnn(mfccBands_all)
    mfccBands_all = featureReshape(mfccBands_all)
    phone_class.mapb_keras(mfccBands_all)
    # phone_class.mapb_keras(mfcc_std_all)
    # phone_class.mapb_xgb(mfccBands_all)
    #
    # obs_gmm = phone_class.mapb_gmm_getter()
    # obs_dnn = phone_class.mapb_dnn_getter()
    obs_keras = phone_class.mapb_keras_getter()
    # obs_xgb = phone_class.mapb_xgb_getter()

    ##-- prediction
    # y_pred_gmm = phone_class.prediction(obs_gmm)
    # y_pred_dnn = phone_class.prediction(obs_dnn)
    y_pred_keras = phone_class.prediction(obs_keras)
    # y_pred_xgb = phone_class.prediction(obs_xgb)
    #
    # np.save('./trainingData/y_pred_gmm_'+dataset+'.npy',y_pred_gmm)
    # np.save('./trainingData/y_pred_dnn_2_512D05M5MB80DO5_plus_validation_300.npy',y_pred_dnn)
    np.save('./trainingData/y_pred_keras_'+dataset+'.npy',y_pred_keras)

    # np.save('y_pred_xgb.npy',y_pred_xgb)
    np.save('./trainingData/y_true_'+dataset+'.npy',y_true)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # print(cm)

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, cm[i, j],
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion_matrix():
    y_pred_gmm = np.load('./trainingData/y_pred_gmm_'+dataset+'.npy')
    # y_pred_dnn = np.load('./trainingData/y_pred_dnn_2_512D05M5MB80DO5_plus_validation_300.npy')
    y_pred_keras = np.load('./trainingData/y_pred_keras_'+dataset+'.npy')
    #y_pred_xgb = np.load('./trainingData/y_pred_xgb.npy')
    y_true = np.load('./trainingData/y_true_'+dataset+'.npy')

    label = []
    for ii in xrange(len(dic_pho_label_inv)):
        label.append(dic_pho_label_inv[ii])

    cm_gmm = confusion_matrix(y_true,y_pred_gmm)
    # cm_dnn = confusion_matrix(y_true,y_pred_dnn)
    cm_keras = confusion_matrix(y_true,y_pred_keras)
    #cm_xgb = confusion_matrix(y_true,y_pred_xgb)

    plt.figure()
    plot_confusion_matrix(cm_gmm, label,
                          normalize=True,
                          title='GMM model confusion matrix. accuracy: '
                                + str(accuracy_score(y_true, y_pred_gmm)))
    plt.show()

    plt.figure()
    plot_confusion_matrix(cm_keras, label,
                          normalize=True,
                          title='Keras DNN model confusion matrix, accuracy: '
                                + str(accuracy_score(y_true, y_pred_keras)))
    plt.show()


if __name__ == '__main__':
    ####---- correct transition phonemes
    doClassification()
    draw_confusion_matrix()