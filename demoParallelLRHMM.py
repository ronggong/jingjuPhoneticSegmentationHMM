#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
# import json
# import operator
# import csv
import numpy as np
import essentia.standard as ess
from sklearn import preprocessing

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from src import metrics
from src.parameters import *
from src.phonemeMap import *
from src.textgridParser import syllableTextgridExtraction
from src.patternMethod import processPatternMethod
# from src.figurePlot import plotAudio
from src.trainTestSeparation import validTransStat
from src.recordingNumber import recordings_test
from phonemeSampleCollection import getFeature,getMBE,getMFCCBands1D
from HSMM.ParallelLRHSMM import ParallelLRHSMM
from ParallelLRHMM import ParallelLRHMM
from LRHMM import _LRHMM
from transProbaParallelLRHelper import processTransPriorIntoParallelLR

import matplotlib.pyplot as plt


# recordings      = getRecordings(textgrid_path)
# number_test     = getRecordingNumber('TEST')
# recordings_test  = [recordings[i] for i in number_test]

def doDemo(pho_duration_threshold=0.00,
           tol=0.04,
           bool_transVaryProb=True,
           enterPenalize=0.0,
           patternMethod=False,
           model_classification='svm',
           valid_string='valid',
           explicit_dur=False,
           plot=False):

    processTransPriorIntoParallelLR(bool_transitionVaryingProbLoop=bool_transVaryProb,ep=enterPenalize)

    numDetectedBoundariesAll, numGroundtruthBoundariesAll, numCorrectAll = 0,0,0
    correctTransPhonemeAll, detectedTransPhonemeAll  = [], []

    if am == 'keras':
        km = _LRHMM.kerasModel(kerasModels_path)


    for recording in recordings_test:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        # plotAudio(audio,15,16)

        print 'calculating mfcc ... ', recording
        if am == 'gmm':
            mfcc                = getFeature(audio)
        elif am == 'dnn':
            mfcc                = getMFCCBands1D(audio)
            mfcc                = preprocessing.StandardScaler().fit_transform(mfcc)
        elif am == 'keras':
            mfcc = getFeature(audio)
            mfcc = preprocessing.StandardScaler().fit_transform(mfcc)
        else:
            print(am+' is not exist.')
            raise

        if patternMethod:
            print 'calculating MBE ...', recording
            MBE = getMBE(audio)

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))

            transcription = []
            # print pho[0]
            starting_time_syllable = pho[0][0]
            frames_jump   = []
            trans_phoneme_gt = []

            for ii_p,p in enumerate(pho[1]):
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]
                # phoneme transcription
                transcription.append(key)

                # collect jump frames
                if len(pho[1])>1 and ii_p<len(pho[1])-1:
                    frames_jump.append(round((p[1]-starting_time_syllable)*fs/float(hopsize)))
                    trans_phoneme_gt.append((ii_p,pho[1][ii_p][2],pho[1][ii_p+1][2]))

            print 'pinyin:',pho[0][2]
            print 'ground truth transcription:',transcription

            # hmm instance
            if explicit_dur:
                hmm = ParallelLRHSMM(pinyin=pho[0][2])
            else:
                hmm = ParallelLRHMM(pinyin=pho[0][2])

            transcription_decoded   = hmm.getTranscription()

            if am == 'gmm':
                # gmmModel
                hmm._gmmModel(gmmModel_path)

            # MFCC feature
            sf = round(pho[0][0]*fs/hopsize)
            ef = round(pho[0][1]*fs/hopsize)

            # mfcc syllable
            mfcc_s  = mfcc[sf:ef,:]

            # mfcc_s = np.expand_dims(mfcc_s[:,0],axis=1)
            # print mfcc_s.shape

            if explicit_dur:
                path = hmm._viterbiHSMM(observations=mfcc_s)
            else:
                if am == 'keras':
                    path = hmm._viterbiLogTranstionVarying(mfcc_s,km)
                else:
                    path = hmm._viterbiLogTranstionVarying(observations=mfcc_s)

            # print path

            # hmm._plotNetwork(path)

            if pho_duration_threshold > 0:
                path = hmm._postProcessing(path,pho_duration_threshold=pho_duration_threshold)

            if plot:
                hmm._pathPlot(transcription_gt= transcription,frames_jump_gt=frames_jump,path=path)

            ##-- evaluation
            time_boundary_gt = np.array(frames_jump)*hopsize/float(fs)

            # boundary decoded
            time_boundary_decoded           = []
            pho_decoded                     = []
            for ii_path in xrange(1,len(path)):
                if path[ii_path] != path[ii_path-1]:
                    time_boundary_decoded.append(ii_path)
                    pho_decoded.append(transcription_decoded[int(path[ii_path])])
            time_boundary_decoded           = np.array(time_boundary_decoded)*hopsize/float(fs)

            if patternMethod:
                # MBE syllable
                MBE_s                                   = MBE[sf:ef,:]
                time_boundary_decoded_pattern_method    = processPatternMethod(transcription_decoded,MBE_s,path,time_boundary_decoded,model_classification)

                # obtain decoded pho_list
                time_boundary_deleted                   = np.setdiff1d(time_boundary_decoded,time_boundary_decoded_pattern_method)
                idx_boundary_deleted                    = [time_boundary_decoded.tolist().index(bd) for bd in time_boundary_deleted.tolist()]
                pho_decoded                             = [pho_decoded[ii] for ii in range(len(time_boundary_decoded)) if ii not in idx_boundary_deleted]
                time_boundary_decoded                   = time_boundary_decoded_pattern_method

            pho_decoded                             = [transcription_decoded[int(path[0])]] + pho_decoded

            if len(pho_decoded) > 1:
                detectedTransPhoneme = []
                for ii_pd in range(len(pho_decoded)-1):
                    detectedTransPhoneme.append([ii_pd,pho_decoded[ii_pd],pho_decoded[ii_pd+1]])
                detectedTransPhonemeAll     += detectedTransPhoneme

            numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, correctTransPhoneme = \
                metrics.boundaryDetection(groundtruthBoundaries=time_boundary_gt,
                                        detectedBoundaries=time_boundary_decoded,
                                        groundtruthTransPhoneme=trans_phoneme_gt,
                                        tolerance=tol)

            numDetectedBoundariesAll    += numDetectedBoundaries
            numGroundtruthBoundariesAll += numGroundtruthBoundaries
            numCorrectAll               += numCorrect
            correctTransPhonemeAll      += correctTransPhoneme

    # evaluate part of the boundaries
    if valid_string != 'valid':
        HR, OS, FAR, F, R, deletion, insertion = \
        metrics.metrics(numDetectedBoundariesAll, numGroundtruthBoundariesAll, numCorrectAll)

        print ('ground truth %i, detected %i, correct %i' % (numGroundtruthBoundariesAll,numDetectedBoundariesAll,numCorrectAll))
    else:
        numValidTransDetected,numValidTransGt,numValidTransCorrect = validTransStat(detectedTransPhonemeAll,correctTransPhonemeAll)

        HR, OS, FAR, F, R, deletion, insertion = \
            metrics.metrics(numValidTransDetected, numValidTransGt, numValidTransCorrect)

        print ('ground truth %i, detected %i, correct %i' % (numValidTransGt,numValidTransDetected,numValidTransCorrect))

    print ('HR %.3f, OS %.3f, FAR %.3f, F %.3f, R %.3f, deletion %i, insertion %i' %
                   (HR, OS, FAR, F, R, deletion, insertion))

    return HR, OS, FAR, F, R, deletion, insertion

if __name__ == '__main__':
    ####---- correct transition phonemes
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                    tol=0.04,
                                                    bool_transVaryProb=False,
                                                    patternMethod=False,
                                                    model_classification='xgb',
                                                    valid_string='idxresort',
                                                    explicit_dur=False,
                                                    plot=True)