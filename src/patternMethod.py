__author__ = 'gong'

import numpy as np
from os import path
from sklearn import preprocessing
from sklearn.externals import joblib
from src.figurePlot import pltBoundaryPattern
from parameters import *

svm_model_filename    = path.join(svmPatternModel_path,'svm_model_biccPattern.pkl')
xgb_model_filename    = path.join(svmPatternModel_path,xgboost_model,'xgb_model_biccPattern.pkl')


def patternPadding(feature,boundary_frame,N):

    len_feature     = feature.shape[0]
    if boundary_frame-int(N/2) < 0 and boundary_frame+int(N/2)+1 <= len_feature:
        # left padding
        p           = abs(boundary_frame-int(N/2))
        pad_left    = np.repeat(np.array([feature[0,:]]),p,axis=0)
        pattern     = np.vstack((pad_left,feature[0:boundary_frame+int(N/2)+1,:]))
    elif boundary_frame-int(N/2) >= 0 and boundary_frame+int(N/2)+1 > len_feature:
        # right padding
        q           = abs(boundary_frame+int(N/2)+1 - len_feature)
        pad_right   = np.repeat(np.array([feature[-1,:]]),q,axis=0)
        # print 'feature shape' ,feature[boundary_frame-int(N/2):,:].shape
        # print 'pad_right shape', pad_right.shape
        pattern     = np.vstack((feature[boundary_frame-int(N/2):,:],pad_right))
    elif boundary_frame-int(N/2) < 0 and boundary_frame+int(N/2)+1 > len_feature:
        # left and right padding
        p           = abs(boundary_frame-int(N/2))
        pad_left    = np.repeat(np.array([feature[0,:]]),p,axis=0)
        q           = abs(boundary_frame+int(N/2)+1 - len_feature)
        pad_right   = np.repeat(np.array([feature[-1,:]]),q,axis=0)
        pattern     = np.vstack((pad_left,feature,pad_right))
    else:
        pattern             = feature[boundary_frame-int(N/2):boundary_frame+int(N/2)+1,:]

    # print pattern.shape

    return pattern

def scalerPattern(pattern):

    scaler              = preprocessing.StandardScaler().fit(pattern)
    pattern_scaled      = scaler.transform(pattern)
    return pattern_scaled


def icdPatternCollection(feature, icd, varin):

    icdPatterns = []
    index       = []        # pattern index in icd
    N = varin['N_pattern']
    for ii in range(len(icd)):

        # if icd_frame-int(N/2)<0 or icd_frame+int(N/2)+1>feature.shape[0]:
        #     continue
        #
        # icdPattern      = feature[icd_frame-int(N/2):icd_frame+int(N/2)+1,:]

        icd_frame       = icd[ii]

        icdPattern      = patternPadding(feature,icd_frame,N)
        # pltBoundaryPattern(icdPattern.transpose())
        icdPattern      = scalerPattern(icdPattern)
        icdPattern      = np.reshape(icdPattern, varin['N_feature']*varin['N_pattern'])
        icdPatterns.append(icdPattern)
        index.append(ii)

    return icdPatterns,index

def predict(voicedPatterns,model='svm'):
    if model=='svm':
        model_object    = joblib.load(svm_model_filename)
    else:
        model_object    = joblib.load(xgb_model_filename)
    target              = model_object.predict(voicedPatterns)
    return target

def targetFrameBoundary(feature,frame_boundary_voiced,model='svm'):
    '''
    predict target frame boundary
    :param feature:
    :param frame_boundary_voiced:
    :return:
    '''
    patterns,index      = icdPatternCollection(feature,frame_boundary_voiced,varin)
    target_patterns     = predict(patterns,model)

    return np.array(frame_boundary_voiced)[np.nonzero(1-target_patterns)]

def processPatternMethod(transcription_decoded,MBE_s,path,time_boundary_decoded,model):
    '''

    :param transcription_decoded: HMM states transcription
    :param MBE_s: mel band energy syllable level
    :param time_boundary_decoded: decoded boundary in time by HMM
    :param path: decoded path
    :param model: svm or xgb
    :return:
    '''
    # boundary decoded in frame index, between voiced phones
    frame_boundary_voiced_decoded   = []
    for ii_path in xrange(1,len(path)):
        if path[ii_path] != path[ii_path-1]:
            pho_decoded_left      = transcription_decoded[int(path[ii_path-1])]
            #pho_decoded_right     = transcription_decoded[int(path[ii_path])]
            if pho_decoded_left != 'nvc' and pho_decoded_left != 'vc':
                frame_boundary_voiced_decoded.append(ii_path)

    time_boundary_voiced_decoded    = np.array(frame_boundary_voiced_decoded)*hopsize/float(fs)

    frame_boundary_pattern_correct = np.array([])
    if len(frame_boundary_voiced_decoded):
        frame_boundary_pattern_correct = targetFrameBoundary(MBE_s,frame_boundary_voiced_decoded,model)

    time_boundary_decoded = np.hstack((np.setdiff1d(time_boundary_decoded,time_boundary_voiced_decoded),
                                       frame_boundary_pattern_correct*hopsize/float(fs)))

    return time_boundary_decoded



