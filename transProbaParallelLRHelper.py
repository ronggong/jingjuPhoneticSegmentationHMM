'''
code to generate all the prior information for building the hmm network

Parallel left-right network
'''

import pickle
from os import path

import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import factorial
import matplotlib.pyplot as plt

from src.phonemeMap import *
from src.pinyinMap import *
from src.parameters import *
# from src.trainTestSeparation import getRecordings, getRecordingNumber
from src.recordingNumber import recordings_train
from src.textgridParser import syllableTextgridExtraction


def getPossibleSampa4Finals(finals,pho,pho_finals,pho_set_initials):
    '''
    possible x sampas for finals
    :param finals: list of all finals
    :param pho:
    :param pho_finals: dictionary return
    :param pho_set_initials: list of all initials x sampa
    :return:
    '''
    for final in finals:
        # get rid of the initials in pinyin
        final_pinyin_test = dic_pinyin_2_initial_final_map[pho[0][2]]['final']

        if final == final_pinyin_test:
            pho_final_syllable = []
            for ii,p in enumerate(pho[1]):
                # get rid of the initials in pho
                if not(p[2] in pho_set_initials and ii==0):
                    pho_final_syllable.append(p[2])

            pho_finals[final].append(pho_final_syllable)

    return pho_finals

####---- final loop functions

def getPossibleFinalsLoop(pho_finals):

    '''
    all finals which contain x, ?, sil
    :param pho_finals:
    :return:
    '''
    pho_finals_loop = {}
    for final in pho_finals:
        pho_finals_loop[final] = []
        for pho_final_syllable in pho_finals[final]:
            if u'x' in pho_final_syllable or u'' in pho_final_syllable or u'?' in pho_final_syllable:
                pho_finals_loop[final].append(pho_final_syllable)

    return pho_finals_loop

def getPossibleLoopPairs(pho_finals):
    '''
    possible loop pairs can be connect into the network
    loop 1 or 2 phonemes with sil, x or ?
    :param pho_finals:
    :return:
    '''

    loopPairs = {}
    for final in pho_finals:
        loopPairs[final] = []
        for pho_final in pho_finals[final]:
            for ii in range(len(pho_final)):
                if pho_final[ii] == u'' or pho_final[ii] == u'x' or pho_final[ii] == u'?':
                    # repetition of previous 2 phonemes
                    if ii>1 and ii<len(pho_final)-2 and [pho_final[ii-2],pho_final[ii-1]] == [pho_final[ii+1],pho_final[ii+2]]:
                        loopPairs[final].append([pho_final[ii-2],pho_final[ii-1],pho_final[ii]])

                    # repetition of previous 1 phoneme
                    elif ii > 0 and ii < len(pho_final)-1 and pho_final[ii-1] == pho_final[ii+1]:
                        loopPairs[final].append([pho_final[ii-1],pho_final[ii]])

                    else:
                        pass

        loopPairsUnique = []
        for pair in loopPairs[final]:
            if pair not in loopPairsUnique:
                loopPairsUnique.append(pair)
        loopPairs[final] = loopPairsUnique
    return loopPairs

def indexIdenticalStringHelper(baseString,toSearchString):
    '''
    find index of toSearchString in baseString
    :param baseString:
    :param toSearchString:
    :return:
    '''
    if len(baseString) < len(toSearchString):
        return []
    else:
        indexIdentical = []
        for ii in range(len(baseString)-len(toSearchString)+1):
            if baseString[ii:ii+len(toSearchString)] == toSearchString:
                if len(toSearchString) == 3:
                    indexIdentical += range(ii,ii+len(toSearchString))
                else:
                    indexIdentical += range(ii,ii+len(toSearchString)-1)

        indexIdentical_unique = []
        if len(indexIdentical):
            for idxIden in indexIdentical:
                if idxIden not in indexIdentical_unique:
                    indexIdentical_unique.append(idxIden)
        return indexIdentical_unique

def loopSubstituteHelper(p_final, idx_identical, lp, counter):

    p_final_copy = list(p_final)
    # collect consecutive index
    idx_identical_list = []
    temp_list = []

    for ii in range(len(idx_identical)-1):
        temp_list.append(idx_identical[ii])
        if idx_identical[ii]+1 != idx_identical[ii+1]:
            idx_identical_list.append(temp_list)
            temp_list = []

    temp_list.append(idx_identical[-1])
    idx_identical_list.append(temp_list)

    # substitute index
    idx_idx_identical_list = range(len(idx_identical_list))
    for ii in idx_idx_identical_list[::-1]:
        idx_iden = idx_identical_list[ii]
        for jj in idx_iden[::-1][:-1]:
            p_final_copy.pop(jj)
            p_final_copy[idx_iden[0]] = 'lp_'+str(counter)

    return p_final_copy

def finalsLoopFold(pho_finals,loopPairs):
    '''
    fold the loop section into loop pair list in phoneme finals, and calculate the loop times
    :param pho_finals:
    :param loopPairs:
    :return:
    '''

    pho_finals_loopFold = {}
    loopTimes = []
    for final in pho_finals:
        pho_finals_loopFold[final] = []
        if len(loopPairs[final]):

            # loop pairs which length == 2 or 3
            lpsLen3 = []
            lpsLen2 = []
            for lp in loopPairs[final]:
                if len(lp) == 3:
                    lpsLen3.append(lp+lp[:2])
                elif len(lp) == 2:
                    lpsLen2.append(lp+lp[0:1])

            for pf in pho_finals[final]:
                counter = 0
                lps_sub = []
                pf_loopFold = pf
                if len(lpsLen3):
                    for lp in lpsLen3:
                        idx_identical_3 = indexIdenticalStringHelper(pf,lp)
                        if idx_identical_3: # only consider when there is loop
                            loopTimes.append((len(idx_identical_3)-1)/3)
                        if len(idx_identical_3):
                            pf_loopFold = loopSubstituteHelper(pf,idx_identical_3,lp,counter)
                            lps_sub.append(lp)
                            counter += 1
                if len(lpsLen2):
                    for lp in lpsLen2:
                        idx_identical_2 = indexIdenticalStringHelper(pf_loopFold,lp)
                        if idx_identical_2:
                            loopTimes.append((len(idx_identical_2)-1)/2)
                        if len(idx_identical_2):
                            pf_loopFold = loopSubstituteHelper(pf_loopFold,idx_identical_2,lp,counter)
                            lps_sub.append(lp)
                            counter += 1

                if counter > 0:
                    for ii,lp in enumerate(lps_sub):
                        lp = lp[:2] if len(lp) == 3 else lp[:3]
                        for jj in range(len(pf_loopFold)):
                            if pf_loopFold[jj] == 'lp_' + str(ii):
                                pf_loopFold[jj] = lp

                pho_finals_loopFold[final].append(pf_loopFold)
        else:
            pho_finals_loopFold[final] = pho_finals[final]

    return pho_finals_loopFold,loopTimes

def loopFoldUniqueEntire(pho_entire_loopFold):
    '''
    unique loop fold for each pinyin
    the structure of pho_entire_loopFold_unique[py] is dictionary
    key: loop fold
    value: occurrence probability
    :param pho_entire_loopFold:
    :return:
    '''
    # collect the unique loop fold and occurrence
    pho_entire_loopFold_unique = {}
    for py in pho_entire_loopFold:
        pho_entire_loopFold_unique[py] = {}
        for pelf in pho_entire_loopFold[py]:
            pelf_tuple = tuple(tuple(x) if isinstance(x, list) else x for x in pelf )
            if pelf_tuple not in pho_entire_loopFold_unique[py].keys():
                pho_entire_loopFold_unique[py][pelf_tuple] = 1
            else:
                pho_entire_loopFold_unique[py][pelf_tuple] += 1

        # convert the occurrence into probability
        sum_values_py = float(sum(pho_entire_loopFold_unique[py].values()))
        for pelfu in pho_entire_loopFold_unique[py]:
            pho_entire_loopFold_unique[py][pelfu] /= sum_values_py
    return pho_entire_loopFold_unique


####---- functions for evaluating the transition varying proba

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def loopTimeDistribution(array_loopTimes):
    '''
    fit poisson distribution to loop times histogram
    :param array_loopTimes:
    :return:
    '''

    # plt.figure(figsize=(10, 6))
    # start from 0
    array_loopTimes = [lt-1 for lt in array_loopTimes]

    # integer bin edges
    bins = np.arange(0, max(array_loopTimes)+2, 1) - 0.5

    # histogram
    entries, bin_edges, patches = plt.hist(array_loopTimes, bins=bins, normed=True, fc=(0, 0, 1, 0.7),label='Looping occurrence histogram')

    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

    parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)

    x = np.linspace(0, max(array_loopTimes), max(array_loopTimes)+1)

    # star before parameters is unpacking the tuple
    p = poisson(x, *parameters)

    # plt.plot(x,p,'r',linewidth=2,label='Poisson distribution fitting curve')
    # plt.legend(fontsize=18)
    # plt.xlabel('Looping occurrence',fontsize=18)
    # plt.ylabel('Probability',fontsize=18)
    # plt.axis('tight')
    # plt.tight_layout()
    # plt.show()

    dict_loopTimesDist = {}
    for ii in xrange(1,max(array_loopTimes)+2):
        dict_loopTimesDist[ii] = p[ii-1]

    return dict_loopTimesDist

def loopEnterSkipTransProbas(dict_loopTimes,enterPenalize=0.0):
    '''
    enter and skip loop transition probabilities
    the index of array is looping time
    example:
        skip trans proba
        0   0.0     the first loop, not possible to skip
        1   0.6849  the second loop, it has 68.5% probability to skip the loop
        ...
    :param dict_loopTimes:
    :return:
    '''

    transitionVaryingProbLoopEnterTrans = [1.0]
    transitionVaryingProbLoopSkipTrans = [0.0]
    for ii in range(dict_loopTimes.keys()[-1]-1):
        # skip trans proba conditional to loop time ii = proba of just looping ii times / proba of at least looping ii times
        probEnterTrans = (sum(dict_loopTimes.values()[ii+1:])/float(sum(dict_loopTimes.values()[ii:])))*(2**(-enterPenalize))
        probSkipTrans = 1.0-probEnterTrans
        transitionVaryingProbLoopEnterTrans.append(probEnterTrans)
        transitionVaryingProbLoopSkipTrans.append(probSkipTrans)

    return transitionVaryingProbLoopEnterTrans,transitionVaryingProbLoopSkipTrans

def processTransPriorIntoParallelLR(bool_transitionVaryingProbLoop=False,ep=0.0):

    ##-- recording names
    # recordings = getRecordings(wav_path)
    # number_train = getRecordingNumber('TRAIN')
    # recordings_train  = [recordings[i] for i in number_train]

     ##-- create bigram model initial matrix
    pho_entire                  = {}

    for recording in recordings_train:

        boundaries_oneSong  = 0
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        for pho in nestedPhonemeLists:
            pho_list_syl    = [p[2] for p in pho[1]]

            # exp
            if pho[0][2] in pho_entire:
                pho_entire[pho[0][2]].append(pho_list_syl)
            else:
                pho_entire[pho[0][2]]=[pho_list_syl]

    # exp
    pho_entire_loop         = getPossibleFinalsLoop(pho_entire)
    loopPairs_entire        = getPossibleLoopPairs(pho_entire_loop)
    pho_entire_loopFold,array_loopTimes_entire = finalsLoopFold(pho_entire,loopPairs_entire)
    pho_entire_loopFold_unique     = loopFoldUniqueEntire(pho_entire_loopFold)

    if bool_transitionVaryingProbLoop:
        dict_loopTimes          = loopTimeDistribution(array_loopTimes_entire)

        transitionVaryingProbLoopEnterTrans,transitionVaryingProbLoopSkipTrans = \
            loopEnterSkipTransProbas(dict_loopTimes,enterPenalize=ep)
    else:
        transitionVaryingProbLoopEnterTrans = [0.5]*100
        transitionVaryingProbLoopSkipTrans  = [0.5]*100

    # print transitionVaryingProbLoopSkipTrans
    # print transitionVaryingProbLoopEnterTrans
    ####---- End experiment


    # -- print unique final set
    # for final in finals:
    #     non_repetitive_values = []
    #     # for value in pho_finals[key]:
    #     #     if value not in non_repetitive_values:
    #     #         non_repetitive_values.append(value)
    #     # pho_finals[key] = non_repetitive_values
    #     print final, pho_finals_loopFold[final]

    # ##-- bigram initials
    # bigramInitials  = convertBigramModelInitialsCount2Dist(bigramInitials)
    #
    # for key in pho_finals:
    #     pho_finals_xsampa_unique[key] = list(set(sum(pho_finals[key],[])))
    #
    # dictBeginningDist, dictEndingDist = beginningEndingDist(pho_finals,pho_finals_xsampa_unique)

    output = open(path.join(transPriorInfo_path,'transPriorInfoParallelLR.pkl'),'wb')

    g = [pho_entire_loopFold_unique,
         loopPairs_entire,
         transitionVaryingProbLoopEnterTrans,
         transitionVaryingProbLoopSkipTrans]

    pickle.dump(g, output)
    output.close()

    # for final in finals:
    #     # print final, bigramInitials[final]
    #     print final, pho_finals_xsampa_unique[final],bigramFinals[final],loopPairs[final],dictBeginningDist[final],dictEndingDist[final]

if __name__ == '__main__':

    processTransPriorIntoParallelLR(bool_transitionVaryingProbLoop=True,ep=0.0)