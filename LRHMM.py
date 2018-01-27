#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import pickle,cPickle,gzip

import numpy as np

import matplotlib.pyplot as plt

from src.parameters import *
from src.phonemeMap import dic_pho_label

from keras.models import load_model
sys.path.append(os.path.join(os.path.dirname(__file__), "gmmModel"))

class _LRHMM(object):
    '''
    make left to right HMM network
    '''

    def __init__(self,transcription,precision=np.double):
        self.n              = len(transcription)                                    # hidden state number
        self.pi             = np.zeros((self.n), dtype=precision)                   # initial state distribution
        self.pi[0]          = 1.0
        self.A              = np.zeros((self.n,self.n), dtype=object)            # transition matrix
        self.gmmModel       = {}

        self.transcription  = transcription                                         # the phonetic transcription in X-SAMPA
        self.precision      = precision

    def getTranscription(self):
        return self.transcription

    def _makeNet(self):
        '''
        make a left to right HMM network
        :return:
        '''

        ##-- transition matrix
        probSelfTrans   = 0.9
        probNextTrans   = 1-probSelfTrans

        for ii in xrange(self.n):
            self.A[ii][ii] = probSelfTrans

        for ii in xrange(self.n-1):
            self.A[ii][ii+1] = probNextTrans

    def _gmmModel(self,model_path):
        '''
        load gmmModel
        :return:
        '''
        for state in self.transcription:
            pkl_file = open(os.path.join(model_path,state+'.pkl'), 'rb')
            self.gmmModel[state] = pickle.load(pkl_file)
            pkl_file.close()

        # with open ('/Users/gong/desktop/original.txt','wb') as f:
        #     for key in self.gmmModel:
        #         f.write(np.array_str(self.gmmModel[key].covars_))

    @staticmethod
    def kerasModel(kerasModels_path):
        kerasModel = load_model(kerasModels_path)
        return kerasModel

    def _mapBGMM(self, observations):
        '''
        observation probability
        :param observations:
        :return:
        '''
        dim_t       = observations.shape[0]
        self.B_map  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for ii,state in enumerate(self.transcription):
            self.B_map[ii,:] = self.gmmModel[state].score_samples(observations)

            # for t in xrange(dim_t):
            #     self.B_map[state][t] = logprob[t]

    def _mapBDNN(self, observations):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''
        ##-- save the feature to pickle
        label = np.array([0] * len(observations))
        filename_feature_temp       = os.path.join(dnnModels_base_path,'feature_temp.pickle.gz')
        filename_observation_temp   = os.path.join(dnnModels_base_path,'observation_temp.pickle.gz')

        with gzip.open(filename_feature_temp, 'wb') as f:
            cPickle.dump((observations, label), f)

        ##-- set environment of the pdnn
        myenv = os.environ
        myenv['PYTHONPATH'] = '/Users/gong/Documents/pycharmProjects/pdnn'

        from subprocess import call

        ##-- call pdnn to calculate the observation from the features
        call(["/usr/local/bin/python", "/Users/gong/Documents/pycharmProjects/pdnn/cmds/run_Extract_Feats.py", "--data",
              filename_feature_temp,
              "--nnet-param", dnnModels_param_path, "--nnet-cfg", dnnModels_cfg_path,
              "--output-file", filename_observation_temp, "--layer-index", "-1",
              "--batch-size", "256"], env=myenv)

        ##-- read the observation from the output
        with gzip.open(filename_observation_temp, 'rb') as f:
            obs = cPickle.load(f)

        obs = np.log(obs)

        # print obs.shape, observations.shape

        dim_t       = observations.shape[0]
        self.B_map  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for ii,state in enumerate(self.transcription):
            self.B_map[ii,:] = obs[:, dic_pho_label[state]]

    def _mapBKeras(self, observations, kerasModel):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''
        ##-- set environment of the pdnn


        ##-- call pdnn to calculate the observation from the features
        obs = kerasModel.predict_proba(observations, batch_size=88,verbose=0)


        ##-- read the observation from the output

        obs = np.log(obs)

        # print obs.shape, observations.shape

        dim_t       = observations.shape[0]
        self.B_map  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for ii,state in enumerate(self.transcription):
            self.B_map[ii,:] = obs[:, dic_pho_label[state]]


    def _viterbi(self, observations):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapBGMM(observations)

        delta = np.ones((len(observations),self.n),dtype=self.precision)
        psi = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x in xrange(self.n):
            delta[0][x] = self.pi[x]*self.B_map[x][0]
            psi[0][x] = 0

        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                        delta[t][j] = delta[t-1][i]*self.A[i][j]
                        psi[t][j] = i
                delta[t][j] *= self.B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0 # max value in time T (max)
        path = np.zeros((len(observations)),dtype=self.precision)
        for i in xrange(self.n):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        return path

    def _viterbiLog(self, observations):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapBGMM(observations)
        pi_log  = np.log(self.pi)
        A_log   = np.log(self.A)

        delta   = np.ones((len(observations),self.n),dtype=self.precision)
        delta   *= -float("inf")
        psi     = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x in xrange(self.n):
            delta[0][x] = pi_log[x]+self.B_map[x][0]
            psi[0][x] = 0
        # print delta[0][:]

        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t-1][i] + A_log[i][j]):
                        delta[t][j] = delta[t-1][i] + A_log[i][j]
                        psi[t][j] = i
                delta[t][j] += self.B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = -float("inf") # max value in time T (max)
        path = np.zeros((len(observations)),dtype=self.precision)

        # last path is self.n-1
        # path[len(observations)-1] = self.n-1
        for i in xrange(self.n):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        return path

    def _viterbiLogTranstionVarying(self,observations):
        '''
        loop entering transition proba varies on the transition time
        :param observations:
        :return:
        '''
        pass

    def _postProcessing(self,path,pho_duration_threshold=0.02):
        '''
        post processing of the decoded path
        set a duration threshold to jump
        :param path:
        :return:
        '''

        pdt_frame = pho_duration_threshold*fs/float(hopsize)

        path_post = [path[0]]
        counter = 1
        for ii in range(1,len(path)):
            if path[ii] != path[ii-1]:

                if counter <= pdt_frame:
                    for jj in xrange(ii-counter-1,ii):
                        path_post[jj] = path[ii-counter-1]

                counter = 1
            else:
                counter += 1
            path_post.append(path[ii])
        return path_post

    def _pathPlot(self,transcription_gt,path_gt,path):
        '''
        plot ground truth path and decoded path
        :return:
        '''
        plt.figure()
        print self.B_map.shape
        y = np.arange(self.B_map.shape[0]+1)
        x = np.arange(self.B_map.shape[1])*hopsize/float(fs)
        plt.pcolormesh(x,y,self.B_map)
        plt.plot(x,path,'b',linewidth=3)
        plt.plot(x,path_gt,'k',linewidth=3)
        plt.xlabel('time (s)')
        plt.ylabel('states')
        plt.show()

