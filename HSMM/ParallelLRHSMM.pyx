# encoding: utf-8
# cython: embedsignature=True

import numpy as np
cimport numpy as np
cimport cython
import time
import json
import matplotlib.pyplot as plt

from LRHSMM import _LRHSMM
from makeParallelLRHSMMNet import MakeParallelLRHSMMNet
from src.phonemeMap import *
from src.parameters import *


def transcriptionMapping(transcription):
    transcription_maped = []
    for t in transcription:
        transcription_maped.append(dic_pho_map[t])
    return transcription_maped

class ParallelLRHSMM(_LRHSMM):

    def __init__(self,pinyin):
        _LRHSMM.__init__(self,pinyin)

        self.gmmModel       = {}

        self.net            = MakeParallelLRHSMMNet(pinyin)
        self._makeNet()

        self.phos_final     = self.net.getPhoFinals()
        self.idx_final_head = self.net.getIndexFinalHead()
        self.idx_final_tail = self.net.getIndexFinalTail()
        self.occurrenceProbasHead = self.net.getOccurrenceProbasHead()

        self.transcription  = transcriptionMapping(self.net.getStates())
        self.n              = len(self.transcription)

        self._loadPhoDurDist()
        self._initialStateDist()

    def _loadPhoDurDist(self):
        '''
        load phoneme duration distribution dictionary
        :return:
        '''
        self.dict_dur_dist = json.load(open(path_dict_dur_dist,'r'))

    def _initialStateDist(self):
        '''
        explicitly set the initial state distribution
        '''
        # list_forced_beginning = [u'nvc', u'vc', u'w']
        self.pi     = np.zeros((self.n), dtype=self.precision)

        # each final head has a change to start
        for ii_idx,idx in enumerate(self.idx_final_head):
            self.pi[idx] = self.occurrenceProbasHead[ii_idx]

    def _makeNet(self):
        self.A = self.net.build()

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
        p_max   = -float("inf") # max value in time T (max)
        path    = np.zeros((len(observations)),dtype=self.precision)

        # last path is self.n-1
        # path[len(observations)-1] = self.n-1
        # for i in xrange(self.n):
        #     if (p_max < delta[len(observations)-1][i]):
        #         p_max = delta[len(observations)-1][i]
        #         path[len(observations)-1] = i
        for i in xrange(self.n):
            # decode only possible from the final node of each path
            ii = i-1 if self.n > len(self.phos_final) else i
            if ii in self.idx_final_tail:
                endingProb = 1.0
            else:
                endingProb = 0.0

            if (p_max < delta[len(observations)-1][i]+np.log(endingProb)):
                p_max = delta[len(observations)-1][i]+np.log(endingProb)
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]

        return path

    def viterbiTransitionVaryingHelper(self):
        '''
        helper function for preparing the variables for viterbi log transition proba varying decoding
        :return:
        '''
        A   = np.zeros((self.n,self.n),dtype=object)

        ##-- loop relevant transition paths
        # nodes index of loop entering, next transition and self
        transitionVaryingProbLoopEnterTrans,transitionVaryingProbLoopSkipTrans,\
        transitionVaryingProbLastLoopSelfTrans,transitionVaryingProbLastLoopEnterTrans = \
            self.net.getTransitionVaringTrans()

        idx_loop_next_enter = []    # loop entering node transition
        idx_loop_next_skip = []     # skip transition, skip the loop node
        idx_loop_last_self = []     # self transition of the loop in the end of the network
        idx_loop_last_enter = []    # entering transition of the loop in the end of the network
        for ii in xrange(self.n):
            for jj in xrange(self.n):
                if isinstance(self.A[ii][jj],np.ndarray):
                    # this code has a flaw, because transitionVaryingProbLoopEnterTrans is equal to
                    # transitionVaryingProbLastLoopEnterTrans, idx_loop_last_enter will be empty,
                    # each last loop enter trans will be saved in idx_loop_next_enter
                    if np.array_equal(self.A[ii][jj],transitionVaryingProbLoopEnterTrans):
                        idx_loop_next_enter.append([ii,jj])
                    elif np.array_equal(self.A[ii][jj],transitionVaryingProbLoopSkipTrans):
                        idx_loop_next_skip.append([ii,jj])
                    elif np.array_equal(self.A[ii][jj],transitionVaryingProbLastLoopSelfTrans):
                        idx_loop_last_self.append([ii,jj])
                    elif np.array_equal(self.A[ii][jj],transitionVaryingProbLastLoopEnterTrans):
                        idx_loop_last_enter.append([ii,jj])
                A[ii][jj] = self.A[ii][jj]

        idx_loop_enter      = idx_loop_next_enter + idx_loop_last_enter
        idx_loop_skip_self  = idx_loop_next_skip + idx_loop_last_self

        idx_loop_enter.sort(key=lambda x: x[0])
        idx_loop_skip_self.sort(key=lambda x: x[0])

        # loop entering time tracker
        tracker_loop_enter  = np.zeros((self.n,len(idx_loop_enter)),dtype=np.int)

        return A,idx_loop_enter,idx_loop_skip_self,tracker_loop_enter

    def _inferenceInit(self,observations):
        '''
        HSMM inference initialization
        :param observations:
        :return:
        '''

        tau = len(observations)

        # Forward quantities
        forwardDelta        = np.ones((self.n,tau),dtype=self.precision)
        forwardDelta        *= -float('inf')
        previousState       = np.zeros((self.n,tau),dtype=np.intc)
        state               = np.zeros((self.n,tau),dtype=np.intc)
        occupancy           = np.zeros((self.n,tau),dtype=np.intc)

        # State-in
        # REMARK : state-in a time t is StateIn(:,t+1), such that StateIn(:1) is
        # the initial distribution
        stateIn             = np.ones((self.n,tau),dtype=self.precision)
        stateIn             *= -float('inf')

        # Set initial states distribution \pi %%%%            % PARAMETERs
        # stateIn[:,0]        = np.log(self.pi)

        # simplify A
        A   = np.zeros((self.n,self.n),dtype=self.precision)
        for jj in xrange(self.n):
            for ii in xrange(self.n):
                if isinstance(self.A[ii][jj],np.ndarray):
                    A[ii][jj] = self.A[ii][jj][0]
                else:
                    A[ii][jj] = self.A[ii][jj]

        return forwardDelta,\
               previousState,\
                state,\
               stateIn,\
                occupancy,\
                A

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _viterbiHSMM(self,observations):

        forwardDelta,\
           previousState,\
            state,\
           stateIn,\
            occupancy,\
            A       = self._inferenceInit(observations)

        tau = len(observations)

        cdef double [:, ::1] cA = np.log(A)
        cdef double [:, ::1] cforwardDelta = forwardDelta
        cdef int [:, ::1] cpreviousState = previousState
        cdef int [:, ::1] cstate         = state
        cdef double [:, ::1] cstateIn = stateIn
        cdef int [:, ::1] coccupancy     = occupancy

        # calculate the observation probability and normalize for each frame into pdf sum(B_map[:,t])=1
        if am == 'gmm':
            self._mapBGMM(observations)
            obs = np.exp(self.B_map)
            obs /= np.sum(obs,axis=0)
            obs = np.log(obs)
        elif am == 'dnn':
            self._mapBDNN(observations)
            obs = self.B_map
        else:
            print(am+' is not exist.')
            raise

        cdef double [:, ::1] cobs = obs
        cdef double [::1] cpi  = np.log(self.pi)

        cdef double cmaxForward = -float('inf') # max value in time T (max)
        cdef int [::1] cpath = np.zeros((tau),dtype=np.intc)
        # print pi
        # print self.net.getStates()
        # print self.transcription


        # get transition probability and others
        # A,idx_loop_enter,idx_loop_skip_self,tracker_loop_enter = self.viterbiTransitionVaryingHelper()

        # print self.A

        # version in guedon 2003 paper
        for t in xrange(0,tau):

            for j in xrange(self.n):

                # Compute occupancies distributions
                d_j             = self.dict_dur_dist[self.transcription[j]]       # d
                D_j             = 1.0-np.cumsum(d_j)                              # D
                M_j             = len(d_j)-1
                d_j             = np.log(d_j)
                D_j             = np.log(D_j)

                # print M_j
                # print tau

                observ          = 0.0

                if t<tau-1:
                    for u in xrange(1,min(t+1,M_j)+1):
                        observ += cobs[j,t-u+1]
                        if u < t+1:
                            prod_occupancy = observ+d_j[u]+cstateIn[j,t-u+1]
                            # print t, j, prod_occupancy, observ, d_j[u], stateIn[j,t-u+1]
                            if prod_occupancy > cforwardDelta[j,t]:
                                cforwardDelta[j,t]   = prod_occupancy
                                cpreviousState[j,t]  = cstate[j,t-u+1]
                                coccupancy[j,t]      = u
                        else:
                            # print u, len(occupancies_j)
                            prod_occupancy = observ+d_j[t+1]+cpi[j]
                            # print t, j, prod_occupancy, observ, d_j[u], pi[j]
                            if prod_occupancy > cforwardDelta[j,t]:
                                cforwardDelta[j,t]   = prod_occupancy
                                coccupancy[j,t]      = t+1

                else:
                    for u in xrange(1,min(tau,M_j)+1):
                        # observ *= obs[j,tau-u]
                        observ += cobs[j,tau-u]
                        if u < tau:
                            prod_survivor = observ+D_j[u]+cstateIn[j,tau-u]
                            # print t, j, prod_survivor, observ, D_j[u], stateIn[j,tau-u]
                            if prod_survivor > cforwardDelta[j,tau-1]:
                                cforwardDelta[j,tau-1]   = prod_survivor
                                cpreviousState[j,t]      = cstate[j,tau-u]
                                coccupancy[j,tau-1]      = u

                        else:
                            prod_survivor = observ+D_j[u]+cpi[j]
                            # print t, j, prod_survivor, observ, D_j[u], pi[j]
                            if prod_survivor > cforwardDelta[j,tau-1]:
                                cforwardDelta[j,tau-1]   = prod_survivor
                                coccupancy[j,tau-1]      = tau

            # ignore normalization

            if t<tau-1:
                for j in xrange(self.n):
                    for i in xrange(self.n):

                        if cstateIn[j,t+1] < cA[i][j] + cforwardDelta[i,t]:
                            cstateIn[j,t+1]        = cA[i][j] + cforwardDelta[i,t]
                            cstate[j,t+1]          = i


        # termination: find the maximum probability for the entire sequence (=highest prob path)

        for i in xrange(self.n):
            # decode only possible from the final node of each path
            ii = i-1 if self.n > len(self.phos_final) else i
            if ii in self.idx_final_tail: endingProb = np.log(1.0)
            else: endingProb = np.log(0.0)

            # print stateIn[i][len(observations)-1]
            if (cmaxForward < cforwardDelta[i][tau-1]+endingProb):
                cmaxForward = cforwardDelta[i][tau-1]+endingProb
                cpath[tau-1] = i

        t = tau-1

        while t>=0:
            j = cpath[t]
            u = coccupancy[j,t]
            # print t,j,u
            for v in xrange(1,u):
                cpath[t-v] = j
            if t >= u:
                cpath[t-u] = cpreviousState[j,t]
            t = t-u

        # avoid memory leaks
        cA              = None
        cforwardDelta   = None
        cpreviousState  = None
        cstate          = None
        cstateIn        = None
        coccupancy      = None

        cobs            = None
        cpi             = None
        # cmaxForward     = None

        return cpath


    def _viterbiTransitionVarying(self,observations):
        '''
        see description of _viterbiLog and parent method in LRHMM.py
        :param observations:
        :return:
        '''
        # calculate the observation probability and normalize for each frame into pdf sum(B_map[:,t])=1
        self._mapBGMM(observations)
        B_map = np.exp(self.B_map)
        B_map /= np.sum(B_map,axis=0)

        pi  = self.pi

        # EXPERIMENT
        A,idx_loop_enter,idx_loop_skip_self,tracker_loop_enter = self.viterbiTransitionVaryingHelper()

        delta   = np.zeros((len(observations),self.n),dtype=self.precision)
        # delta   *= -float('inf')
        psi     = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x in xrange(self.n):
            delta[0][x] = pi[x]*B_map[x][0]
            psi[0][x] = 0

        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):

                    # EXPERIMENT
                    # analysis of trans proba according to path [i,j]
                    if [i,j] not in idx_loop_enter + idx_loop_skip_self:
                        proba_trans = A[i][j]
                    else:
                        if [i,j] in idx_loop_enter:
                            idx_ij = idx_loop_enter.index([i,j])
                        else:
                            # [i,j] in idx_loop_skip_self
                            idx_ij = idx_loop_skip_self.index([i,j])

                        if tracker_loop_enter[i][idx_ij] < len(A[i][j]):
                            # tracker_loop_enter[i][idx_ij] is the loop entering [i, j] transition time at time t-1
                            proba_trans = A[i][j][tracker_loop_enter[i][idx_ij]]
                        else:
                            proba_trans = A[i][j][-1]

                    # induction max
                    if (delta[t][j] < delta[t-1][i]*proba_trans):
                        delta[t][j] = delta[t-1][i]*proba_trans
                        psi[t][j] = i

                # EXPERIMENT
                # state j at time t will inherit the tracker of state psi[t][j] at time t - 1
                tracker_loop_enter[j] = tracker_loop_enter[psi[t][j]]
                # update the tracker if [psi[t][j],j] is an entering transition
                if [psi[t][j],j] in idx_loop_enter:
                    idx_ij = idx_loop_enter.index([psi[t][j],j])
                    tracker_loop_enter[j][idx_ij] = tracker_loop_enter[psi[t][j]][idx_ij] + 1

                delta[t][j] *= B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0 # max value in time T (max)
        path = np.zeros((len(observations)),dtype=self.precision)

        for i in xrange(self.n):
            # decode only possible from the final node of each path
            ii = i-1 if self.n > len(self.phos_final) else i
            if ii in self.idx_final_tail: endingProb = 1.0
            else: endingProb = 0.0

            print delta[len(observations)-1][i]
            if (p_max < delta[len(observations)-1][i]*endingProb):
                p_max = delta[len(observations)-1][i]*endingProb
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]

        return path

    def _plotNetwork(self,path):
        self.net.plotNetwork(path)

    def _pathPlot(self,transcription_gt,frames_jump_gt,path):
        '''
        plot ground truth path and decoded path
        :return:
        '''

        ##-- unique transcription and path
        transcription_unique = []
        transcription_number_unique =[]
        for ii,t in enumerate(self.transcription):
            if t not in transcription_unique:
                transcription_unique.append(t)
                transcription_number_unique.append(ii)

        B_map_unique = []
        for ii in transcription_number_unique:
            B_map_unique.append(self.B_map[ii,:])
        B_map_unique = np.array(B_map_unique)

        trans2transUniqueMapping = {}
        for ii in range(len(self.transcription)):
            trans2transUniqueMapping[ii] = transcription_unique.index(self.transcription[ii])

        path_unique = []
        for ii in range(len(path)):
            path_unique.append(trans2transUniqueMapping[path[ii]])


        ##-- figure plot
        plt.figure()
        print self.B_map.shape
        y = np.arange(B_map_unique.shape[0]+1)
        x = np.arange(B_map_unique.shape[1])*hopsize/float(fs)
        fjs_gt = np.array(frames_jump_gt)*hopsize/float(fs)
        plt.pcolormesh(x,y,B_map_unique)
        plt.plot(x,path_unique,'b',linewidth=3)
        plt.text (0,B_map_unique.shape[0]*1.05,transcription_gt[0])
        for ii,fj in enumerate(fjs_gt):
            plt.plot((fj,fj),(0,B_map_unique.shape[0]),'k-',linewidth=3)
            plt.text(fj,B_map_unique.shape[0]*1.05,transcription_gt[ii+1])
        plt.xlabel('time (s)')
        plt.ylabel('states')
        plt.yticks(y, transcription_unique, rotation='horizontal')
        plt.show()

