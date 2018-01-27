#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import numpy as np
import pydot

from os import path
from src.pinyinMap import *
from src.phonemeMap import *
from src.parameters import *
from makeHSMMNet import MakeHSMMNet


class MakeParallelLRHSMMNet(MakeHSMMNet):


    def __init__(self,pinyin,precision=np.double):
        MakeHSMMNet.__init__(self,pinyin)


        pkl_file = open(path.join(transPriorInfo_path,'transPriorInfoParallelLR.pkl'), 'rb')
        priorInfo = pickle.load(pkl_file)
        pkl_file.close()

        self.pho_entire_loopFold         = priorInfo[0]
        self.loopPairs                   = priorInfo[1]
        transitionVaryingProbLoopEnterTrans = priorInfo[2]
        transitionVaryingProbLoopSkipTrans  = priorInfo[3]

        # EXPERIMENT
        # transition varying transition proba
        self.transitionVaryingProbLoopEnterTrans    = np.array(transitionVaryingProbLoopEnterTrans)
        self.transitionVaryingProbLoopSkipTrans     = np.array(transitionVaryingProbLoopSkipTrans)

        self.transitionVaryingProbLastLoopEnterTrans    = np.array(transitionVaryingProbLoopEnterTrans)
        self.transitionVaryingProbLastLoopSelfTrans     = np.array(transitionVaryingProbLoopSkipTrans)*0

        self.idx_final_head         = []
        self.occurrence_probas_head = []
        self.idx_final_tail         = []


    def getTransitionVaringTrans(self):
        return [self.transitionVaryingProbLoopEnterTrans,self.transitionVaryingProbLoopSkipTrans,
                self.transitionVaryingProbLastLoopSelfTrans,self.transitionVaryingProbLastLoopEnterTrans]

    def assignFinalTransProb(self,p_final):
        '''
        assign the single final transition probabilities
        :return:
            p_final_noLoop: final phoneme list removing the parenthesis of loops
            idx_p_final:    indices of the loop elements in the final list
            transMat_sub:   transition probabilities matrix of the final
        '''
        counter         = 0
        idx_p_final     = []    # idx of loop element
        p_final_noLoop  = []
        for element_p in p_final:
            idx_temp = []
            if isinstance(element_p,tuple):
                # which means here element_p is a loop
                for element_list in element_p:
                    idx_temp.append(counter)
                    p_final_noLoop.append(element_list)
                    counter += 1
                idx_p_final.append(idx_temp)
            else:
                idx_temp = [counter]
                p_final_noLoop.append(element_p)
                counter += 1
                idx_p_final.append(idx_temp)

        transMat_sub = np.zeros((counter,counter),dtype=object)

        # EXPERIMENT
        for ii,ipf in enumerate(idx_p_final):
            if ii < len(idx_p_final)-1:
                # loop skip transition
                if idx_p_final[ii+1][0] == idx_p_final[ii][0]+1:
                    transMat_sub[idx_p_final[ii][0]][idx_p_final[ii+1][0]] = self.probNextTrans
                else:
                    transMat_sub[idx_p_final[ii][0]][idx_p_final[ii+1][0]] = self.transitionVaryingProbLoopSkipTrans

            for jj,ipf_loop in enumerate(ipf):
                # self transition
                if ii == len(idx_p_final)-1 and jj == 0:
                    # if ipf is the last loop and jj is the first node in this loop
                    transMat_sub[ipf_loop][ipf_loop] = self.transitionVaryingProbLastLoopSelfTrans
                else:
                    transMat_sub[ipf_loop][ipf_loop] = self.probSelfTrans

            if len(ipf)==2: # loop 2

                # EXPERIMENT
                # enter loop transition depends on the loop position in the network
                if ii < len(idx_p_final)-1:
                    transMat_sub[ipf[0]][ipf[1]] = self.transitionVaryingProbLoopEnterTrans
                else:
                    transMat_sub[ipf[0]][ipf[1]] = self.transitionVaryingProbLastLoopEnterTrans#self.probNextTrans

                transMat_sub[ipf[1]][ipf[0]] = self.probNextTrans
            if len(ipf)==3: # loop 3

                # EXPERIMENT
                # enter loop transition depends on the loop position in the network
                if ii < len(idx_p_final)-1:
                    transMat_sub[ipf[0]][ipf[1]] = self.transitionVaryingProbLoopEnterTrans
                else:
                    transMat_sub[ipf[0]][ipf[1]] = self.transitionVaryingProbLastLoopEnterTrans#self.probNextTrans

                transMat_sub[ipf[1]][ipf[2]] = self.probNextTrans
                transMat_sub[ipf[2]][ipf[0]] = self.probNextTrans
            if len(idx_p_final[-1]) == 1:
                transMat_sub[-1][-1] = self.probSelfTrans   # the network last diagonal self-transition is 0

        return p_final_noLoop, idx_p_final, transMat_sub

    def checkPinyin(self):
        '''
        check self.pinyin
        :return:
        '''
        if self.pinyin in self.pho_entire_loopFold:
            py = self.pinyin
            return py
        else:
            initial_py = dic_pinyin_2_initial_final_map[self.pinyin]['initial']
            final_py   = dic_pinyin_2_initial_final_map[self.pinyin]['final']
            initials_available = []
            for key in self.pho_entire_loopFold.keys():
                final_key = dic_pinyin_2_initial_final_map[key]['final']
                if final_key == final_py:
                    initial_key = dic_pinyin_2_initial_final_map[key]['initial']
                    initials_available.append(initial_key)
            if len(initials_available) > 1:
                # search for same class initials
                for ia in initials_available:
                    for key_ic in initials_class:
                        if ia in key_ic and initial_py in key_ic:
                            py = ia+final_py
                            return py
                py = initials_available[0]+final_py
                return py

            elif len(initials_available) == 1:
                py = initials_available[0]+final_py
                return py
            else:
                raise KeyError('pinyin with the final '+final_py+' not exists ...')


    def createPhosFinal(self):
        '''
        finals phoneme states
        :return:
            idx_final_head: index of the first phoneme of the finals
            idx_final_tail: index of the last phoneme of the finals
        '''

        phos_final_noLoop = []
        transMat_final    = []
        counter = 0
        self.py = self.checkPinyin()

        print 'checked pinyin',self.py

        for ii, p_final in enumerate(self.pho_entire_loopFold[self.py]):
            p_final_noLoop,idx_p_final,transMat_sub = self.assignFinalTransProb(p_final)
            phos_final_noLoop.append(tuple(p_final_noLoop))
            transMat_final.append(transMat_sub)

            self.occurrence_probas_head.append(self.pho_entire_loopFold[self.py][p_final])
            self.idx_final_tail.append(counter + idx_p_final[-1][0])
            counter += len(p_final_noLoop)

        self.phos_final = sum(tuple(phos_final_noLoop),())
        n               = len(self.phos_final)
        self.A_final    = np.zeros((n,n),dtype=object)
        counter         = 0
        for transMat_sub in transMat_final:
            self.idx_final_head.append(counter)
            self.A_final[counter:counter+transMat_sub.shape[0],counter:counter+transMat_sub.shape[0]] = transMat_sub
            counter += transMat_sub.shape[0]

        self.A = self.A_final

        return


    def getStates(self):
        self.states = self.phos_final
        return self.states

    def getPhoFinals(self):
        return self.phos_final

    def getIndexFinalHead(self):
        return self.idx_final_head

    def getIndexFinalTail(self):
        return self.idx_final_tail

    def getOccurrenceProbasHead(self):
        return self.occurrence_probas_head

    def build(self):

        self.createPhosFinal()

        return self.A

    def pathLoopHelperHelper(self,path_unique,dict_loopTimes,loopLength=4):
        '''
        helper function to find the unique loop pairs:lps and their occurrence
        :param path_unique:
        :param dict_loopTimes:
        :param loopLength:
        :return:
        '''

        # find the unique loop pairs
        lps         = []
        lps_unique  = []
        for ii in range(len(path_unique)-loopLength+1):
            if path_unique[ii] == path_unique[ii+loopLength-1] and path_unique[ii] < path_unique[ii+loopLength-2]:
                lps.append(path_unique[ii:ii+loopLength])
                if path_unique[ii:ii+loopLength] not in lps_unique:
                    lps_unique.append(path_unique[ii:ii+loopLength])

        # count the repetition
        if len(lps):
            for lp_u in lps_unique:
                counter = 0
                for lp in lps:
                    if lp == lp_u:
                        counter += 1
                dict_loopTimes[tuple(lp_u)] = counter

    def pathLoopHelper(self,path):
        '''
        identify the loop index and loop time in path
        :param path:
        :return:
        '''

        path_unique = []
        for p in path:
            if not len(path_unique) or p != path_unique[-1]:
                path_unique.append(p)
        print path_unique

        dict_loopTimes = {}
        self.pathLoopHelperHelper(path_unique,dict_loopTimes,loopLength=4)
        self.pathLoopHelperHelper(path_unique,dict_loopTimes,loopLength=3)

        return dict_loopTimes

    def plotNetwork(self,path=[]):
        '''
        plot hmm topology network
        :return:
        '''
        # get all the states
        self.getStates()

        graph = pydot.Dot(graph_type='digraph')
        nodes = []

        if len(path):
            dict_loopTimes = self.pathLoopHelper(path)

        # add nodes
        # kk = 0
        # number2rm = [1,2,6,9,10,11]
        for ii,state in enumerate(self.states):
            if ii in path:
                nodes.append(pydot.Node(str(ii)+dic_pho_map[state],style='filled',fillcolor="green"))
            else:
                # nodes.append(pydot.Node(str(ii)+dic_pho_map[state]))
                nodes.append(pydot.Node(str(ii)+' '+dic_pho_map_topo[state]))
            # if ii not in number2rm:
            graph.add_node(nodes[ii])
            # kk = ii

        # nodes for start and end for ying qmLonUpfLaosheng
        # nodes.append(pydot.Node('start'))
        # nodes.append(pydot.Node('end'))
        # graph.add_node(nodes[kk+1])
        # graph.add_node(nodes[kk+2])

        # add edges
        for ii in range(self.A.shape[0]):
            for jj in range(self.A.shape[1]):
                if ii != jj: #and ii not in number2rm and jj not in number2rm:
                    if not isinstance(self.A[ii][jj],int):
                        if isinstance(self.A[ii][jj],float) and self.A[ii][jj] != 0.0 or isinstance(self.A[ii][jj],np.ndarray):
                            # graph.add_edge(pydot.Edge(nodes[ii],nodes[jj],label='%.3f' % self.A[ii][jj]))
                            # add loop time (occurrence) if looping
                            if len(path) and len(dict_loopTimes):
                                for lp in dict_loopTimes.keys():
                                    if ii == lp[0] and jj == lp[1]:
                                        graph.add_edge(pydot.Edge(nodes[ii],nodes[jj],label='%d' % dict_loopTimes[lp]))
                                    else:
                                        graph.add_edge(pydot.Edge(nodes[ii],nodes[jj]))
                            else:
                                graph.add_edge(pydot.Edge(nodes[ii],nodes[jj]))

        # edge for start and end for ying qmLonUpfLaosheng
        # graph.add_edge(pydot.Edge(nodes[kk+1],nodes[0]))
        # graph.add_edge(pydot.Edge(nodes[2],nodes[kk+2]))
        # graph.add_edge(pydot.Edge(nodes[5],nodes[kk+2]))
        # graph.add_edge(pydot.Edge(nodes[6],nodes[kk+2]))
        # graph.add_edge(pydot.Edge(nodes[7],nodes[kk+2]))
        # graph.add_edge(pydot.Edge(nodes[11],nodes[kk+2]))



        graph.write(path=self.py,prog='/usr/local/bin/dot',format='png')

if __name__ == '__main__':
    net = MakeParallelLRHSMMNet('wo')
    net.build()
    net.plotNetwork()

    # np.savetxt("Output.txt",net.build(),delimiter=' ',fmt='%.3f')

