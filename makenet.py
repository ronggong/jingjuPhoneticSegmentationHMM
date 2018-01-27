from src.pinyinMap import *
from src.phonemeMap import *
import numpy as np

class MakeNet(object):


    def __init__(self,pinyin,precision=np.double):
        self.pinyin = pinyin
        self.A = []
        self.initial = dic_pinyin_2_initial_final_map[pinyin]['initial']
        self.pho_initial = dic_pho_map[dic_initial_2_sampa[self.initial]]

        self.final   = dic_pinyin_2_initial_final_map[pinyin]['final']  # in pinyin

        self.probNextTrans  = 0.1
        self.probSelfTrans  = 1-self.probNextTrans

        self.transitionWeight = 0.2

        self.precision      = precision

    def assignInitalTransProb(self):
        pass
    def assignSelfTransProb(self):
        pass
    def getStates(self):
        pass

    def build(self):
        pass
    def plotNetwork(self):
        pass