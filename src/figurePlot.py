__author__ = 'gong'

from src.parameters import *
import matplotlib.pyplot as plt

def plotAudio(audio,startTs, endTs):
    plt.figure()
    startP  = startTs*fs
    endP    = endTs*fs

    plt.plot(audio[startP:endP])
    plt.show()

def pltBoundaryPattern(pattern):
    plt.figure()
    plt.pcolormesh(pattern)
    plt.axis('tight')
    plt.show()