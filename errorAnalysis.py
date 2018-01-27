__author__ = 'gong'
import json
import operator
from src.parameters import *

numTrans_phoneme_gt   = json.load(open('dict_numTrans_phoneme_'+dataset+'_gt.json','r'))
numTrans_phoneme      = json.load(open('dict_numTrans_phoneme_'+dataset+'.json','r'))

dict_numTrans_phoneme_gt = {}
dict_numTrans_phoneme = {}

for ntpsgt in numTrans_phoneme_gt:
    dict_numTrans_phoneme_gt[ntpsgt[0]] = ntpsgt[1]

for ntps in numTrans_phoneme:
    dict_numTrans_phoneme[ntps[0]] = ntps[1]

print sum(dict_numTrans_phoneme_gt.values())
print sum(dict_numTrans_phoneme.values())

dict_error_rate = {}
for tp in dict_numTrans_phoneme_gt.keys():
    dict_error_rate[tp] = (dict_numTrans_phoneme_gt[tp]-dict_numTrans_phoneme[tp])/float(dict_numTrans_phoneme_gt[tp])

sorted_error_rate = sorted(dict_error_rate.items(), key=operator.itemgetter(1))[::-1]

for ser in sorted_error_rate:
    print ser[0],dict_numTrans_phoneme_gt[ser[0]],dict_numTrans_phoneme_gt[ser[0]]-dict_numTrans_phoneme[ser[0]],ser[1]
