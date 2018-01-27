import csv
from demoParallelLRHMM import doDemo
import numpy as np
from src.parameters import dataset,xgboost_model,am


for valid_string in ['valid','idxresort']:
    export=open("eval/"+dataset+'_'+xgboost_model+'_'+valid_string+'_hsmm'+'_'+am+'.csv', "wb")
    writer=csv.writer(export, delimiter=',')

    ##-- VTP false, BPC false
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=False,
                                                        valid_string=valid_string,
                                                        explicit_dur=True,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    ##-- VTP false, BPC svm
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=True,
                                                        valid_string=valid_string,
                                                        explicit_dur=True,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    ##-- VTP false, BPC xgb
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=True,
                                                        model_classification='xgb',
                                                        valid_string=valid_string,
                                                        explicit_dur=True,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    export.close()
