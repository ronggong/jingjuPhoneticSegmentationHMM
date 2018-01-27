import csv
from demoParallelLRHMM import doDemo
import numpy as np
from src.parameters import dataset,xgboost_model,am,dnn_node


for valid_string in ['valid','idxresort']:
    export=open("eval/"+dataset+'_'+xgboost_model+'_'+valid_string+'_'+am+'.csv', "wb")
    writer=csv.writer(export, delimiter=',')

    ##-- VTP false, BPC false
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=False,
                                                 valid_string=valid_string,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])


    ##-- VTP false, BPC svm
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=True,
                                                    valid_string=valid_string,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    ##-- VTP false, BPC xgb
    HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=False,
                                                        patternMethod=True,
                                                        model_classification='xgb',
                                                    valid_string=valid_string,
                                                        plot=False)

    writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])
    """
    ##-- VTP true, BPC false
    for ep in np.arange(0.0,4.0,1.0):
        if ep == 0:
            ep = 0
        else:
            ep = 10**ep

        print 'enter penalize:',ep

        HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=True,
                                                        enterPenalize=ep,
                                                        patternMethod=False,
                                                        valid_string=valid_string,
                                                        plot=False)

        writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    ##-- VTP true, BPC svm
    for ep in np.arange(0.0,4.0,1.0):
        if ep == 0:
            ep = 0
        else:
            ep = 10**ep

        print 'enter penalize:',ep

        HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=True,
                                                        enterPenalize=ep,
                                                        patternMethod=True,
                                                        valid_string=valid_string,
                                                        plot=False)

        writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])

    ##-- VTP true, BPC xgb
    for ep in np.arange(0.0,4.0,1.0):
        if ep == 0:
            ep = 0
        else:
            ep = 10**ep

        print 'enter penalize:',ep

        HR, OS, FAR, F, R, deletion, insertion = doDemo(pho_duration_threshold=0.0,
                                                        tol=0.04,
                                                        bool_transVaryProb=True,
                                                        enterPenalize=ep,
                                                        patternMethod=True,
                                                        model_classification='xgb',
                                                        valid_string=valid_string,
                                                        plot=False)
        writer.writerow([HR*100, OS*100, FAR*100, F*100, R*100, deletion, insertion])
    """
    export.close()
