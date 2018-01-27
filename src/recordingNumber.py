__author__ = 'gong'

from src.trainTestSeparation import getRecordings,getRecordingNames
from src.parameters import *

recordings          = getRecordings(textgrid_path)
recordings_train    = getRecordingNames('TRAIN',dataset)
recordings_test     = getRecordingNames('TEST',dataset)

number_train        = [recordings.index(r) for r in recordings_train]
number_test         = [recordings.index(r) for r in recordings_test]