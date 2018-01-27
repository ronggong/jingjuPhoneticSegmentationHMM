from os.path import dirname,join

root_path    = join(dirname(__file__),'..')

# dataset         = 'qmLonUpfLaosheng'
dataset         = 'danAll'
xgboost_model   = 'xgb_old'
# xgboost_model   = 'xgb_new'

am                 = 'keras'
dnn_node           = '2_512D05M5MB80DO5_plus_validation_300'
keras_cfg          = 'choi_danAll_mfccBands_2D_all_optim'

if dataset == 'sourceSeparation':
    base_path = 'sourceSeparation'
    syllableTierName = 'pinyin'
elif dataset == 'danAll':
    base_path = 'danAll'
    syllableTierName = 'dian'
elif dataset == 'qmLonUpfLaosheng':
    base_path = 'qmLonUpf/laosheng'
    syllableTierName = 'dian'
phonemeTierName = 'details'

dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/'

wav_path        = join(dataset_path,'wav',base_path)
textgrid_path   = join(dataset_path,'textgrid',base_path)

gmmModel_path           = join(root_path,'gmmModel',base_path)

dnnModels_base_path     = join(root_path, 'dnnModel')
dnnModels_cfg_path      = join(root_path, 'dnnModel', base_path,
                          'dnn_phonemeSeg_layers_' + dnn_node + '.cfg')
dnnModels_param_path    = join(root_path, 'dnnModel', base_path,
                            'dnn_phonemeSeg_layers_' + dnn_node + '.param')

scaler_path             = join(root_path, 'dnnModel', base_path,
                               'scaler_'+dataset+'_phonemeSeg_mfccBands2D.pkl')

kerasModels_path        = join(root_path, 'dnnModel', base_path,
                               'keras.cnn_' + keras_cfg + '.h5')

xgbModels_path          = join(root_path,'xgbModel',base_path,
                               'xgb_model_lr1_nesti450_md9.pkl')

svmPatternModel_path    = join(root_path,'svmPatternModel',base_path)

transPriorInfo_path     = join(root_path,'transPriorInfo')

mixModel_path       = '/Users/gong/Documents/pycharmProjects/jingjuAcousticModelExp/gmmModel/mixModels'
mixAdaptModel_path  = '/Users/gong/Documents/pycharmProjects/jingjuAcousticModelExp/gmmModel/mixAdaptModels'
homemadeModel_path  = '/Users/gong/Documents/pycharmProjects/jingjuAcousticModelExp/gmmModel/homemadeModels'

##-- phoneme duration distribution json path
path_dict_dur_dist  = join(root_path, 'trainingData','dict_dur_dist_'+dataset+'.json')

##-- other parameters

fs = 44100
framesize_t = 0.025     # in second
hopsize_t   = 0.010

framesize   = int(round(framesize_t*fs))
hopsize     = int(round(hopsize_t*fs))

# MFCC params
highFrequencyBound = fs/2 if fs/2<11000 else 11000

varin                = {}
varin['N_feature']   = 40
varin['N_pattern']   = 21                # adjust this param, l in paper

# mfccBands feature half context window length
varin['nlen']        = 10
