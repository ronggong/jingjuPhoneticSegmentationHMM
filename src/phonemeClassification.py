from phonemeMap import dic_pho_label_inv,dic_pho_label
from parameters import dnnModels_base_path,dnnModels_cfg_path,dnnModels_param_path,xgbModels_path
from sklearn.externals import joblib
import pickle
import os
import numpy as np
import cPickle
import gzip


class PhonemeClassification(object):

    def __init__(self):
        self.gmmModel = {}
        self.n = 0
        self.precision = np.double

    def create_gmm(self, model_path):
        """
        load gmmModel
        :return:
        """
        for state in dic_pho_label:
            pkl_file = open(os.path.join(model_path, state + '.pkl'), 'rb')
            self.gmmModel[state] = pickle.load(pkl_file)
            pkl_file.close()
        self.n = len(self.gmmModel)

    def mapb_gmm(self, observations):
        """
        observation probability
        :param observations:
        :return:
        """
        dim_t       = observations.shape[0]
        self.B_map_gmm  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_gmm[dic_pho_label[state],:] = self.gmmModel[state].score_samples(observations)

    def mapb_gmm_getter(self):
        if len(self.B_map_gmm):
            return self.B_map_gmm
        else: return

    def mapb_dnn(self, observations):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''

        ##-- save the feature to pickle
        label = np.array([0] * len(observations))
        filename_feature_temp = os.path.join(dnnModels_base_path, 'feature_temp.pickle.gz')
        filename_observation_temp = os.path.join(dnnModels_base_path, 'observation_temp.pickle.gz')

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
        self.B_map_dnn  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_dnn[dic_pho_label[state],:] = obs[:, dic_pho_label[state]]

    def mapb_keras(self, observations):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''
        ##-- set environment of the pdnn

        from keras.models import load_model
        from parameters import kerasModels_path

        model = load_model(kerasModels_path)

        ##-- call pdnn to calculate the observation from the features
        # observations = [observations, observations, observations, observations, observations, observations]
        obs = model.predict_proba(observations, batch_size=128)


        ##-- read the observation from the output

        obs = np.log(obs)

        # print obs.shape, observations.shape

        dim_t       = obs.shape[0]
        self.B_map_keras  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_keras[dic_pho_label[state],:] = obs[:, dic_pho_label[state]]

    def mapb_dnn_getter(self):
        if len(self.B_map_dnn):
            return self.B_map_dnn
        else: return

    def mapb_keras_getter(self):
        if len(self.B_map_keras):
            return self.B_map_keras
        else: return


    def mapb_xgb(self, observations):
        '''
        xgboost observation probability
        :param observations:
        :return:
        '''
        clf = joblib.load(xgbModels_path)
        obs = clf.predict_proba(observations)

        # print obs.shape

        obs = np.log(obs)

        # print obs.shape, observations.shape

        dim_t       = observations.shape[0]
        self.B_map_xgb  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_xgb[dic_pho_label[state],:] = obs[:, dic_pho_label[state]]

    def mapb_xgb_getter(self):
        if len(self.B_map_xgb):
            return self.B_map_xgb
        else: return


    def prediction(self,obs):
        """
        find index of the max value in axis=1
        :param obs:
        :return:
        """
        y_pred = np.argmax(obs,axis=0)
        return y_pred