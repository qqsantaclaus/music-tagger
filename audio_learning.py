#!/usr/bin/python
# -*- coding: utf-8 -*-
# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
from compact_cnn import models as my_models
from keras import backend as K
import pdb
import numpy as np
import os
import librosa
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from shutil import copyfile
import pickle

class AudioLearner:
    def __init__(self):
        self.rootdir = "./music/songs/"
        self.tags = ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 
'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 
'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock',
'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening',
'sexy', 'catchy', 'funk', 'electro' ,'heavy metal', 'Progressive rock',
'60s', 'rnb', 'indie pop', 'sad', 'House', 'happy']
    def build_model(self, mode, conv_until=None):
        # setup stuff to build model

        # This is it. use melgram, up to 6000 (SR is assumed to be 12000, see model.py),
        # do decibel scaling
        assert mode in ('feature', 'tagger')
        if mode == 'feature':
            last_layer = False
        else:
            last_layer = True

        if conv_until is None:
            conv_until = 4

        assert K.image_dim_ordering() == 'th', ('image_dim_ordering should be "th". ' +
                                                'open ~/.keras/keras.json to change it.')

        args = Namespace(tf_type='melgram',  # which time-frequency to use
                        normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                        n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                        conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
        # set in [0, 1, 2, 3, 4] if feature extracting.

        model = my_models.build_convnet_model(args=args, last_layer=last_layer)
        model.load_weights('compact_cnn/weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                        by_name=True)
        # model.layers[1].summary()
        # model.summary()
        return model

    def prepare_audio(self, file_loc):
        src, sr = librosa.load(file_loc, sr=12000, mono=True)
        if (float(len(src))/float(sr))<60.0:
            print file_loc, float(len(src))/float(sr)
        # now src: (N, ) and sr: 12000.
        len_seconds = 29.
        pieces = int(float(len(src))/(len_seconds*sr))
        acc= 0
        res = []
        while acc<pieces:
            cut = src[:int(sr*len_seconds)]
            src = src[int(sr*len_seconds):]
            cut = cut[np.newaxis, np.newaxis, :]
            res.append(cut)
        # print cut.shape
            acc=acc+1
        # the src might be shorter than 29*12000 if the original signal is shorter than that.
        # print res.shape
        return res

    def feature(self, load = False):
        if not load:
            models = []
            # # load models that predict features from different level
            model4 = self.build_model('feature')  # equal to build_model('feature', 4), highest-level feature extraction
            model3 = self.build_model('feature', 3)  # low-level feature extraction.
            model2 = self.build_model('feature', 2)  # lower-level..
            model1 = self.build_model('feature', 1)  # lowerer...
            model0 = self.build_model('feature', 0)  # lowererer.. no, lowest level feature extraction.

            # # prepare the models
            models.append(model4)
            models.append(model3)
            models.append(model2)
            models.append(model1)
            models.append(model0)

            points = []
            ids = []
            for subdir, dirs, files in os.walk(self.rootdir):
                for file in files:
                    #print os.path.join(subdir, file)
                    filepath = subdir + os.sep + file
                    # source for example
                    # src = np.load('1100103.clip.npy')  # (348000, )
                    # src = src[np.newaxis, :]  # (1, 348000)
                    # src = np.array([src]) # (1, 1, 348000) to make it batch
                    if filepath.endswith(".mp3"):
                        src = self.prepare_audio(filepath)
                        #
                        print file[:-4], len(src)
                        feats = np.zeros(160)
                        weights = 0
                        for i in range(len(src)):
                            feat = [md.predict(src[i])[0] for md in models] # get 5 features, each is 32-dim
                            feat = np.array(feat).reshape(-1) # (160, ) (flatten)
                            feats += ((i+1)*(len(src)-i)) * feat
                            weights += ((i+1)*(len(src)-i))
                        if (weights > 0):
                            feats /= float(weights)
                            ids.append(file[:-4])
                            points.append(feats)
                        # now use this feature for whatever MIR tasks.
            X = np.array(points)
            with open("./Xarr.p", "w") as tempstore:
                pickle.dump(X, tempstore)
            with open("./Xid.p", "w") as tempstore:
                pickle.dump(ids, tempstore)
        else:
            with open("./Xarr.p", "r") as tempstore:
                X = pickle.load(tempstore)
            with open("./Xid.p", "r") as tempstore:
                ids = pickle.load(tempstore)
        pca = PCA(n_components=32)
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        new_X = pca.fit_transform(X)
        with open("./audio_features.p", "w") as tempstore:
            pickle.dump(zip(ids, new_X), tempstore)
        return zip(ids, new_X)
        # num_clusters = 4
        # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(new_X)
        # for i in range(num_clusters):
        #     if not os.path.exists("./clusters/"+str(i)):
        #         os.makedirs("./clusters/"+str(i))
        # for i in range(len(ids)):
        #     copyfile(self.rootdir+ids[i]+"/"+ids[i]+".mp3", "./clusters/"+str(kmeans.labels_[i])+"/"+ids[i]+".mp3")

    def tag(self):
        models = []
        models.append(self.build_model('tagger')) # music tagger

        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                #print os.path.join(subdir, file)
                filepath = subdir + os.sep + file
        
                # source for example
                if filepath.endswith(".mp3"):
                    src = self.prepare_audio(filepath)
                    #
                    feats = np.zeros(50)
                    weights = 0
                    for i in range(len(src)):
                        feat = [md.predict(src[i])[0] for md in models] # get 5 features, each is 32-dim
                        feat = np.array(feat).reshape(-1) # (160, ) (flatten)
                        feats += ((i+1)*(len(src)-i)) * feat
                        weights += ((i+1)*(len(src)-i))
                    feats /= float(weights)
                    # now use this feature for whatever MIR tasks.
                    result = zip(self.tags, feats)
                    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
                    with open(subdir + os.sep + "scores.json", "w") as outputfile:
                        json.dump([{"tag": name, "score": '%5.3f' % score} for name, score in sorted_result], outputfile, indent=4)
