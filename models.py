# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:35:00 2019

@author: CUI
"""
import os
import gc
import datetime
import numpy as np
import pandas as pd
import random
import itertools
import zipfile
import io
import time

from copy import deepcopy
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import seaborn as sns

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
#from keras_tqdm import TQDMCallback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import NullFormatter
from skimage.metrics import structural_similarity as ssim

from libs.pconv_model import PConvUnet
from libs.lstm_model import LSTM
from libs.lstmsimple_model import LSTM as LSTMS
from libs.gru_model import GRU
from libs.grusimple_model import GRU as GRUS
from libs.nbeats_model import NBeats

from dataset.data_process import kinter, forplot_assignement_accuracy, kcentroids_equal, to_kcentroid_seq, chunkdata_for_longpredict, retrieve_traintimeseq, create_labelines_timeseq_dataset, convertdata_for_training, no_cosmic, rescale_data_by_seqs
from featuring.brandon_features import feature_transform, Mg_settings
from sklearn.metrics import confusion_matrix#, ConfusionMatrixDisplay

class_parms = None
from featuring.mts_metrics import NPMtsMetrics
from featuring.class_n2_metrics import tss_hss_all
try:
    from main_classify import Settings, update_settings_fromclass
    from models_classify import SP_Conv_Dense, create_class_mask
    from featuring.center_stat import NPJointCenterStat#, NPCenterStat, CenterStat
    from libs.class_pconv_model import NP_CategoricalCrossentropy, NP_BinaryCrossentropy, NP_CategoricalAccuracy, NP_BinaryAccuracy
    from libs.countdict3k_acc import NPAccuracyOverTime3D
except:
    print("Could not import libraries on centers and classification")
else:
    print("successfuly imported libraries on centers and classification")

now = datetime.datetime.now
plt.rcParams.update({'font.size': 44})
plt.rcParams.update({'font.family': 'Cambria'})
manual_mode = True
change_traindata = True # whether to save/overwrite data_longformat.npz

def saveCompressed(fh, allow_pickle, **namedict):
     with zipfile.ZipFile(fh,
                          mode="w",
                          compression=zipfile.ZIP_DEFLATED,
                          allowZip64=True) as zf:
         for k, v in namedict.items():
             buf = io.BytesIO()
             np.lib.npyio.format.write_array(buf,
                                             np.asanyarray(v),
                                             allow_pickle=allow_pickle)
             zf.writestr(k + '.npy',
                         buf.getvalue())

class SP_PCUNet(object):
    def __init__(self, config, classes_and_inclusions_addnoclass=None,
                 feat_legends=None):
        if 'n_blocks' in dir(config):
            self.n_blocks = config.n_blocks
        self.with_features = config.with_features
        self.debug = config.debug
        self.given_tvt = config.given_tvt
        self.model_type = config.model_type
        self.with_centerloss = config.with_centerloss
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.batch_norm = config.batch_norm
        self.learning_rate_BN = config.learning_rate_BN
        self.learning_rate_FINE = config.learning_rate_FINE
        self.dataset = config.dataset
        self.dataset_address = config.dataset_address
        self.train_ratio = config.train_ratio
        self.test_ratio = config.test_ratio
        self.label_length = config.label_length
        self.mask_ratio = config.mask_ratio
        self.random_ratio = config.random_ratio
        self.labels = config.labels
        self.labels = self.labels.split('_')
        self.nolabel = config.nolabel
        self.noclass = config.noclass
        self.test_labels = config.test_labels
        self.test_labels = self.test_labels.split('_')
        self.name = config.name
        self.checkpoint_dir = config.checkpoint_dir
        self.logs_dir = config.logs_dir
        self.results_dir = config.results_dir
        self.train1 = config.train1
        self.train2 = config.train2
        self.preload_train = config.preload_train
        self.testload_FINE = config.testload_FINE
        self.test = config.test
        self.test_ds = config.test_ds
        self.test_ds = self.test_ds.split('_')
        if 'TE' in self.test_ds:
            self.test_ds = self.test_ds + ['TE_%s'%ds for ds in self.test_labels]
        if 'TEL' in self.test_ds:
            self.test_ds = self.test_ds + ['TEL_%s'%ds for ds in self.test_labels]
        self.add_classifier = config.add_classifier
        self.classes = config.classes
        self.classes_and_inclusions_addnoclass = classes_and_inclusions_addnoclass
        self.class_inclusions = config.class_inclusions
        self.add_centercount = config.add_centercount
        self.predict = config.predict
        self.predict_ds = config.predict_ds
        self.number_predict = config.number_predict
        self.show_res = config.show_res
        self.show_dist_polar = config.show_dist_polar
        self.backg_color = config.backg_color
        self.fig_form = config.fig_form
        self.frame_res = config.frame_res
        self.coef_diff = 5
        # self.feat_legends = config.feat_legends
        self.feat_legends = feat_legends
        self.all_labels = list(set(self.labels+self.test_labels))

        # labels = self.labels.split('_')
        # test_labels = self.test_labels.split('_')
        
        print('[%s - START] Creating the datasets..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
        if self.dataset in ['iris_level_2C','iris_level_2B','al_2C','ld_2C','pb_2C']:
            self.c_dim = 1
        if self.train1 and not(self.preload_train) and not(manual_mode):
            data, positions = create_labelines_timeseq_dataset(self.dataset_address, self.labels, self.label_length)
            if config.cosmic_to_mean:   # V2
                data = no_cosmic(data, config.cosmic_t)
#            mx = find_max_data(data)
#            if mx != 1: # V2
#                data = rescale_data(data, mx)
            data = rescale_data_by_seqs(data)
            self.data_pack = list(zip(data, positions)) # [(data, position)]
            # partition by activity {act_lab: list of data}
            activities = self.labels
            self.data_pack = {a: [e for e in self.data_pack if any([a in str(ee) for ee in e[1]])] for a in activities} # {act: [(data, pos)]}
            if self.given_tvt:
                self.data_pack = {dt: {a: [e for e in self.data_pack[a] if any([['train', 'valid', 'test'][['TR','VAL','TE'].index(dt)] in str(ee) for ee in e[1]])] for a in self.data_pack} for dt in ['TR','VAL','TE']}# {tvt: {act: [(data, pos), ...]}}
            else:
                percentages = {a: {f: sum([e[0].shape[0] for e in self.data_pack[a] if f in e[1]]) for f in set([e[1][[a in str(ee) for ee in e[1]].index(True)] for e in self.data_pack[a]])} for a in self.data_pack}
                percentages = {a: {f: percentages[a][f]/sum(percentages[a][ff] for ff in percentages[a]) for f in percentages[a]} for a in percentages}
                for a in percentages:
                    events = list(percentages[a].keys())
                    shuffle = True
                    attempts = 0
                    max_attempts = 100
                    while shuffle and attempts<max_attempts:
                        random.shuffle(events)
                        attempts += 1
                        cumul_left = self.loop_cumul(percentages[a], events, self.train_ratio, attempts==max_attempts-1)
                        cumul_right = self.loop_cumul(percentages[a], events, self.test_ratio, attempts==max_attempts-1, start='right')
                        if cumul_left is None or cumul_right is None:
                            continue
                        if cumul_right+cumul_left>len(events):
                            cumul_left -= (cumul_right+cumul_left)-len(events)
                        assert cumul_left>0, "problem with the partition, train and test overlap, coding issue or too small amount of events"
                        train_events = events[:cumul_left]
                        test_events = events[::-1][:cumul_right]
                        if cumul_left+cumul_right<len(events):
                            valid_events = [e for e in events if e not in train_events+test_events]
                            assert len(valid_events)!=0, "problem during the partition, see code"
                        else:
                            print("Label %s : No separate event for validation dataset, will use an overlapping part of the training data"%a)
                            valid_ratio = 1-self.train_ratio-self.test_ratio
                            valid_events = [e for e in train_events if percentages[a][e]<valid_ratio]
                            if len(valid_events)!=0:
                                random.shuffle(valid_events)
                                cumul_left = self.loop_cumul(percentages[a], valid_events, valid_ratio, False)
                                valid_events = valid_events[:cumul_left]
                            else:
                                valid_events = [train_events[[percentages[a][e] for e in train_events].index(min([percentages[a][e] for e in train_events]))]]
                        self.data_pack[a] = {
                            'TR': [e for e in self.data_pack[a] if any(ee in e[1] for ee in train_events)],
                            'VAL': [e for e in self.data_pack[a] if any(ee in e[1] for ee in valid_events)],
                            'TE': [e for e in self.data_pack[a] if any(ee in e[1] for ee in test_events)]}# {act: {tvt: [(data, pos)]}}
                        shuffle = False
                    
                    assert all(len(lev)>0 for lev in [train_events, test_events, valid_events]) and len([e for e in train_events if e in test_events])==0, "Did not find how to partition data"
                    del cumul_right, cumul_left, train_events, test_events, valid_events
                self.data_pack = {dt: {a: self.data_pack[a][dt] for a in self.data_pack} for dt in ['TR','VAL','TE']}# {tvt: {act: [(data, pos), ...]}}
            self.data_pack['TE'] = {
                **(self.data_pack['TE']),
                **{a: list(zip(*create_labelines_timeseq_dataset(self.dataset_address, [a], self.label_length))) for a in self.test_labels if a not in self.labels}}
            self.data_pack = {
                **{'_'.join([dt,a]): self.data_pack[dt][a] for dt in self.data_pack for a in self.data_pack[dt]},
                'TR': list(itertools.chain(*(self.data_pack['TR'].values()))),
                'VAL': list(itertools.chain(*(self.data_pack['VAL'].values()))),
                'TE': list(itertools.chain(*(self.data_pack['TE'].values()))),}# [(data, pos)..]
            self.data_pack = {k: list(zip(*(self.data_pack[k]))) for k in self.data_pack} # {k: [(data,),(position,)]}
        
        elif not(manual_mode):
            print("Loading previous dataset..")
            data_info = np.load(os.path.join(self.checkpoint_dir, 'input_data','data_longformat.npz'), allow_pickle = True)
            keys = [k.replace('data_', '') for k in list(data_info.keys()) if 'data_' in k]
            self.data_pack = {k: (data_info['data_'+k], data_info['position_'+k]) for k in keys}
            data_info.close()
            change_traindata = False
        
        if not(manual_mode):
            self.data_train = {k: convertdata_for_training(list(self.data_pack[k][0]), list(self.data_pack[k][1]), self.label_length, self.mask_ratio) for k in self.data_pack}
            
            self.generators = {
                k: [AugmentingDataGenerator().flow_from_data(
                    np.array(self.data_train[k][0]), 
                    np.array(list(zip(*(self.data_train[k][1])))[0]),
                    self.mask_ratio, 
                    self.random_ratio,
                    batch_size=self.batch_size
                    ), # 1st is data for model
                    len(self.data_train[k][0]), # 2nd is length of data
                    {p:l for p,l in zip(
                        list(zip(*(self.data_train[k][1])))[0],
                        [self.label_from_pos(p) for p in list(
                            zip(*(self.data_train[k][1])))[2]])} # 3rd is dict{pos:lab} info
                    ]for k in self.data_train}
            self.generators = {
                **self.generators,
                'show': [AugmentingDataGenerator().flow_from_data(
                         np.array(self.data_train['TR'][0])[:self.batch_size], 
                         np.array(list(zip(*(self.data_train['TR'][1])))[0])[:self.batch_size],
                         self.mask_ratio, 
                         self.random_ratio,
                         batch_size=self.batch_size
                         ), # 1st is data for model
                         self.batch_size, # 2nd is length of data
                         {p:l for p,l in zip(
                             list(zip(*(self.data_train['TR'][1])))[0],
                             [self.label_from_pos(p) for p in list(
                                 zip(*(self.data_train['TR'][1])))[2]])} # 3rd is dict{pos:lab} info
                         ]}
            self.test_generators = {
                k: [AugmentingDataGenerator().flow_from_data(
                    np.array(self.data_train[k][0]), 
                    np.array(list(zip(*(self.data_train[k][1])))[0]),
                    self.mask_ratio, 
                    self.random_ratio,
                    batch_size=self.batch_size,
                    seed=1
                    ), # 1st is data for model
                    len(self.data_train[k][0]), # 2nd is length of data
                    {p:l for p,l in zip(
                        list(zip(*(self.data_train[k][1])))[0],
                        [self.label_from_pos(p) for p in list(
                            zip(*(self.data_train[k][1])))[2]])} # 3rd is dict{pos:lab} info
                    ]for k in self.data_train}
            
            if not os.path.exists(os.path.join(self.checkpoint_dir, 'input_data')):
                os.makedirs(os.path.join(self.checkpoint_dir, 'input_data'))
            
            # if change_traindata:
            if True:
                saveCompressed(open(os.path.join(self.checkpoint_dir, 'input_data','data_longformat.npz'), 'wb'),
                         **{'data_'+u: v[0] for u,v in self.data_pack.items()},
                         **{'position_'+u: v[1] for u,v in self.data_pack.items()},
                         allow_pickle=True)
        
        # Metrics
        if self.add_classifier:
            if self.classes_and_inclusions_addnoclass is None:
                self.classes_and_inclusions_addnoclass = [
                    (self.classes, self.class_inclusions, self.noclass)]
            self.classifier = []
            counter_classes = {}
            self.classes = None
            self.class_inclusions = None
            clsn_nolab_nocls = lambda u,v,w:'%s_nolabel-%s_noclass-%s'%(u,v,w)
            for classes, class_inclusions, addnoclass in self.classes_and_inclusions_addnoclass:
                class_parms = Settings()
                assert class_parms is not None, "Could not import the classification settings"
                class_parms.skip_data = True
                class_parms.nolabel = self.nolabel
                if addnoclass is not None:
                    class_parms.noclass = addnoclass
                else:
                    class_parms.noclass = self.noclass
                class_parms.train1 = False
                class_parms.train2 = False
                class_parms.inference_only = True
                class_parms.testload_FINE = self.testload_FINE
                class_parms = update_settings_fromclass(class_parms, classes, class_inclusions)
                # skip_data to not load unnecessary data
                self.classifier += [SP_Conv_Dense(class_parms, skip_data=True, 
                                                  c_dim=self.c_dim)]
                # Load model with pretrained parameters
                self.classifier[-1].model_instance(False, True)
                
                key_name = clsn_nolab_nocls(self.classifier[-1].name,
                                            self.nolabel,
                                            self.classifier[-1].noclass)
                counter_classes[key_name] = [c for c in self.classifier[-1].classes]
                if self.classifier[-1].noclass is not None:
                    counter_classes[key_name] += [self.classifier[-1].noclass]
            
            self.classifier = {clsn_nolab_nocls(clsfier.name, self.nolabel, clsfier.noclass): clsfier for clsfier in self.classifier}
        else:
            self.classifier = {'noclassifier': None}
            counter_classes = {'noclassifier': ['NoClass']}
        print('counter_classes', counter_classes)
        assert counter_classes.keys()==self.classifier.keys(), "error in the code"
        
        # usual mts metrics
        self.mts_metrics = {}
        if 'TEL' in self.test_ds:
            self.glob_mts_metrics = {}
        counter_labels = self.all_labels
        if self.nolabel is not None:
            counter_labels += [self.nolabel]
        for ccn, cc in counter_classes.items():
            self.mts_metrics[ccn] = NPMtsMetrics(counter_labels, cc, cc)
            self.mts_metrics[ccn].reset()
            if 'TEL' in self.test_ds:
                self.glob_mts_metrics[ccn] = NPMtsMetrics(counter_labels, cc, cc)
                self.glob_mts_metrics[ccn].reset()
        
        if self.add_classifier:
            self.atc_metric = {}
            for ccn, cc in counter_classes.items():
                self.atc_metric[ccn] = NPAccuracyOverTime3D(counter_labels, cc, cc)
        
        # stat and info metrics on centers
        if self.add_centercount:
            self.center_counter_pio = {}
            if 'TEL' in self.test_ds:
                self.glob_center_counter_pio = {}
            for ccn, cc in counter_classes.items():
                self.center_counter_pio[ccn] = NPJointCenterStat(
                    counter_labels, cc, cc, 
                    *(([str(c) for c in range(53)],)*3))
                self.center_counter_pio[ccn].reset()
                if 'TEL' in self.test_ds:
                    self.glob_center_counter_pio[ccn] = NPJointCenterStat(
                        counter_labels, cc, cc, 
                        *(([str(c) for c in range(53)],)*3))
                    self.glob_center_counter_pio[ccn].reset()
    
    def to_rgb(self, data):
        if self.c_dim == 1:
            # convert to RGB data
            return np.tile(data, [1, 1, 1, 3])
        return data
    
    def cumul_side(self, percent_dict, sorted_list, threshold):
        # returns a boolean list
        return [sum(percent_dict[ee] for ee in sorted_list[:sorted_list.index(e)+1])<threshold for e in sorted_list]
    
    def loop_cumul(self, percent_dict, sorted_list, threshold, max_attempt, start='left'):
        if start=='right':
            sorted_list = sorted_list[::-1]
        if percent_dict[sorted_list[0]]<=threshold:
            return sum(self.cumul_side(percent_dict, sorted_list, threshold))+1
        elif max_attempt:
            return 1
    
    def label_from_pos(self, pos, predictonly=True):
        try:
            return self.all_labels[[l in pos for l in self.all_labels].index(True)]
        except:
            assert predictonly, "No Label can only be used for predictions, not train/eval or test, for pos: {}".format(pos)
            return self.nolabel
    
    def parse_lab_pos(self, pos, i, dict_poslab):
        """Outputs pos as int and vlabel as str
        """
        # tf.print('parse lab pos')
        # labi = tf.strings.split(tf.expand_dims(pos[i],0), '_').values[1]
        posi = int(pos[i])
        labi = dict_poslab[posi]
        return labi, posi
    
    def model_instance(self, train_bn, inference_only):
        
        if inference_only:
            vgg_weights = None
        else:
            vgg_weights = 'D:/ML savings/IRIS_predspectra/data/logs/pytorch_to_keras_vgg16.h5'
        
        # Instantiate the model
        if self.model_type == 'PCUNet':
            self.model = PConvUnet(
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        if self.model_type == 'LSTM':
            self.model = LSTM(
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        if self.model_type == 'LSTMS':
            self.model = LSTMS(
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        if self.model_type == 'GRU':
            self.model = GRU(
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        if self.model_type == 'GRUS':
            self.model = GRUS(
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        if self.model_type == 'NBeats':
            self.model = NBeats(n_blocks=self.n_blocks,
                img_rows=self.label_length, img_cols=self.label_length,
                c_dim=self.c_dim, with_centerloss = self.with_centerloss,
                inference_only=inference_only, 
                vgg_weights=vgg_weights, net_name=self.name) #img_rows=512, img_cols=512, vgg_weights="imagenet", inference_only=False, net_name='default', gpus=1, vgg_device=None
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'trained_model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        if train_bn:
            learning_rate = self.learning_rate_BN
        else:
            learning_rate = self.learning_rate_FINE
        if inference_only:
            print('io')
            if self.testload_FINE:
                ckpt_name = max(glob(os.path.join(
                    checkpoint_dir,'BN'+str(self.learning_rate_BN)+'FINE'+str(self.learning_rate_FINE)+'*')), key = os.path.getctime)
            else:
                ckpt_name = max(glob(os.path.join(
                    checkpoint_dir,'BN'+str(self.learning_rate_BN)+'w*')), key = os.path.getctime)
            if ckpt_name:
                print(" [*] Checkpoint found")
                print(ckpt_name)
                self.model.load(ckpt_name, train_bn=False)
                print(" [*] Model loaded")
            else:
                print(" [*] Failed to find a checkpoint, train a model first")
        elif self.preload_train or not(train_bn): # training from pretrained
            print('pt')
            search = glob(os.path.join(
                checkpoint_dir,'BN'+str(self.learning_rate_BN)+'*'))
            if len(search)!=0:
                ckpt_name = max(search, key = os.path.getctime)
                if ckpt_name:
                    print(" [*] Checkpoint found")
                    print(ckpt_name)
                    self.model.load(ckpt_name, train_bn=train_bn, lr=learning_rate)
                    print(" [*] Model loaded")
                else:
                    print(" [*] Could not find a pretrained model, please retry")
            else:
                print(" [*] Failed to find a checkpoint, will train a model first")
                self.set_model(train_bn=train_bn, lr=learning_rate)
                print(" [*] Model is set")
        else:
            #setting model
            self.set_model(train_bn=train_bn, lr=learning_rate)
            # print('compiled3', self.model.model._is_compiled)
            print(" [*] Model is set")
        
    def plot_callback(self, samples, epoch):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        
        if not os.path.exists(os.path.join(self.results_dir, 'training', 'samples')):
            os.makedirs(os.path.join(self.results_dir, 'training', 'samples'))
        
        # Parse samples
        (masked, mask, pos), ori = samples
        
        # Get samples & Display them        
        pred_img = self.model.predict([masked, mask])
    
        # Clear current output and display test images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:].squeeze().transpose(), cmap = 'gist_heat', vmin = 0, vmax = 1)
            axes[1].imshow(pred_img[i,:,:,:].squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
            axes[2].imshow(ori[i,:,:,:].squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
            axes[0].set_title('Masked Image')
            axes[1].set_title('Predicted Image')
            axes[2].set_title('Original Image')
                    
            self.savefig_autodpi(os.path.join(
                self.results_dir, 'training', 'samples',
                'Epoch_img_{}_{}_{}.png').format(i, epoch, pos[i]),
                bbox_inches='tight')
            plt.close()
    
    def set_model(self, train_bn=True, lr=0.0002):
        # Create UNet-like model
        self.model.build_pconv_unet(train_bn)
        self.model.built = True
        self.model.compile(lr) 
        # print('compiled4', self.model.model._is_compiled)
        # print('compiled4b', self.model._is_compiled)
        print('[*] Model built and compiled')
    
    def train(self, show_samples):
        if not os.path.exists(os.path.join(self.results_dir, 'training')):
            os.makedirs(os.path.join(self.results_dir, 'training'))
        if self.train1:
            self.train_phase(show_samples, True)
        if self.train2:
            self.train_phase(show_samples, False)
    
    def train_phase(self, show_samples, train_bn):
        print('[START] Loading Model for train - train_bn %s - inference_only %s'%(train_bn, False))
        self.model_instance(train_bn, False)
        # self.model.summary()
        # print("test compiled1", self.model.model._is_compiled)
        if train_bn:
            checkpoint_phase = '/BN'+str(self.learning_rate_BN)
        else:
            checkpoint_phase = '/BN'+str(self.learning_rate_BN)+'FINE'+str(self.learning_rate_FINE)
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'trained_model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('[%s - START] Training Phase %i ..'%(now().strftime('%d.%m.%Y - %H:%M:%S'), 2-train_bn))
        # Run training for certain amount of epochs
        # print("test compiled2", self.model.model._is_compiled)
        self.model.fit_generator(
            self.generators['TR'][0], 
            steps_per_epoch=self.generators['TR'][1]//self.batch_size,
            validation_data=self.generators['VAL'][0],
            validation_steps=self.generators['VAL'][1]//self.batch_size,
            epochs=[self.epoch,1][int(self.debug)],  
            verbose=0,
            callbacks=[
                TensorBoard(
                    log_dir=self.logs_dir,
                    write_graph=False
                ),
                ModelCheckpoint(
                    checkpoint_dir+checkpoint_phase+'weights.{epoch:02d}-{loss:.2f}.h5',
                    monitor='val_loss', 
                    save_best_only=False, 
                    save_weights_only=True
                ),
                # LambdaCallback(
                #     on_epoch_end=lambda epoch, logs: self.plot_callback(show_samples, epoch)
                #     # on_epoch_end=lambda epoch, logs: self.plot_callback(self.to_rgb(show_samples), epoch)
                # ),
                #TQDMCallback()
            ]
        )
        print('[%s - END] Training.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def features_feedback(self):
        if os.path.isfile(os.path.join(self.checkpoint_dir, 'input_data', 'feats_fback.npz')):
            feats_fback = np.load(os.path.join(self.checkpoint_dir, 'input_data', 'feats_fback.npz'), allow_pickle=True)['feats_fback']
            if len(feats_fback.shape) == 2 and feats_fback.shape[1] == 2:
                feats_fback = [list(a) for a in feats_fback]
            else:
                feats_fback = []
        else:
            feats_fback = []
        for name, [gen, length, dict_poslab] in self.generators.items():
            if name == 'show':
                continue
            positions_f = self.data_train[name][1]
            n = 0
            print('Checking and writing feedbacks about features on original %s data'%name)
            for (masked, mask, pos), ori in tqdm(gen, total = np.ceil(length / self.batch_size)):
                feats = [feature_transform(o.squeeze()[-int(ori.shape[1] * self.mask_ratio):, :], o.squeeze()[-int(ori.shape[1] * self.mask_ratio):, :]).transpose() for o in ori]
                for posi, feati in zip(pos, feats):
                    if np.isnan(feati).any():
                        idx = np.where(np.array(list(zip(*positions_f))[0])==posi)
                        assert len(idx) ==  1 and idx[0] and idx[0].shape == (1,), "Found no or several options for an NoClass index %s: %s"%(posi, idx)
                        idx = idx[0][0]
                        posi_f = positions_f[idx]
                        feats_fback.append([np.sum(np.isnan(feati)), retrieve_traintimeseq(posi_f)])
                
                n += len(feats)
                
                if n > length:
                    break
        np.savez(os.path.join(
            self.checkpoint_dir, 'input_data',
            'feats_fback'), feats_fback = feats_fback)
    
    def predicts(self):
        if not os.path.exists(os.path.join(self.results_dir, 'prediction')):
            os.makedirs(os.path.join(self.results_dir, 'prediction'))
        if self.model_type in ['LSTM', 'LSTMS', 'GRU', 'GRUS', 'NBeats']:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(True, True))
            self.model_instance(True, True)
        else:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(False, True))
            self.model_instance(False, True)
        print('[%s - START] Predicting..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
        predict_ds = self.predict_ds.split('_')
        for ds in predict_ds:
            if 'TEL' in ds:
                self.long_predicts(ds)
            else:
                gen = self.generators[ds][0]
                print('[Predicting] %s Samples'%ds)
                
                n = 1
                for (masked, mask, pos), ori in tqdm(gen, total = int(np.ceil(self.number_predict/self.batch_size))):
                    
                    n = self.predict_save_1batch(masked, mask, ori, pos, ds, n)
                    
                    # Only create predictions for about 100 images
                    if n >= self.number_predict:
                        break
        print('[%s - END] Predicting.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def predict_save_1batch(self, masked, mask, ori, pos, ds, n):
        pred_img = self.model.predict([masked, mask])
        
        # class prediction on input and output
        class_mask = create_class_mask(ori, self.mask_ratio, 
                                            self.random_ratio)
        if self.add_classifier:
            pred_class_in = {}
            pred_class_out = {}
            in_class = {}
            out_class = {}
            for clsn, clsfier in self.classifier.items():
                pred_class_in[clsn] = clsfier.model.predict([ori, class_mask])
                in_class[clsn] = clsfier.model.np_assign_class(pred_class_in[clsn])
                pred_class_out[clsn] = clsfier.model.predict([pred_img, class_mask])
                out_class[clsn] = clsfier.model.np_assign_class(pred_class_out[clsn])
        else:
            in_class = {'noclassifier':['NoClass']*len(ori)}
            out_class = {'noclassifier':['NoClass']*len(ori)}
        
        errors, errors5 = onebatchpredict_errors(ori, pred_img, self.mask_ratio)
        
        # Clear current output and display test images
        for i in range(len(ori)):
            labi, posi = self.parse_lab_pos(pos, i, self.generators[ds][2])
            # print('current pos {}'.format(posi))
            # print('current lab {}'.format(labi))
            # print('current n {}'.format(n))
            # print('max n {}'.format(self.number_predict))
            mts_results = {}
            for clsn in self.mts_metrics.keys():
                self.mts_metrics[clsn].reset()
                self.mts_metrics[clsn].update(
                    ([labi], [in_class[clsn][i]], [out_class[clsn][i]]),
                    (np.expand_dims(ori[i][-int(ori.shape[1]*self.mask_ratio):],0),
                     np.expand_dims(pred_img[i][-int(ori.shape[1]*self.mask_ratio):],0)))
                mts_results[clsn] = self.mts_metrics[clsn].result_by_no()
            if self.add_centercount:
                pio_centers = {}
                for clsn, ccp in self.center_counter_pio.items():
                    ccp.reset()
                    ccp.fit_batch(
                        [labi], ([in_class[clsn][i]], [out_class[clsn][i]]), 
                        (np.expand_dims(ori[i][:-int(ori.shape[1]*self.mask_ratio)],0),
                         np.expand_dims(ori[i][-int(ori.shape[1]*self.mask_ratio):],0),
                         np.expand_dims(pred_img[i][-int(ori.shape[1]*self.mask_ratio):],0)))
                    pio_centers[clsn] = ccp.result_cond_1()
            else:
                pio_centers = {'noclassifier':None}
            
            psnr = -10.0 * np.log10(np.mean(np.square(pred_img[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:] - ori[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:])))
            kcenter_accuracy = [kcentroids_equal(to_kcentroid_seq(pred_img[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=n_centers)[1], to_kcentroid_seq(ori[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=n_centers)[1]) for n_centers in range(1,7)]
#            kcenter1_accuracy = kcentroids_equal(to_kcentroid_seq(pred_img[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=1)[1], to_kcentroid_seq(ori[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=1)[1])
#            kcenter5_accuracy = kcentroids_equal(to_kcentroid_seq(pred_img[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=5)[1], to_kcentroid_seq(ori[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], k=5)[1])
            
            meta_class_ori = '\nWeak Label '+labi
            if self.add_classifier:
                # meta_class_in = {}
                # meta_class_out = {}
                makedir = {}
                for clsn in list(self.classifier.keys()):
                    # meta_class_in[clsn] = '\nClassified '+in_class[clsn][i]
                    # meta_class_out[clsn] = '  Classified '+out_class[clsn][i]
                    makedir[clsn] = clsn
            else:
                # meta_class_in = {'noclassifier':''}
                # meta_class_out = {'noclassifier':''}
                makedir = {'noclassifier':'noclassifier'}
            
            for i_img in list(makedir.keys()):
                n_srow = 1
                plot_classes = False
                if self.add_centercount:
                    n_srow += 1
                if self.add_classifier and self.classifier[i_img].nclass>1:
                    plot_classes = True
                    n_srow += 1
                widths = [1]*n_srow+[2]*4
                ncols = len(widths)
                heights = [1]*4
                nrows = len(heights)
                figsize = (10*sum(widths),10*sum(heights))
                
                fig = plt.figure(constrained_layout=True, figsize=figsize)
                spec = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, 
                                         width_ratios=widths, 
                                         height_ratios=heights)
                axes = {}
                for row in range(nrows):
                    for col in range(ncols):
                        if self.add_centercount and col==int(plot_classes)+1 and row !=nrows-1 or plot_classes and col==1 and row in [1,2]:
                            axes[row, col] = fig.add_subplot(spec[row, col], polar=self.show_dist_polar)
                        elif not(plot_classes) or (plot_classes and col!=1):
                            axes[row, col] = fig.add_subplot(spec[row, col])
                
                if self.add_classifier and not(plot_classes):
                    # should be a unique value
                    probain_unique_class = ' %s'%'\n'.join(
                        ['%s (%.3f)'%(self.classifier[i_img].classes[ic], 
                        pred_class_in[i_img][i][ic]) for ic in range(self.classifier[i_img].nclass)])
                    probaout_unique_class = ' %s'%'\n'.join(
                        ['%s (%.3f)'%(self.classifier[i_img].classes[ic], 
                        pred_class_out[i_img][i][ic]) for ic in range(self.classifier[i_img].nclass)])
                else:
                    probain_unique_class = ''
                    probaout_unique_class = ''
                if self.add_classifier:
                    axes[1, 0].set_title('Original Sequence\nClass Pred:%s%s'%(
                        in_class[i_img][i], 
                        probain_unique_class))
                    axes[2, 0].set_title('Predicted Sequence PSNR=%.2f\nClass Pred:%s%s'%(
                        psnr,
                        out_class[i_img][i],
                        probaout_unique_class))
                else:
                    axes[1, 0].set_title('Original Sequence')
                    axes[2, 0].set_title('Predicted Sequence PSNR=%.2f'%psnr)#, y =-0.01)
                # Plot first column (images of spectra)
                axes[0, 0].imshow(masked[i].squeeze().transpose(), cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[0, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(masked[i].squeeze()), 
                                       xlabel='time', 
                                       lambda_size=masked[i].squeeze().shape[1], 
                                       mask=True)
                axes[0, 0].set_title('Masked Sequence'+meta_class_ori)
                # axes[0, 0].set_title('Masked Sequence'+meta_class_in[i_img])
                
                axes[2, 0].imshow(pred_img[i].squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[2, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(pred_img[i].squeeze()), 
                                       xlabel='time', 
                                       lambda_size=pred_img[i].squeeze().shape[1])
                # axes[2, 0].set_title('Predicted Sequence PSNR=%.2f'%(psnr)+meta_class_out[i_img])#, y =-0.01)
                
                axes[1, 0].imshow(ori[i].squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[1, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(ori[i].squeeze()), 
                                       xlabel='time', 
                                       lambda_size=ori[i].squeeze().shape[1])
                
                axes[3, 0].imshow(self.coef_diff * np.abs(ori[i].squeeze().transpose() * 1. - pred_img[i].squeeze().transpose() * 1.), vmin = 0, vmax = 1)
                self.format_image_axes(axes[3, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(ori[i].squeeze()), 
                                       xlabel='time', 
                                       lambda_size=ori[i].squeeze().shape[1])
                axes[3, 0].set_title('%s x Difference' % str(self.coef_diff))#, y =-0.01)
                
                addc = 0
                # Plot second column (center clusters distributions)
                if plot_classes:
                    addc += 1
                    probain_class = {
                        self.classifier[i_img].classes[ic]: pred_class_in[i_img][i][ic] for ic in range(self.classifier[i_img].nclass)}
                    probaout_class = {
                        self.classifier[i_img].classes[ic]: pred_class_out[i_img][i][ic] for ic in range(self.classifier[i_img].nclass)}
                    keys_in, vals_in = self.dict_to_keyval(probain_class)
                    keys_out, vals_out = self.dict_to_keyval(probaout_class)
                    assert keys_in==keys_out
                    # print('vals_in',vals_in)
                    # print('vals_out',vals_out)
                    # print('for rmax',vals_in+vals_out)
                    # print('rmax',max(vals_in+vals_out))
                    self.polar_bar_plot(axes[1, addc], vals_in, keys_in, bottom=.1,
                                        count_step=0.1, count_ceil=0, 
                                        rmax=max(vals_in+vals_out), 
                                        color='tab:purple')
                    axes[1, addc].set_title('Class Pred IN\n')
                    self.polar_bar_plot(axes[2, addc], vals_out, keys_out, bottom=.1,
                                        count_step=0.1, count_ceil=0, 
                                        rmax=max(vals_in+vals_out), 
                                        color='tab:purple')
                    axes[2, addc].set_title('Class Pred OUT\n')
                
                if self.add_centercount:
                    addc +=1
                    keys_prior, vals_prior = self.dict_to_keyval(pio_centers[i_img]['centers']['c0'][labi])
                    keys_in, vals_in = self.dict_to_keyval(pio_centers[i_img]['centers']['c1'][labi])
                    keys_out, vals_out = self.dict_to_keyval(pio_centers[i_img]['centers']['c2'][labi])
                    assert keys_prior==keys_in==keys_out
                    self.polar_bar_plot(axes[0, addc], vals_prior, keys_prior, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[0,addc].set_title('Center distribution PRIOR\n')
                    self.polar_bar_plot(axes[1,addc], vals_in, keys_in, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[1, addc].set_title('Center distribution IN\n')
                    self.polar_bar_plot(axes[2, addc], vals_out, keys_in, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[2,addc].set_title('Center distribution OUT\n')
                    cm, (kin, kout) = self.dict2d_to_array(
                        pio_centers[i_img]['centers']['c1c2'][labi], 
                        with_keys=True)
                    self.plot_heatmap(
                        cm, kin, kout, 
                        'Center IN VS Center OUT',
                        axes[3,addc], with_cbar=True, with_labels=False,
                        xtick_step=5, ytick_step=5, linewidths=.5,
                        vmin=0, vmax=np.max(cm))
                    # self.polar_bar_plot(axes[3, 1], 
                    #                     [abs(vi-vo) for vi,vo in zip(vals_in, vals_out)], 
                    #                     keys, bottom=.1, count_ceil=0.05, 
                    #                     rmax=max([
                    #                         abs(vi-vo) for vi,vo in zip(
                    #                             vals_in, vals_out)]))
                    axes[3, addc].set_title('Center distribution ERROR\nH(in)=%.2f H(out)=%.2f I(in;out)=%.2f\n'%(
                        pio_centers[i_img]['info_c1c2']['entropies'][0][labi],
                        pio_centers[i_img]['info_c1c2']['entropies'][1][labi],
                        pio_centers[i_img]['info_c1c2']['mutual_info'][labi]))
                
                # Start plotting 3rd column (or 2nd when no center): Raw CV errors (PSNR, SSIM)
                axes[0, 1+addc].plot(range(1, len(errors[0,i])+1), 
                                     -10*np.log10(errors[0, i]), 
                                     label='PSNR')
                self.format_axis(axes[0, 1+addc], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[0, 1+addc], vmin=0, vmax=len(errors[0, i]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[0, 1+addc], legend_loc='upper center', fontsize='x-small')
                
                axes[2, 1+addc].plot(range(1, len(errors[1,i])+1), 
                                     errors[1, i], label='SSIM')
                axes[2, 1+addc].plot(range(1, len(errors[1,i])+1), 
                                     np.ones_like(errors[1, i]), label='best', linestyle=':', color='g')
                self.format_axis(axes[2, 1+addc], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[2, 1+addc], vmin=0, vmax=len(errors[1,i]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[2, 1+addc], legend_loc='upper center', fontsize='x-small')
                
                # Start plotting the other columns: Raw physical errors (centers assignment)
                for j in range(1,7):
                    row, col = [int((j-1)%2)*2, 2+int((j-1)//2)+addc]
                    axes[row, col].plot(range(1, len(kcenter_accuracy[j-1])+1), 
                                        kcenter_accuracy[j-1], label='%i-Center'%j)
                    axes[row, col].plot(range(1, len(kcenter_accuracy[j-1])+1), 
                                        [kinter(j) for _ in range(len(kcenter_accuracy[j-1]))], label='%i-RandomBaseground'%j, linestyle=':', color='r')
                    axes[row, col].plot(range(1, len(kcenter_accuracy[j-1])+1), 
                                        np.ones_like(kcenter_accuracy[j-1]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(kcenter_accuracy[j-1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', fontsize='x-small')
                
                # Plot 3rd column (or 2nd when no center): Avg on 5% time CV errors (PSNR, SSIM)
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=errors5[0, i], tofit=errors[0, i])
                axes[1, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     -10*np.log10(errors5[0, i]), label='PSNR')
                self.format_axis(axes[1, 1+addc], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[1, 1+addc], vmin=vmin, vmax=len(errors5[0, i]), lmin=0, lmax=len(errors[0, i]), step = 10, axis = 'x', type_labels='int', ax_label='time')
                self.set_description(axes[1, 1+addc], legend_loc='upper center', fontsize='x-small')
                
                axes[3, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     errors5[1, i], label='SSIM')
                axes[3, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     np.ones_like(errors5[1, i]), label='best', linestyle=':', color='g')
                self.format_axis(axes[3, 1+addc], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[3, 1+addc], vmin=vmin, vmax=len(errors5[1,i]), lmin=0, lmax=len(errors[1,i]), step = 10, axis = 'x', type_labels='int', ax_label='time')
                self.set_description(axes[3, 1+addc], legend_loc='upper center', fontsize='x-small')
    
                # Plot all the other columns: Avg on 5% time physical errors (centers assignment)
                for j in range(1,7):
                    row, col = [int((j-1)%2)*2+1, 2+int((j-1)//2)+addc]
                    axes[row, col].plot(*forplot_assignement_accuracy(kcenter_accuracy[j-1], bin_size=int(self.label_length * 0.05)), label='%i-Center'%j)
                    axes[row, col].plot(np.arange(.5, len(kcenter_accuracy[j-1])+.5), 
                                        [kinter(j) for _ in range(len(kcenter_accuracy[j-1]))], label='%i-RandomBaseground'%j, linestyle=':', color='r')
                    axes[row, col].plot(np.arange(.5, len(kcenter_accuracy[j-1])+.5), 
                                        np.ones_like(kcenter_accuracy[j-1]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(kcenter_accuracy[j-1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', fontsize='x-small')            
                
                if not os.path.exists(os.path.join(self.results_dir, 'prediction', makedir[i_img])):
                    os.makedirs(os.path.join(self.results_dir, 'prediction', makedir[i_img]))
                
                for u in range(2):
                    for v in range(1,5):
                        axes[2 * u, v+addc].set_title('time predictions')
                        axes[2 * u + 1, v+addc].set_title('Avg 5% time slices predictions')
            
                # spec.tight_layout(fig)
                
                self.savefig_autodpi(os.path.join(
                    self.results_dir, 'prediction', makedir[i_img],
                    'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_spectra_polar-{}.png'.format(
                        ds, n, labi, posi, self.show_dist_polar)),
                    bbox_inches='tight')
                plt.close()
                
                if self.add_centercount:
                    # add sample_info
                    self.plot_1simp_pred_centers(pio_centers[i_img], labi, 
                                                 in_class[i_img][i], 
                                                 out_class[i_img][i],
                                                 save_name=os.path.join(
                                                     'prediction', 
                                                     makedir[i_img],
                                                     'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_detailedcentercount_polar-{}.png'.format(
                                                         ds, n, labi, posi, self.show_dist_polar)))
                
                self.plot_mtsres(
                    {'by_no':mts_results[i_img]}, 
                    meta=(labi, 
                          in_class[i_img][i], 
                          out_class[i_img][i]),
                    save_name=os.path.join(
                        'prediction', makedir[i_img],
                        'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_mts_results.png'.format(
                            ds, n, labi, posi)))
            
            self.save_features_sequence(
                ori[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], 
                pred_img[i,:,:,:].squeeze()[-int(ori.shape[1] * self.mask_ratio):,:], 
                ds, n, posi, labi, self.feat_legends, save_dir='prediction')
            
            n += 1
            
        return n
    
    def plot_mtsres(self, mts_results, meta=None, glob=None, glob_meta=None, 
                    save_name='mts_results.png'):
        # mts_results = {'by_no':self.mts_metrics[clsn].result_by_no(),
        #                 'by_1':self.mts_metrics[clsn].result_by_1(),
        #                 'by_1_2':self.mts_metrics[clsn].result_by_1_2(),
        #                 'by_1_3':self.mts_metrics[clsn].result_by_1_3(),
        #                 'by_1_2_3':self.mts_metrics[clsn].result_by_1_2_3()}
        # mts_results may not include only 'by_no'
        # meta = (lab, clas1, clas2) is info to be used when given to plot and 
        # inform only on the corresponding info, can contain Nones
        # glob = {'by_no':self.mts_metrics[clsn].result_by_no()} is mts global 
        # metrics to be ploted when given
        # glob_meta = (lab, clas1, clas2) is info to be used for the glob when 
        # given to plot and informa only on the glob corresponding info
        
        # Plot the multi time series usual metrics
        w_box = 15 # width for one box of result
        h_box = 15 # height for one box of result
        test_case = False
        if mts_results.keys()=={'by_no':None}.keys():
            # mts_results only contains 'by_no' results (simple prediction case)
            assert meta is not None, "information on data should be given"
            # fig contains only one time all mts metrics (one plot for each metric)
            # similar to self.plot_1simp_pred_centers
            previous_font_size = plt.rcParams['font.size']
            plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*6/5)})
            parms = self.parms_mtsres_one_label('simple_pred', (w_box, h_box),
                                                mts_results, meta)      
            dict_plot_fn_keys = ['by_no']
        elif 'by_no' not in mts_results.keys():
            # long prediction case
            assert meta is not None and meta[0] is not None, "meta info should be given for the long prediction case"
            assert all(key in mts_results.keys() for key in [
                'by_1', 'by_1_2', 'by_1_3', 'by_1_2_3']), "all keys except 'by_no' should be given"
            assert glob is not None
            assert glob.keys()=={'by_no':None}.keys(), "glob should contain only by_no result"
            assert glob_meta is not None, "information on glob_data should be given"
            # fig contains (key2+1)*(key3+1) plots for each mts metric plus global mts metrics
            # similar to self.plot_1long_pred_centers
            previous_font_size = plt.rcParams['font.size']
            plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*6/5)})
            parms = self.parms_mtsres_one_label('long_pred', (w_box, h_box),
                                                mts_results, meta)
            dict_plot_fn_keys = ['by_no', 'by_1', 'by_1_2', 'by_1_3', 'by_1_2_3']
        else:
            # test or longtest case
            assert all(key in mts_results.keys() for key in [
                'by_no', 'by_1', 'by_1_2', 'by_1_3', 'by_1_2_3']), "all keys except should be given"
            # key2 figs created and saved that each contains 
            # (key2+1)*(key3+1) plots for each mts metric
            # plus global and key1 plots for each mts metric
            # similar to key2 times self.plot_clsres_one_label
            previous_font_size = plt.rcParams['font.size']
            plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*6/5)})
            parms = self.parms_mtsres_one_label('test', (w_box, h_box),
                                                mts_results, meta)
            dict_plot_fn_keys = ['by_no', 'by_1', 'by_1_2', 'by_1_3', 'by_1_2_3']
            test_case = True
        
        dict_plot_fn = {
            'by_no': lambda fig, parms, mts_results, meta: self.plot_mts_no(
                 fig, parms, mts_results, meta, 
                 update=False, lc=None, globline=True),
            'by_1': lambda fig, parms, mts_results, meta: self.plot_mts_1(
                 fig, parms, mts_results, meta, with_all=test_case,
                 update=False, lc=None, globline=True),
            'by_1_2': lambda fig, parms, mts_results, meta: self.plot_mts_1_2(
                 fig, parms, mts_results, meta, 
                 update=False, lc=None, globline=True),
            'by_1_3': lambda fig, parms, mts_results, meta: self.plot_mts_1_3(
                 fig, parms, mts_results, meta, 
                 update=False, lc=None, globline=True),
            'by_1_2_3': lambda fig, parms, mts_results, meta: self.plot_mts_1_2_3(
                 fig, parms, mts_results, meta, 
                 update=False, lc=None, globline=True)}
        
        save_name = save_name.split('.')
        for label in parms[0]:
            save_name_onelab = save_name[:-2]+[save_name[-2]+'__label_{}'.format(label)]+save_name[-1:]
            self.create_onefig_mtsmeasures(label,
                parms, mts_results, meta, glob, glob_meta, 
                {key: dict_plot_fn[key] for key in dict_plot_fn_keys}, 
                save_name_onelab, update=False)
        
        plt.rcParams.update({'font.size': int(previous_font_size)})
    
    def plot_classacc(self, cta_results, no_labcls, 
                      save_name='cta_results.png'):
        # cta_results = {'all': {l:{k1:{k2:{t:value}}}},
        #                '0,1,2':{l:{k1:{k2:value}}},
        #                '0,1,3':{l:{k1:{t:value}}},
        #                '0,2,3':{l:{k2:{t:value}}},
        #                '1,2,3':{k1:{k2:{t:value}}},
        #                '0,1':{l:{k1:value}},
        #                '...':..,
        #                '2,3':{k2:{t:value}},
        #                '0':{l:value},
        #                '..':..,
        #                '3':{t:value},
        #                'glob':value}
        # 
        
        # Plot the multi time series usual metrics
        w_box = 15 # width for one box of result
        h_box = 15 # height for one box of result
        previous_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': int(previous_font_size*w_box/30)})
        parms = self.parms_ctares_one_label((w_box, h_box),
                                            cta_results)      
        save_name = save_name.split('.')
        self.create_onefig_ctameasures(parms, cta_results, 
                                       no_labcls,  save_name)
        
        plt.rcParams.update({'font.size': int(previous_font_size)})
    
    def parms_ctares_one_label(self, wh_box, cta_results):
        
        w_box, h_box = wh_box # width and height for one box of result
        dict_temp = cta_results['all']
        labels = list(cta_results['all'].keys())
        nlabels = len(labels)
        dict_temp = dict_temp[labels[0]]
        classes1 = list(dict_temp.keys()) # keys as k1
        nk1 = len(classes1)
        dict_temp = dict_temp[classes1[0]]
        classes2 = list(dict_temp.keys()) # keys are k2
        nk2 = len(classes2)
        w_boxes = (1+max(nlabels, nk1, nk2))
        h_boxes = (3)
        w_sep = .05 # horizontal separation in %
        h_sep = .05 # vertical separation in %
        width = w_box * w_boxes
        height = h_box * h_boxes
        w_sep *= width
        h_sep *= height
        w_box = (width-w_sep)/w_boxes
        h_box = (height-h_sep)/h_boxes
        w_sep /= w_boxes
        h_sep /= h_boxes
        
        return (labels, classes1, classes2, 
                w_sep, h_sep, w_box, h_box, width, height)
    
    def parms_mtsres_one_label(self, type_result, wh_box,
                               mts_results, meta):
        
        w_box, h_box = wh_box # width and height for one box of result
        if type_result == 'test':
            dict_temp = mts_results['by_1_2_3'][list(mts_results['by_1_2_3'].keys())[0]]
            labels = list(dict_temp.keys())
            nlabels = len(labels)
            dict_temp = dict_temp[labels[0]]
            classes1 = list(dict_temp.keys()) # keys as k1
            nk1 = len(classes1)
            dict_temp = dict_temp[classes1[0]]
            classes2 = list(dict_temp.keys()) # keys are k2
            nk2 = len(classes2)
            w_boxes = (1+max(nlabels, nk1))
            h_boxes = (2+nk2)
            w_sep = .05 # horizontal separation in %
            h_sep = .05 # vertical separation in %
        elif type_result == 'simple_pred':
            w_boxes = 1
            h_boxes = 1
            w_sep = 0 # horizontal separation in %
            h_sep = 0 # vertical separation in %
            labels = [meta[0]]
            classes1 = None
            classes2 = None
            nlabels = 1
        else:
            #(type_result == 'long_pred')
            dict_temp = mts_results['by_1_2_3'][list(mts_results['by_1_2_3'].keys())[0]]
            labels = [meta[0]]
            nlabels = len(labels)
            dict_temp = dict_temp[labels[0]]
            classes1 = list(dict_temp.keys()) # keys as k1
            nk1 = len(classes1)
            dict_temp = dict_temp[classes1[0]]
            classes2 = list(dict_temp.keys()) # keys are k2
            nk2 = len(classes2)
            w_boxes = (1+nk1)
            h_boxes = (2+nk2)
            w_sep = .05 # horizontal separation in %
            h_sep = .05 # vertical separation in %
        width = w_box * w_boxes
        height = h_box * h_boxes
        w_sep *= width
        h_sep *= height
        w_box = (width-w_sep)/w_boxes
        h_box = (height-h_sep)/h_boxes
        w_sep /= w_boxes
        h_sep /= h_boxes
        
        return (labels, classes1, classes2, 
                w_sep, h_sep, w_box, h_box, width, height)
    
    def plot_mts_no(self, fig, parms, mts_results, meta, 
                    update=False, lc=None, globline=True, common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        frame_dims = (w_sep/2,height-h_box-h_sep/2,w_box,h_box)
        if update:
            axs = self.get_axes_mts(fig, "Global avg mts metrics by Label {}".format(meta[0]))
        else:
            axs = self.define_axes_mts(fig, frame_dims)
        if self.backg_color:
            axs[0].set_facecolor('thistle')
        self.plot_mts_set(
            axs, mts_results['by_no'], meta, update=update, lc=lc, 
            globline=globline,
            title="Global avg mts metrics by Label {}".format(meta[0]),
            common_leg = common_leg)
    
    def plot_mts_1_alll(self, fig, parms, mts_results, meta, 
                        update=False, lc=None, globline=True, common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        assert meta is not None
        assert meta[0] is not None
        for i_box, lbl in enumerate(labels):
            # by_1 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, height-h_box-h_sep/2, 
                          w_box, h_box)
            if update:
                axs = self.get_axes_mts(fig, "Remainder By Label {}".format(lbl))
            else:
                axs = self.define_axes_mts(fig, frame_dims)
            if lbl==meta[0] and self.backg_color:
                axs[0].set_facecolor('peachpuff')
            self.plot_mts_set(
                axs, 
                {metric:mts_value[lbl] for metric, mts_value in mts_results['by_1'].items()}, 
                (lbl,None,None), 
                update=update, lc=lc, globline=globline,
                title="Remainder By Label {}".format(lbl),
                common_leg = common_leg)
    
    def plot_mts_1_onel(self, fig, parms, mts_results, meta,
                        update=False, lc=None, globline=True, 
                        common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        assert meta is not None
        assert meta[0] is not None
        label = meta[0]
        assert label in labels
        frame_dims = (w_sep/2, height-2*h_box-3*h_sep/2, w_box, h_box)
        if update:
            axs = self.get_axes_mts(fig, "By Label {}".format(label))
        else:
            axs = self.define_axes_mts(fig, frame_dims)
        if self.backg_color:
            axs[0].set_facecolor('peachpuff')
        self.plot_mts_set(
            axs,
            {metric:mts_value[label] for metric, mts_value in mts_results['by_1'].items()}, 
            (label,None,None), 
            update=update, lc=lc, globline=globline,
            title="By Label {}".format(label),
            common_leg = common_leg)
    
    def plot_mts_1(self, fig, parms, mts_results, meta, with_all=True,
                   update=False, lc=None, globline=True, common_leg = False):
        self.plot_mts_1_alll(fig, parms, mts_results, meta,
                             update=update, lc=lc, globline=globline,
                             common_leg = common_leg)
        self.plot_mts_1_onel(fig, parms, mts_results, meta,
                             update=update, lc=lc, globline=globline,
                             common_leg = common_leg)
    
    def plot_mts_1_2(self, fig, parms, mts_results, meta,
                     update=False, lc=None, globline=True, common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        assert meta[0] is not None
        label = meta[0]
        if meta[1] is not None:
            assert meta[1] in classes1
        for i_box, clsn in enumerate(classes1):
            if meta[1] is not None and clsn != meta[1]:
                continue
            # cond_1_2 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, height-2*h_box-3*h_sep/2,
                          w_box, h_box)
            if update:
                axs = self.get_axes_mts(fig, "By Input Class {}".format(clsn))
            else:
                axs = self.define_axes_mts(fig, frame_dims)
            if self.backg_color:
                axs[0].set_facecolor('palegreen')
            self.plot_mts_set(
                axs,
                {metric:mts_value[label][clsn] for metric, mts_value in mts_results['by_1_2'].items()}, 
                (label,clsn,None),
                update=update, lc=lc, globline=globline,
                title="By Input Class {}".format(clsn),
                common_leg = common_leg)
    
    def plot_mts_1_3(self, fig, parms, mts_results, meta,
                     update=False, lc=None, globline=True, common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        assert meta[0] is not None
        label = meta[0]
        if meta[2] is not None:
            assert meta[2] in classes2
        for i_box, clsn in enumerate(classes2):
            if meta[2] is not None and clsn != meta[2]:
                continue
            # cond_1_2 results
            frame_dims = (w_sep/2, height-(i_box+3)*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            if update:
                axs = self.get_axes_mts(fig, "By Output Class {}".format(clsn))
            else:
                axs = self.define_axes_mts(fig, frame_dims)
            if self.backg_color:
                axs[0].set_facecolor('palegoldenrod')
            self.plot_mts_set(
                axs,
                {metric:mts_value[label][clsn] for metric, mts_value in mts_results['by_1_3'].items()}, 
                (label,None,clsn), 
                update=update, lc=lc, globline=globline,
                title="By Output Class {}".format(clsn),
                common_leg = common_leg)
    
    def plot_mts_1_2_3(self, fig, parms, mts_results, meta,
                       update=False, lc=None, globline=True, 
                       common_leg = False):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        assert meta[0] is not None
        label = meta[0]
        if meta[1] is not None:
            assert meta[1] in classes1
        if meta[2] is not None:
            assert meta[2] in classes2
        for i_box1, clsn1 in enumerate(classes1):
            if meta[1] is not None and clsn1 != meta[1]:
                continue
            for i_box2, clsn2 in enumerate(classes2):
                if meta[2] is not None and clsn2 != meta[2]:
                    continue
                # cond_1_2_3 results
                frame_dims = ((i_box1+1)*(w_box + w_sep)+w_sep/2, 
                              height-(i_box2+3)*(h_box+h_sep)+h_sep/2,
                              w_box, h_box)
                if update:
                    axs = self.get_axes_mts(fig, "By Input class {} Output class {}".format(clsn1, clsn2))
                else:
                    axs = self.define_axes_mts(fig, frame_dims)
                self.plot_mts_set(
                    axs,
                    {metric:mts_value[label][clsn1][clsn2] for metric, mts_value in mts_results['by_1_2_3'].items()}, 
                    (label,clsn1,clsn2),
                    update=update, lc=lc, globline=globline,
                    title="By Input class {} Output class {}".format(clsn1, clsn2),
                    common_leg = common_leg)
    
    def create_onefig_mtsmeasures(self, label, parms, mts_results, meta, 
                                  glob, glob_meta, key_pltfn, save_name, 
                                  overwrite=False, update=False):
        # parms is tuple(labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height)
        # pio_results contains measurements defined in self.plot_clsres_one_label
        # key_pltfn is a dict {key: function}
        # where for each key from pio_results, it gives the plotting function to apply
        fname = os.path.join(self.results_dir,'.'.join(save_name))
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        if update:
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(width, height))
        
        for key, fn in key_pltfn.items():
            if key in mts_results.keys():
                use_res = mts_results
                use_meta = meta
            else:
                assert key in glob.keys()
                use_res = glob
                use_meta = glob_meta
            if use_meta is not None:
                assert label == use_meta[0]
                use_meta = (label, *(use_meta[1:]))
            else:
                use_meta = (label, None, None)
            # create axes and plot
            fn(fig, parms, use_res, use_meta)
        
        if not(manual_mode):
            self.savefig_autodpi(fname,
                bbox_inches=None)
                # bbox_inches='tight')
            plt.close()
        else:
            return os.path.join(self.results_dir,save_name)
    
    def plot_cta(self, fig, parms, results, legends, idx, chunks_every=60):
        # Plots a row of results in fig
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        acc_nn, glob3_acc_nn, globn_acc_nn, globn3_acc_nn = results
        title, index = legends
        frame_dims = (w_sep/2, 
                      height-(idx+1)*(h_box+h_sep)+h_sep/2,
                      w_box, h_box)
        # global results
        axs = self.define_axes_cta(fig, frame_dims)
        self.plot_cta_set(
            axs,
            globn3_acc_nn, 
            globn_acc_nn, 
            title=title+"\nGlobal : ",
            chunks_every=chunks_every)
        for i_box, name in enumerate(acc_nn.keys()):
            # cond_1_2_3 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, 
                          height-(idx+1)*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            axs = self.define_axes_cta(fig, frame_dims)
            self.plot_cta_set(
                axs,
                glob3_acc_nn[name], 
                acc_nn[name], 
                title=title+"\n"+index+" : "+name, 
                chunks_every=chunks_every)
    
    def create_onefig_ctameasures(self, parms, cta_results, no_labcls,
                                  save_name, overwrite=False, chunks_every=60):
        # parms is tuple(labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height)
        # cta_results contains measurements defined in self.plot_classacc
        nolabel, noclass = no_labcls
        fname = os.path.join(self.results_dir,'.'.join(save_name))
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        fig = plt.figure(figsize=(width, height))
        
        for idx, (nn, title, index) in enumerate(zip( # a line of results
                ['0,1', '0,2','1,2'], # accuracy 1st VS 2nd, indexed by 1st
                ['Weak labels VS Input Classification', 
                 'Weak labels VS Output Classification', 
                 'Input Classification VS Output Classification'], # name of the accuracy
                ['By label', 'By label', 'By Input Classification'])): # type of indexing
			
            get_count_jointkey = lambda dicti, key, dim: get_count_jointkey(
			
                dicti[
                    [[noclass, nolabel][int(nn.split(',')[dim]=='0')], 
                     key][int(key in dicti.keys())]],key, dim+1) if (
                    isinstance(dicti, dict) and dim<2
                    ) else dicti
			
            nul_div = lambda x,y: 0 if x==0 else np.divide(x,y)
			
            count_nn = {l: get_count_jointkey(
                cta_results[nn+',3'],l,0) for l in cta_results[nn+',3'].keys()}
            get_time_key = lambda dicti, key: dicti[key] if key in dicti.keys() else 0
            # time accuracies dict by name nn of indexation
            acc_nn = {
                l: {
                    t: nul_div(get_time_key(count_nn[l],t),
                               get_time_key(cta_results[nn.split(',')[0]+',3'][l],t)) for t in natsorted(
                        set([vv for v in [
                            list(cta_results[nn.split(',')[0]+',3'][ll].keys()
                                 ) for ll in cta_results[
                                     nn.split(',')[0]+',3'].keys()
                            ] for vv in v]))
                    } for l in count_nn.keys()
                }
			
            # global accuracies over time dict by name nn of indexation
            glob3_acc_nn = {l: nul_div(
                np.sum(list(count_nn[l].values())), 
                np.sum(list(cta_results[nn.split(',')[0]+',3'][l].values()))
                ) for l in count_nn.keys()}
			
            get_any_key = lambda dicti, key: dicti[key] if key in dicti.keys() else 0
            # global (over all keys0) time accuracies dict by time
			
            globn_acc_nn = {
                t: nul_div(np.sum(get_time_key(count_nn[l],t) for l in count_nn.keys()),
                           np.sum(get_time_key(cta_results[nn.split(',')[0]+',3'][l],t
                                               ) for l in count_nn.keys())) for t in natsorted(
                               set([vv for v in [
                                   list(cta_results[nn.split(',')[0]+',3'][ll].keys()
                                        ) for ll in cta_results[
                                            nn.split(',')[0]+',3'].keys()
                                            ] for vv in v]))
                }
            # global (over all keys0) accuracies over time scalar
			
            globn3_acc_nn = nul_div(
                np.sum(np.sum(get_time_key(count_nn[l],t) for l in count_nn.keys()
                              ) for t in natsorted(
                    set([vv for v in [
                        list(cta_results[nn.split(',')[0]+',3'][ll].keys()
                            ) for ll in cta_results[
                                nn.split(',')[0]+',3'].keys()
                                ] for vv in v]))),
                np.sum(np.sum(get_time_key(cta_results[nn.split(',')[0]+',3'][l],t
                                    ) for l in count_nn.keys()) for t in natsorted(
                                        set([vv for v in [
                                            list(cta_results[nn.split(',')[0]+',3'][ll].keys()
                                                ) for ll in cta_results[
                                                    nn.split(',')[0]+',3'].keys()
                                                    ] for vv in v]))))
            self.plot_cta(fig, parms, (
                acc_nn, glob3_acc_nn,
                globn_acc_nn, globn3_acc_nn),
                (title, index), idx, chunks_every=chunks_every)
        
        if not(manual_mode):
            self.savefig_autodpi(fname,
                bbox_inches=None)
                # bbox_inches='tight')
            plt.close()
        else:
            return fname
    
    def plot_time_mts(self, axs, mts_results, chunks_every=60, update=False, 
                      lc=None, globline=True, common_leg = False, ax_leg=None):
        if lc == None:
            label = 'model?'
            color = 'b'
            alpha = 1
        else:
            try:
                label, (color, alpha) = lc
            except:
                label, color = lc
                alpha = 1
        for i_key, key in enumerate(natsorted(mts_results.keys())):
            # for each metric
            axs[i_key].plot(
                np.arange(1, len(mts_results[key][2])+1), 
                mts_results[key][2], color=color, alpha=alpha, linestyle='-', 
                label=label)
            if globline:
                axs[i_key].plot(
                    np.arange(1, len(mts_results[key][2])+1), 
                    [mts_results[key][1]]*len(mts_results[key][2]),
                    color='m', alpha=alpha,linestyle=':', label='global'+[' ',''][int(label=='')]+label)
            if update:
                old_title = axs[i_key].get_title(loc='left')
                # print(old_title.count('/'))
                # print(old_title.count('/')==3)
                axs[i_key].set_title(
                    old_title+['','\n'][int(old_title.count('/')==3)]+" / {:.2f}".format(
                        mts_results[key][1]), loc='left', fontsize='xx-small')
            else:
                axs[i_key].set_title("Metric {} - global result:\n {:.2f}".format(key,mts_results[key][1]), loc='left', fontsize='xx-small')
            if common_leg and ax_leg is not None:
                leg_by_col = 1
                ax_leg.legend(
                    *axs[i_key].get_legend_handles_labels(),
                    loc='lower center', fontsize='xx-small', frameon=False,
                    ncol=int(len(axs[i_key].get_legend_handles_labels()[1])//leg_by_col) + int(
                        len(axs[i_key].get_legend_handles_labels()[1])%leg_by_col!=0))
            else:
                axs[i_key].legend(loc='upper center', fontsize='xx-small')
            axs[i_key].set_xlabel('time', fontsize='xx-small')
            self.format_axis(axs[i_key], vmin=1, vmax=len(mts_results[key][2]),
                             step=int(np.ceil(int(len(mts_results[key][2])/6)/5)*5), 
                             axis='x', ax_label='time', type_labels='int', 
                             minor=False, fontsize='xx-small')
            vmin=np.nanmin(mts_results[key][2])
            vmax=np.nanmax(mts_results[key][2])
            step = (vmax-vmin)/2
            self.format_axis(axs[i_key], vmin=vmin, vmax=vmax, step=step, 
                             axis='y', ax_label=None, type_labels='%.2f', 
                             minor=False, margin=[1,1], fontsize='xx-small')
    
    def plot_time_cta(self, axs, cta_results, chunks_every=60):
        cta_time, cta_glob = cta_results
        axs.plot(
            np.arange(chunks_every, len(cta_time)*chunks_every+1, chunks_every), 
            [cta_time[kt] for kt in natsorted(cta_time.keys())], 'b-', label='by time')
        axs.plot(
            np.arange(chunks_every, len(cta_time)*chunks_every+1, chunks_every), 
            [cta_glob]*len(cta_time), 'm:', label='global')
        # axs.set_title("Metric {} - global result: {:.2f}".format(key,mts_results[key][1]), loc='left', fontsize='xx-small')
        # axs.legend(loc='upper center', fontsize='xx-small')
        axs.set_xlabel('time', fontsize='xx-small')
        self.format_axis(axs, vmin=1, vmax=len(cta_time)*chunks_every,
                         step=chunks_every, 
                         axis='x', ax_label='time', type_labels='int', 
                         minor=False, fontsize='xx-small')
        vmin=np.nanmin(list(cta_time.values()))
        vmax=np.nanmax(list(cta_time.values()))
        step = (vmax-vmin)/2
        self.format_axis(axs, vmin=vmin, vmax=vmax, step=step, 
                         axis='y', ax_label=None, type_labels='%.2f', 
                         minor=False, margin=[1,1], fontsize='xx-small')
    
    def plot_mts_set(self, axs, mts_results, meta, update=False, lc=None, 
                     globline=True, title="Global avg mts metrics",
                     common_leg = False):
        # 
        assert len(mts_results.keys())==3, "code made just for 3 metrics"
        global_count = mts_results[list(mts_results.keys())[0]][0]
        assert all(mts_results[m_key][0]==global_count for m_key in mts_results.keys()), "the global count should not depend opn the metric"
        axs[0].set_title(title, loc='left', fontsize='xx-large')
        # self.plot_title(axs[1], "Global time metrics")
        # self.plot_glob_mts(axs[2], mts_results, stride_info=4)
        self.plot_title(
            axs[1], 
            "Time and Global evaluated metrics\nGlobal Count : {:d}".format(
                int(global_count)),
            update=update, 
            posxy=(axs[1].get_xlim()[1]/2,
                   axs[1].get_ylim()[1]*3/4))
        self.plot_time_mts((axs[2], axs[3], axs[4]), mts_results, 
                           update=update, lc=lc, globline=globline,
                           common_leg = common_leg, ax_leg=axs[1])
    
    def plot_cta_set(self, axs, cta_results_glob, cta_results_time, 
                     title="Global avg mts metrics", chunks_every=60):
        # 
        axs[0].set_title(title, loc='left', fontsize='xx-large')
        # self.plot_title(axs[1], "Global time metrics")
        # self.plot_glob_mts(axs[2], mts_results, stride_info=4)
        self.plot_title(axs[1], "Time and Global classification accuracy\nGlobal Accuracy : {:.2f}%".format(int(cta_results_glob*100)))
        self.plot_time_cta(axs[2], (cta_results_time, cta_results_glob), chunks_every=chunks_every)
    
    # TODO: superpose ds on feats
    # TODO: pdf instead of png
    
    def add_axes_inches(self, fig, inches_xywh, projection=None):
        #
        w,h = fig.get_size_inches()
        return fig.add_axes((
            inches_xywh[0]/w,
            inches_xywh[1]/h,
            inches_xywh[2]/w,
            inches_xywh[3]/h,
            ), 
            projection=projection)
    
    def get_axes_mts(self, fig, title):
        index = [(idx, ax) for idx,ax in enumerate(fig.axes) if title==ax.get_title(loc='left')]
        assert len(index)==1, "there should be just one index in {}".format(index)
        index = index[0][0]
        return fig.axes[index: index+5]
    
    def define_axes_mts(self, fig, frame_dims):
        # 
        ratio_pad_fn = lambda ratio_for_padplot: lambda x,y,w,h:(
            x+(1-ratio_for_padplot)/2*w,
            y+(1-ratio_for_padplot)/2*h,
            ratio_for_padplot*w,
            ratio_for_padplot*h)
        # print('frame_dims',frame_dims)
        (x, y, w, h) = ratio_pad_fn(0.95)(*frame_dims)
        # print('(x,y,w,h)', (x,y,w,h))
        # draw the frame box
        ax_frame = self.add_axes_inches(fig, (x, y, w, h))
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        if not(self.frame_res):
            ax_frame.set_axis_off()
        ratio_pad = ratio_pad_fn(0.8)
        delta = 2 # number of subrows for a plot
        n_rows_full = 2
        n_cols_full = 3
        parms = lambda x:(x, # number of rows (correspond to height for plots)
                          x*delta, # number of subrows (correspond to height for titles)
                          (n_rows_full-x)*delta) # padding before the first row in terms of subrows
        
        n_rows, n_subrows, pad_subrows = parms(n_rows_full) 
        
        pos_time_title = (
            x, y+(n_rows_full*delta-pad_subrows-1)*h/(n_rows_full*delta), 
            w, h/(n_rows_full*delta))
        
        ax_time_title = self.add_axes_inches(fig, pos_time_title)
        # w_pad = pos_global_title[2]/n_cols_full
        
        # ax_globals = self.add_axes_inches(fig, ratio_pad(*(
        #     pos_global_title[0], pos_global_title[1]-h/n_rows_full,
        #     pos_global_title[2], h/n_rows_full)))
        
        # w_stride = 0
        # w_col = pos_global_title[2]/n_cols_full
        
        # ax_time_title = self.add_axes_inches(fig, ratio_pad(*(
        #     pos_time_title[0],
        #     pos_time_title[1]-h/n_rows_full-h/(n_rows_full*delta),
        #     pos_time_title[2], h/(n_rows_full*delta))))
        
        w_pad = pos_time_title[2]/n_cols_full
        ax_metric1 = self.add_axes_inches(fig, ratio_pad(*(
            pos_time_title[0]+w_pad*0, 
            pos_time_title[1]-h/n_rows_full,
            pos_time_title[2]/n_cols_full, h/n_rows_full)))
        ax_metric2 = self.add_axes_inches(fig, ratio_pad(*(
            pos_time_title[0]+w_pad*1, 
            pos_time_title[1]-h/n_rows_full,
            pos_time_title[2]/n_cols_full, h/n_rows_full)))
        ax_metric3 = self.add_axes_inches(fig, ratio_pad(*(
            pos_time_title[0]+w_pad*2, 
            pos_time_title[1]-h/n_rows_full,
            pos_time_title[2]/n_cols_full, h/n_rows_full)))
        return (ax_frame, ax_time_title, 
                ax_metric1, ax_metric2, ax_metric3)
    
    def define_axes_cta(self, fig, frame_dims):
        # 
        ratio_pad_fn = lambda ratio_for_padplot: lambda x,y,w,h:(
            x+(1-ratio_for_padplot)/2*w,
            y+(1-ratio_for_padplot)/2*h,
            ratio_for_padplot*w,
            ratio_for_padplot*h)
        # print('frame_dims',frame_dims)
        (x, y, w, h) = ratio_pad_fn(0.95)(*frame_dims)
        # print('(x,y,w,h)', (x,y,w,h))
        # draw the frame box
        ax_frame = self.add_axes_inches(fig, (x, y, w, h))
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        if not(self.frame_res):
            ax_frame.set_axis_off()
        ratio_pad = ratio_pad_fn(0.8)
        delta = 2 # number of subrows for a plot
        n_rows_full = 2
        n_cols_full = 1
        parms = lambda x:(x, # number of rows (correspond to height for plots)
                          x*delta, # number of subrows (correspond to height for titles)
                          (n_rows_full-x)*delta) # padding before the first row in terms of subrows
        
        n_rows, n_subrows, pad_subrows = parms(n_rows_full) 
        
        pos_time_title = (
            x, y+(n_rows_full*delta-pad_subrows-1)*h/(n_rows_full*delta), 
            w, h/(n_rows_full*delta))
        
        ax_time_title = self.add_axes_inches(fig, pos_time_title)
        
        w_pad = pos_time_title[2]/n_cols_full
        ax_acc = self.add_axes_inches(fig, ratio_pad(*(
            pos_time_title[0]+w_pad*0, 
            pos_time_title[1]-h/n_rows_full,
            pos_time_title[2]/n_cols_full, h/n_rows_full)))
        return (ax_frame, ax_time_title, ax_acc)
    
    def dict2d_to_array(self, dict2d, with_keys=False, dtype='float32', 
                        avgbysample=False):
        # Transforms a 2D dict to a np.array with natural order sorted keys
        # Can also return the np.array of keys as tuples: np.array(tuple(key1,key2))
        take_mean = lambda x: x['mean'] if avgbysample else x
        dict2d = take_mean(dict2d)
        if with_keys:
            return (np.array([[dict2d[u][v] for v in natsorted(dict2d[u].keys())] for u in natsorted(dict2d.keys())],dtype=dtype),
                    (natsorted(dict2d.keys()),natsorted(dict2d[list(dict2d.keys())[0]].keys())))
        else:
            return np.array([[dict2d[u][v] for v in natsorted(dict2d[u].keys())] for u in natsorted(dict2d.keys())],dtype=dtype)
    
    def adjust_xcoord(self, toshow, tofit):
        # Adjusts x-coords to plot the 'toshow' sequence with coords corresponding to 'tofit'
        vmin = 0 # x-axis start
        vstart = (2*len(toshow)+len(tofit)-1)/(2*len(tofit)) # x-coord first point
        vstep = (len(tofit)-1)/len(tofit) # x-coord step
        vend = vstart + len(toshow)*vstep
        return vmin, vstart, vstep, vend
    
    def dict_to_keyval(self, count_dict, avgbysample=False):
        # takes a dict and parses it in 2 sorted lists of keys and values:
        # output list[[keys,], [values,]]
        if avgbysample:
            return [list(l) for l in zip(*natsorted([
                [k,{
                    'mean':count_dict['mean'][k],
                    'std': count_dict['std'][k]}] for k in count_dict['mean'].keys()]))]
        else:
            return [list(l) for l in zip(*natsorted([[k,v] for k,v in count_dict.items()]))]
    
    def polar_bar_plot(self, ax, cnts, lbls, bottom=0.1, count_ceil=0.1, 
                       count_step=None, rmax=1, color='tab:blue', 
                       fontsize='x-small', avgbysample=False):
        if not(self.show_dist_polar):
            bottom = 0
        else:
            assert ax._projection_init[0].name=='polar'
        # if avgbysample: cnts is list[{'mean':, 'std':}] 
        # else: cnts is list[means]
        if count_step is None:
            count_step = max(rmax/10, count_ceil)
        # ax should be polar axes
        N = len(cnts)
        assert len(lbls)==N, "counts and labels should have the same length"
        if np.isnan(rmax) or rmax in [0, None, np.nan]:
            rmax = np.nanmax([bottom, 0.1])
        if avgbysample:
            cnts_val = [val['mean'] for val in cnts]
            arrCnts = np.array(cnts_val)
            arrErrs = np.array([val['std'] for val in cnts])
        else:
            cnts_val = cnts
            arrCnts = np.array(cnts)
            arrErrs = None
        
        if self.show_dist_polar:
            theta = np.arange(0,2*np.pi,2*np.pi/N)
            width = (2*np.pi)/N *0.9
        else:
            theta = np.arange(1,N+1)
            width = 0.9
        
        bars = ax.bar(theta, arrCnts, yerr=arrErrs, ecolor='gray', 
                      width=width, bottom=bottom, color=color,
                      error_kw=dict(lw=width/4, capsize=width/4, capthick=width/3))
        
        # ax.axis('off')
        if self.show_dist_polar:
            ax.spines['polar'].set_visible(False)
            ax.set_xticks([])
        # ax.set_ylim([0, bottom+rmax])
        ax.set_yticks([0, count_step*(rmax//count_step+1)+bottom], labels=[None, None])
        y_minors =[v+bottom for v in np.arange(0,rmax+count_step,count_step)]
        if len(y_minors)<2:
            y_minors += [bottom, bottom+count_step]
        fontsizes = ['xx-small', 'x-small', 'small', 
                     'medium', 'large', 'x-large', 'xx-large']
        if self.show_dist_polar:
            ax.set_yticks(y_minors, 
                          labels=['0%%', '%i%%'%int(100*count_step)]+[None]*(len(y_minors)-2)+['%i%%'%int(100*(y_minors[-1]-bottom))]*int(len(y_minors)>1),
                          minor=True, fontsize=fontsize, color='grey')
        else:
            ax.set_yticks(y_minors, 
                          labels=['%i%%'%(int(100*count_step)*i) for i in range(len(y_minors))],
                          minor=True, fontsize=fontsize)
        ax.grid(which='minor', linestyle=':')
        
        if self.show_dist_polar:
            rotations = np.rad2deg(theta)
            y0,y1 = ax.get_ylim()
            
            for x, bar, rotation, label, count in zip(theta, bars, rotations, lbls, cnts_val):
                if count>count_ceil or count==max(cnts_val) and count!=0:
                    offset = (bottom+bar.get_height()+count_step)/(y1-y0)
                    lab = ax.text(0, 0, '%s\n%i%%'%(label, count*100), transform=None, 
                         ha='center', va='center', fontsize=fontsizes[max(0,fontsizes.index(fontsize)-1)])
                    renderer = ax.figure.canvas.get_renderer()
                    bbox = lab.get_window_extent(renderer=renderer)
                    invb = ax.transData.inverted().transform([[0,0],[bbox.width,0] ])
                    lab.set_position((x,offset+(invb[1][0]-invb[0][0])/2.*2.7 ) )
                    lab.set_transform(ax.get_xaxis_transform())
                    # lab.set_rotation(rotation)
        else:
            ax.set_xticks(
                theta,
                # labels=[
                #     [None, label][int(
                #         count>count_ceil or count==max(cnts_val) and count!=0)
                #         ] for label, count in zip(lbls, cnts_val)], 
                labels=[lbls,[[None, label][int(il%5==0)] for il, label in enumerate(lbls)]][int(len(lbls)>10)], 
                fontsize=fontsize)
    
    def save_features_sequence(self, ori_seq, pred_seq, ds, n, pos, lab, 
                               legends, save_dir='prediction'):
        # legends is [(name, format of labtickels)]
        # Saves evolution in time of the physical measurements
        previous_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': 12})
        
        ori_feats = feature_transform(ori_seq, ori_seq).transpose() # WARNING: both are 2C
        pred_feats = feature_transform(pred_seq, pred_seq).transpose() # WARNING: both are 2C
        
        assert ori_feats.shape == pred_feats.shape, "Original and predicted features must have the same shape"
        
        fig = plt.figure(constrained_layout=True, figsize=(12, 3))
        subfig_w = int(ori_feats.shape[1] / 20)
        widths, heights = [[subfig_w, subfig_w, subfig_w, subfig_w], [1, 1, 1]]
        spec = gridspec.GridSpec(nrows=3, ncols=4, figure=fig, 
                                 width_ratios=widths, 
                                 height_ratios=heights)
        axes = {}
        for row in range(3):
            for col in range(4):
                axes[row, col] = fig.add_subplot(spec[row, col])
        
        i_fig = 0
        
        for i_fig in range(len(ori_feats)):
            row = i_fig % 3
            col = i_fig // 3
            x_indices = np.arange(1, ori_feats.shape[1]+1, 1)
            
            axes[row, col].plot(x_indices, ori_feats[i_fig], 'b:', x_indices, pred_feats[i_fig], 'g-')
            axes[row, col].set_title(legends[i_fig][0], loc='left', fontsize='xx-small')
            axes[row, col].set_xlabel('time', fontsize='xx-small')
            self.format_axis(axes[row, col], vmin=1, vmax=len(ori_feats[i_fig]), 
                             step=np.ceil(int(len(ori_feats.shape[1])/6)/5)*5, 
                             axis='x', ax_label='time', type_labels='int', 
                             minor=False, fontsize='xx-small')
            vmin=np.nanmin(np.hstack([ori_feats[i_fig],pred_feats[i_fig]]))
            vmax=np.nanmax(np.hstack([ori_feats[i_fig],pred_feats[i_fig]]))
            step = (vmax-vmin)/2
            self.format_axis(axes[row, col], vmin=vmin, vmax=vmax, step=step, 
                             axis='y', ax_label=None, type_labels=legends[i_fig][1], 
                             minor=False, margin=[1,1], fontsize='xx-small')
        
            axes[row, col].set_title('Prediction', color='g', loc='center', fontsize='xx-small')
            axes[row, col].set_title('Original', color='b', loc='right', fontsize='xx-small')
        
        if not(manual_mode):
            self.savefig_autodpi(os.path.join(
                self.results_dir, save_dir,
                'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_features_polar-{}.png'.format(
                    ds, n, lab, pos, self.show_dist_polar)),
                bbox_inches='tight')
            plt.close()
        else:
            plt.rcParams.update({'font.size': previous_font_size})
            return os.path.join(
                self.results_dir, save_dir,
                'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_features_polar-{}.png'.format(
                    ds, n, lab, pos, self.show_dist_polar))
        
        plt.rcParams.update({'font.size': previous_font_size})
        
        return
    
    def long_predicts(self, ds):
        if not os.path.exists(os.path.join(self.results_dir, 'prediction_long')):
            os.makedirs(os.path.join(self.results_dir, 'prediction_long'))
        if self.model_type in ['LSTM', 'LSTMS', 'GRU', 'GRUS', 'NBeats']:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(True, True))
            self.model_instance(True, True)
        else:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(False, True))
            self.model_instance(False, True)
        print('ds', ds)
        assert 'TEL' in ds, "error in the label os the data to predict"
        print('[Predicting] Long %s Samples'%ds)
        
        n = 1
        for seq, pos in tqdm(zip(self.data_pack[ds.replace('TEL','TE')][0], self.data_pack[ds.replace('TEL','TE')][1]), total = self.number_predict):
            masked_seq = deepcopy(seq)
            masked_seq[self.label_length-int(self.label_length * self.mask_ratio): , :] = 1
            seq_pred = np.expand_dims(np.zeros(seq.shape), axis=-1)
            pos_accum = np.zeros((seq_pred.shape[0]))
            data_chunked, pos_chunked = chunkdata_for_longpredict(seq, pos, self.label_length, self.mask_ratio)
            patch_nb = data_chunked.shape[0]
            idx = 0
            stride = int(seq.shape[1]*(self.mask_ratio))
            last = ((seq.shape[0]-self.label_length) % stride) # 0 if no strange patch at the end of the sequence, last resting timevalues otherwise
            
            if self.add_classifier:
                pred_class_in = {}
                pred_class_out = {}
                for clsn, clsfier in self.classifier.items():
                    pred_class_in[clsn] = np.array([0]*clsfier.nclass, np.float32)
                    pred_class_out[clsn] = np.array([0]*clsfier.nclass, np.float32)
            
            lab = self.label_from_pos(pos[['.' in str(p) for p in pos].index(True)],
                                      predictonly=True)
            
            for clsn in self.mts_metrics.keys():
                self.mts_metrics[clsn].reset()
                self.glob_mts_metrics[clsn].reset()
            
            if self.add_centercount:
                for clsn in list(self.classifier.keys()):
                    self.center_counter_pio[clsn].reset()
                    self.glob_center_counter_pio[clsn].reset()
            
            for data_patch, pos_patch in zip(data_chunked, pos_chunked):
                
                data_patch = np.expand_dims(data_patch, axis = 0)
                assert idx == pos_patch[0], "ERROR in parsing image chunks"
                mask = np.ones(data_patch.shape)
                mask[:, -int(mask.shape[1] * self.mask_ratio):, :, :] = 0
                
                if idx == 0:
                    masked = deepcopy(data_patch)
                    masked[mask==0] = 1
                    pred_img = self.model.predict([masked, mask])
                    seq_pred[idx * stride : idx * stride + self.label_length, :, :] += pred_img[0]
                    pos_accum[idx * stride : idx * stride + self.label_length] += 1
                elif idx == patch_nb-1 and last > 0:
                    masked = np.zeros(data_patch.shape)
                    masked[:, :-last, :, :] = pred_img[:, last:, :, :]
                    masked[mask==0] = 1
                    pred_img = self.model.predict([masked, mask])
                    seq_pred[-self.label_length:, :, :] += pred_img[0]
                    pos_accum[-self.label_length:] += 1
                else:
                    masked = np.zeros(data_patch.shape)
                    masked[:, :-stride, :, :] = pred_img[:, stride:, :, :]
                    masked[mask==0] = 1
                    pred_img = self.model.predict([masked, mask])
                    seq_pred[idx * stride : idx * stride + self.label_length,:,:] += pred_img[0]
                    pos_accum[idx * stride : idx * stride + self.label_length] += 1
                
                if self.add_classifier:
                    # class prediction on input and output
                    class_mask = create_class_mask(data_patch, 
                                                   self.mask_ratio, 
                                                   self.random_ratio)
                    in_class = {}
                    out_class = {}
                    for clsn, clsfier in self.classifier.items():
                        pcii = clsfier.model.predict([data_patch, class_mask])
                        pcoi = clsfier.model.predict([pred_img, class_mask])
                        pred_class_in[clsn] += pcii[0]
                        pred_class_out[clsn] += pcoi[0]
                        in_class[clsn] = clsfier.model.np_assign_class(pcii)
                        out_class[clsn] = clsfier.model.np_assign_class(pcoi)
                else:
                    in_class = {'noclassifier': ['NoClass']}
                    out_class = {'noclassifier': ['NoClass']}
                
                for clsn in list(self.classifier.keys()):
                    self.mts_metrics[clsn].update(
                        ([lab], in_class[clsn], out_class[clsn]),
                        (data_patch[:,-int(data_patch.shape[1]*self.mask_ratio):],
                         pred_img[:,-int(pred_img.shape[1]*self.mask_ratio):]))
                
                if self.add_centercount:
                    for clsn in list(self.classifier.keys()):
                        self.center_counter_pio[clsn].fit_batch(
                            [lab], (in_class[clsn], out_class[clsn]),
                            (data_patch[:,:-int(data_patch.shape[1]*self.mask_ratio)],
                             data_patch[:,-int(data_patch.shape[1]*self.mask_ratio):],
                             pred_img[:,-int(pred_img.shape[1]*self.mask_ratio):]))
                
                idx += 1
                
            seq_pred /= pos_accum[:, None, None]
            errors, errors5 = onelongbatchpredict_errors(np.expand_dims(np.expand_dims(seq, axis = 0), axis = -1), np.expand_dims(seq_pred, axis = 0),  mask.shape[1] - int(mask.shape[1] * self.mask_ratio))
            seq_pred = seq_pred.squeeze()
            
            psnr = -10.0 * np.log10(np.mean(np.square(seq_pred - seq)))
            
            kcenter_accuracy = [kcentroids_equal(to_kcentroid_seq(seq_pred.squeeze()[-(seq.squeeze().shape[0] - (mask.shape[1] - int(mask.shape[1] * self.mask_ratio))):,:], k=n_centers)[1], to_kcentroid_seq(seq.squeeze()[-(seq.squeeze().shape[0] - (mask.shape[1] - int(mask.shape[1] * self.mask_ratio))):,:], k=n_centers)[1]) for n_centers in range(1,7)]
            
            if self.add_classifier:
                class_in = {}
                class_out = {}
                makedir = {}
                for clsn, clsfier in self.classifier.items():
                    pred_class_in[clsn] = pred_class_in[clsn]/idx
                    pred_class_out[clsn] = pred_class_out[clsn]/idx
                        
                    class_in[clsn] = clsfier.model.np_assign_class([pred_class_in[clsn]])
                    class_out[clsn] = clsfier.model.np_assign_class([pred_class_out[clsn]])
                    makedir[clsn] = clsn
            else:
                class_in = {'noclassifier': ['NoClass']}
                class_out = {'noclassifier': ['NoClass']}
                makedir = {'noclassifier': 'noclassifier'}
            
            for _, mkd in makedir.items():
                if not os.path.exists(os.path.join(self.results_dir, 'prediction_long', mkd)):
                    os.makedirs(os.path.join(self.results_dir, 'prediction_long', mkd))
            
            # mts_metrics
            mts_results = {}
            glob_mts_results = {}
            for clsn in list(self.classifier.keys()):
                mts_results[clsn] = {
                    'by_1':self.mts_metrics[clsn].result_by_1(),
                    'by_1_2':self.mts_metrics[clsn].result_by_1_2(),
                    'by_1_3':self.mts_metrics[clsn].result_by_1_3(),
                    'by_1_2_3':self.mts_metrics[clsn].result_by_1_2_3()}
                # count centers for the global seq class prediction in the same way for pred (in addition to what is already done)
                self.glob_mts_metrics[clsn].reset()
                self.glob_mts_metrics[clsn].update(
                    ([lab], class_in[clsn], class_out[clsn]),
                    (np.expand_dims(seq[self.label_length-int(self.label_length*self.mask_ratio):], axis=0),
                     np.expand_dims(seq_pred[self.label_length-int(self.label_length*self.mask_ratio):], axis=0)))
                glob_mts_results[clsn] = {'by_no': self.glob_mts_metrics[clsn].result_by_no()}           
            # Center counts is freq dict {lab{class{center}}}
            # For long predicts, there may counts in all classes because each partition may provide different classes.
            if self.add_centercount:
                pio_centers = {}
                globpio_centers = {}
                for clsn in list(self.classifier.keys()):
                    pio_centers[clsn] = {
                        'cond_1':self.center_counter_pio[clsn].result_cond_1(),
                        'cond_1_2':self.center_counter_pio[clsn].result_cond_1_2(),
                        'cond_1_3':self.center_counter_pio[clsn].result_cond_1_3(),
                        'cond_1_2_3':self.center_counter_pio[clsn].result_cond_1_2_3()}
                    # count centers for the global seq class prediction in the same way for pred (in addition to what is already done)
                    self.glob_center_counter_pio[clsn].reset()
                    self.glob_center_counter_pio[clsn].fit_batch(
                        [lab], (class_in[clsn], class_out[clsn]),
                        (np.expand_dims(seq[:self.label_length-int(self.label_length*self.mask_ratio)], axis=0),
                         np.expand_dims(seq[self.label_length-int(self.label_length*self.mask_ratio):], axis=0),
                         np.expand_dims(seq_pred[self.label_length-int(self.label_length*self.mask_ratio):], axis=0)))
                    globpio_centers[clsn] = self.glob_center_counter_pio[clsn].result_cond_no()
            
            for i_img in list(makedir.keys()):
                n_srow = 1
                plot_classes = False
                if self.add_centercount:
                    n_srow += 1
                if self.add_classifier and self.classifier[i_img].noclass is not None or self.classifier[i_img].nclass>1:
                    plot_classes = True
                    n_srow += 1
                widths = [1]*n_srow+[2]*4
                ncols = len(widths)
                heights = [1]*4
                nrows = len(heights)
                figsize = (10*sum(widths),10*sum(heights))
                
                fig = plt.figure(constrained_layout=True, figsize=figsize)
                spec = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, 
                                         width_ratios=widths, 
                                         height_ratios=heights)
                axes = {}
                for row in range(nrows):
                    for col in range(ncols):
                        if self.add_centercount and col==int(plot_classes)+1 and row !=nrows-1 or plot_classes and col==1 and row in [1,2]:
                            axes[row, col] = fig.add_subplot(spec[row, col], polar=self.show_dist_polar)
                        elif not(plot_classes) or (plot_classes and col!=1):
                            axes[row, col] = fig.add_subplot(spec[row, col])
                
                # Plot first column (images of spectra)
                axes[0, 0].imshow(masked_seq.squeeze().transpose(), cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[0, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(masked_seq.squeeze()), 
                                       xlabel='time', 
                                       lambda_size=masked_seq.squeeze().shape[1], 
                                       mask=True)
                
                axes[2, 0].imshow(seq_pred.squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[2, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(seq_pred.squeeze()), 
                                       xlabel='time', 
                                       lambda_size=seq_pred.squeeze().shape[1])
                axes[1, 0].imshow(seq.squeeze().transpose() * 1., cmap = 'gist_heat', vmin = 0, vmax = 1)
                self.format_image_axes(axes[1, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)),
                                       time_max=len(seq.squeeze()), 
                                       xlabel='time', 
                                       lambda_size=seq.squeeze().shape[1])
                axes[3, 0].imshow(self.coef_diff * np.abs(seq.squeeze().transpose() * 1. - seq_pred.squeeze().transpose() * 1.), vmin = 0, vmax = 1)
                self.format_image_axes(axes[3, 0], 
                                       start_pred=int(self.label_length*(1-self.mask_ratio)), 
                                       time_max=len(seq.squeeze()),
                                       xlabel='time', 
                                       lambda_size=seq_pred.squeeze().shape[1])
                
                if self.add_classifier and not(plot_classes):
                    # should be a unique value
                    probain_unique_class = ' %s'%'\n'.join(
                        ['%s (%.3f)'%(self.classifier[i_img].classes[i], 
                        pred_class_in[i_img][i]) for i in range(self.classifier[i_img].nclass)])
                    probaout_unique_class = ' %s'%'\n'.join(
                        ['%s (%.3f)'%(self.classifier[i_img].classes[i], 
                        pred_class_out[i_img][i]) for i in range(self.classifier[i_img].nclass)])
                else:
                    probain_unique_class = ''
                    probaout_unique_class = ''
                if self.add_classifier:
                    axes[0, 0].set_title('Masked Sequence\nClass Pred:%s%s'%(
                        class_in[i_img][0], 
                        probain_unique_class))
                    axes[2, 0].set_title('Predicted Sequence PSNR=%.2f\nClass Pred:%s%s'%(
                        psnr,
                        class_out[i_img][0],
                        probaout_unique_class))
                else:
                    axes[0, 0].set_title('Masked Sequence')
                    axes[2, 0].set_title('Predicted Sequence PSNR=%.2f'%psnr)#, y =-0.01)
                axes[1, 0].set_title('Original Sequence\nWeak Label %s'%lab)
                axes[3, 0].set_title('%s x Difference' % str(self.coef_diff))#, y =-0.01)
                
                addc = 0
                # Plot second column (center clusters distributions)
                if plot_classes:
                    addc += 1
                    probain_class = {
                        self.classifier[i_img].classes[i]: pred_class_in[i_img][i] for i in range(self.classifier[i_img].nclass)}
                    probaout_class = {
                        self.classifier[i_img].classes[i]: pred_class_out[i_img][i] for i in range(self.classifier[i_img].nclass)}
                    keys_in, vals_in = self.dict_to_keyval(probain_class)
                    keys_out, vals_out = self.dict_to_keyval(probaout_class)
                    assert keys_in==keys_out
                    # print("vals_in", vals_in)
                    # print("rmax",max(vals_in+vals_out))
                    self.polar_bar_plot(axes[1, addc], vals_in, keys_in, bottom=.1,
                                        count_step=0.1, count_ceil=0, 
                                        rmax=max(vals_in+vals_out), 
                                        color='tab:purple')
                    axes[1, addc].set_title('Class Pred IN\n')
                    self.polar_bar_plot(axes[2, addc], vals_out, keys_out, bottom=.1,
                                        count_step=0.1, count_ceil=0, 
                                        rmax=max(vals_in+vals_out), 
                                        color='tab:purple')
                    axes[2, addc].set_title('Class Pred OUT\n')
                
                # Plot second/third column (center clusters distributions)
                if self.add_centercount:
                    addc += 1
                    keys_prior, vals_prior = self.dict_to_keyval(globpio_centers[i_img]['centers']['c0'])
                    keys_in, vals_in = self.dict_to_keyval(globpio_centers[i_img]['centers']['c1'])
                    keys_out, vals_out = self.dict_to_keyval(globpio_centers[i_img]['centers']['c2'])
                    assert keys_in==keys_out==keys_prior
                    self.polar_bar_plot(axes[0, addc], vals_prior, keys_prior, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[0, addc].set_title('Center distribution PRIOR\n')
                    self.polar_bar_plot(axes[1, addc], vals_in, keys_in, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[1, addc].set_title('Center distribution IN\n')
                    self.polar_bar_plot(axes[2, addc], vals_out, keys_in, bottom=.1,
                                        count_ceil=0.1, 
                                        rmax=max(vals_prior+vals_in+vals_out))
                    axes[2, addc].set_title('Center distribution OUT\n')
                    cm, (kin, kout) = self.dict2d_to_array(
                        globpio_centers[i_img]['centers']['c1c2'], 
                        with_keys=True)
                    self.plot_heatmap(
                        cm, kin, kout,
                        'Center IN VS Center OUT',
                        axes[3,addc], with_cbar=True, with_labels=False,
                        xtick_step=5, ytick_step=5, linewidths=.5,
                        vmin=0, vmax=np.max(cm))
                    axes[3, addc].set_title('Center distribution ERROR\nH(in)=%.2f H(out)=%.2f I(in;out)=%.2f\n'%(
                        globpio_centers[i_img]['info_c1c2']['entropies'][0],
                        globpio_centers[i_img]['info_c1c2']['entropies'][1],
                        globpio_centers[i_img]['info_c1c2']['mutual_info']))
                    # self.polar_bar_plot(axes[3, 1], 
                    #                     [abs(vi-vo) for vi,vo in zip(vals_in, vals_out)], 
                    #                     keys, bottom=.1, count_ceil=0.05, 
                    #                     rmax=max([
                    #                         abs(vi-vo) for vi,vo in zip(
                    #                             vals_in, vals_out)]))
                
                # Start plotting 2nd/3rd/4th column: Raw CV errors (PSNR, SSIM)
                axes[0, 1+addc].plot(range(1, len(errors[0, 0])+1), 
                                     -10*np.log10(errors[0, 0]), label='PSNR')
                self.format_axis(axes[0, 1+addc], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[0, 1+addc], vmin=0, vmax=len(errors[0, 0]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[0, 1+addc], legend_loc='upper center', fontsize='x-small')
                
                axes[2, 1+addc].plot(range(1, len(errors[1, 0])+1), 
                                     errors[1, 0], label='SSIM')
                axes[2, 1+addc].plot(range(1, len(errors[1, 0])+1), 
                                     np.ones_like(errors[1, 0]), label='best', linestyle=':', color='g')
                self.format_axis(axes[2, 1+addc], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[2, 1+addc], vmin=0, vmax=len(errors[1,0]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[2, 1+addc], legend_loc='upper center', fontsize='x-small')
                
                # Start plotting the other columns: Raw physical errors (centers assignment)
                for i in range(1,7):
                    row, col = [int((i-1)%2)*2, 2+int((i-1)//2)+addc]
                    axes[row, col].plot(range(1, len(kcenter_accuracy[i-1])+1), 
                                        kcenter_accuracy[i-1], label='%i-Center'%i)
                    axes[row, col].plot(range(1, len(kcenter_accuracy[i-1])+1), 
                                        [kinter(i) for _ in range(len(kcenter_accuracy[i-1]))], label='%i-RandomBaseground'%i, linestyle=':', color='r')
                    axes[row, col].plot(range(1, len(kcenter_accuracy[i-1])+1), 
                                        np.ones_like(kcenter_accuracy[i-1]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(kcenter_accuracy[i-1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', fontsize='x-small')
                
                # Plot 3rd column (or 2nd when no center): Avg on 5% time CV errors (PSNR, SSIM)
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=errors5[0,0], tofit=errors[0, 0])
                axes[1, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     -10*np.log10(errors5[0, 0]), label='PSNR')
                self.format_axis(axes[1, 1+addc], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[1, 1+addc], vmin=vmin, vmax=len(errors5[0, 0]), lmin=0, lmax=len(errors[0, 0]), step = 10, axis = 'x', type_labels='int', ax_label='time')
                self.set_description(axes[1, 1+addc], legend_loc='upper center', fontsize='x-small')
                axes[3, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     errors5[1, 0], label='SSIM')
                axes[3, 1+addc].plot(np.arange(vstart, vend, vstep), 
                                     np.ones_like(errors5[1, 0]), label='best', linestyle=':', color='g')
                self.format_axis(axes[3, 1+addc], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[3, 1+addc], vmin=vmin, vmax=len(errors5[1,0]), lmin=0, lmax=len(errors[1,0]), step = 10, axis = 'x', type_labels='int', ax_label='time')
                self.set_description(axes[3, 1+addc], legend_loc='upper center', fontsize='x-small')
    
                # Plot all the other columns: Avg on 5% time physical errors (centers assignment)
                for j in range(1,7):
                    row, col = [int((j-1)%2)*2+1, 2+int((j-1)//2)+addc]
                    axes[row, col].plot(*forplot_assignement_accuracy(kcenter_accuracy[j-1], bin_size=int(self.label_length * 0.05)), label='%i-Center'%j)
                    axes[row, col].plot(np.arange(.5, len(kcenter_accuracy[j-1])+.5), 
                                        [kinter(j) for _ in range(len(kcenter_accuracy[j-1]))], label='%i-RandomBaseground'%j, linestyle=':', color='r')
                    axes[row, col].plot(np.arange(.5, len(kcenter_accuracy[j-1])+.5), 
                                        np.ones_like(kcenter_accuracy[j-1]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(kcenter_accuracy[j-1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', fontsize='x-small')
    
                for i in range(2):
                    for j in range(1,5):
                        axes[2 * i, j+addc].set_title('time predictions')
                        axes[2 * i + 1, j+addc].set_title('5% slices predictions')
            
                # spec.tight_layout(fig)
                
                self.savefig_autodpi(os.path.join(
                    self.results_dir, 'prediction_long', makedir[i_img],
                    'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_polar-{}.png'.format(
                        ds, n, lab, pos[0], self.show_dist_polar)),
                    bbox_inches='tight')
                plt.close()
            
            if self.add_centercount:
                for i_img in list(makedir.keys()):
                    self.plot_1long_pred_centers(pio_centers[i_img], 
                                                 globpio_centers[i_img], lab, 
                                                 class_in[i_img][0], 
                                                 class_out[i_img][0],
                                                 save_name=os.path.join(
                                                     'prediction_long', 
                                                     makedir[i_img],
                                                     'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_detailedcentercount_polar-{}.png'.format(
                                                         ds, n, lab, pos[0], self.show_dist_polar)))
            
            for i_img in list(makedir.keys()):
                self.plot_mtsres(
                    mts_results[i_img], 
                    meta=(lab, 
                          None, 
                          None),
                    glob = glob_mts_results[i_img],
                    glob_meta=(lab, 
                               class_in[i_img][0], 
                               class_out[i_img][0]),
                    save_name=os.path.join(
                        'prediction_long', 
                        makedir[i_img],
                        'Dataset-{}_Sample-{}_Lab-{}_Pos-{}_mts_metrics.png'.format(
                            ds, n, lab, pos[0])))
            
            self.save_features_sequence(
                seq.squeeze()[-(seq.squeeze().shape[0] - (mask.shape[1] - int(mask.shape[1] * self.mask_ratio))):,:], 
                seq_pred.squeeze()[-(seq.squeeze().shape[0] - (mask.shape[1] - int(mask.shape[1] * self.mask_ratio))):,:], 
                'TEL', n, pos[0], lab, self.feat_legends, 
                save_dir='prediction_long')
            
            n += 1
            # Only create predictions for about self.number_predict images
            if n >= self.number_predict:
                break
    
    def polar_center_counts(self, ax_title, ax_prior, ax_in, ax_out, ax_err, 
                            title, prior_center, in_center, out_center):
        # plot one set of center counts  result
        # used for the 'detailed' and 'global' versions of polar center counts plots
        ax_title.text(
            ax_title.get_xlim()[1]/2,
            ax_title.get_ylim()[1]/2,
            title,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax_title.transAxes,
            fontsize='large',
            fontvariant='small-caps',
            fontweight='bold')
        ax_title.set_axis_off()
        
        keys_prior, vals_prior = self.dict_to_keyval(prior_center)
        keys, vals_in = self.dict_to_keyval(in_center)
        keys_out, vals_out = self.dict_to_keyval(out_center)
        assert keys == keys_out
        self.polar_bar_plot(ax_prior, vals_prior, keys_prior, 
                            bottom=0.1, count_ceil=0.1, 
                            rmax=max(vals_prior+vals_in+vals_out))
        ax_prior.set_title('Center distribution PRIOR\n')
        
        self.polar_bar_plot(ax_in, vals_in, keys, 
                            bottom=0.1, count_ceil=0.1, 
                            rmax=max(vals_prior+vals_in+vals_out))
        ax_in.set_title('Center distribution IN\n')
        
        self.polar_bar_plot(ax_out, vals_out, keys_out, 
                            bottom=0.1, count_ceil=0.1, 
                            rmax=max(vals_prior+vals_in+vals_out))
        ax_out.set_title('Center distribution OUT\n')
        
        self.polar_bar_plot(ax_err, 
                            [abs(vi-vo) for vi,vo in zip(vals_in, vals_out)], 
                            keys, bottom=.1, count_ceil=0.05, 
                            rmax=max([
                                abs(vi-vo) for vi,vo in zip(
                                    vals_in, vals_out)]))
        ax_err.set_title('Center distribution ERROR\n')
    
    def format_image_axes(self, ax, start_pred=180, time_max=240, lambda_size=240, xlabel='time', mask=False):
        # x settings
        xticks=[0,start_pred, time_max]
        xlabels=[None,0,time_max-start_pred]
        for side in ['top', 'right', 'left', 'bottom']:
            ax.spines[side].set_visible(False)
        ax.set_xticks(xticks,labels=xlabels, fontsize='small')
        ax.set_xlabel(xlabel, fontsize='small')
        # y settings
        spectra_settings = Mg_settings()
        lmin = spectra_settings.lambda_min
        lmax = spectra_settings.lambda_max
        ax.set_yticks([0, lambda_size], labels = [lmin, lmax], fontsize='small')
        ax.set_ylabel('lambda', fontsize='small')
        # add mask on the image as a crossed out rectangle
        if mask:
            ax.add_line(matplotlib.lines.Line2D(
                [start_pred, time_max, time_max, start_pred, start_pred, time_max, time_max, start_pred],
                [0, 0, lambda_size, 0, lambda_size, 0, lambda_size, lambda_size],
                color='b'))
    
    def plot_proba(self, p_c, ax, title=None, rmax=None,
                   bottom=0.1, count_ceil=0.1, fontsize='x-small',
                   avgbysample=False):
        # polar plot on ax the dict of probabilities p_c eventually with title
        
        if self.show_dist_polar:
            assert 'polar' in ax.name, "the ax should be polar fo the distribution"
        keys, vals = self.dict_to_keyval(p_c, avgbysample=avgbysample)
        if rmax is None:
            if avgbysample:
                rmax = max([
                    val['mean']+val['std'] for val in vals])
            else:
                rmax = np.max(vals)
        self.polar_bar_plot(ax, vals, keys, 
                            bottom=bottom, count_ceil=count_ceil, 
                            rmax=rmax, fontsize=fontsize, 
                            avgbysample=avgbysample)
        if title is not None:
            ax.set_title(title+'\n')
    
    def plot_jointproba(self, p_ab, name_dist, ax, title=None, cbar=True, 
                        with_labels=False, xtick_step=1, ytick_step=1, 
                        linewidths=1, adapt_max=True, avgbysample=False):
        # name_dist is str 'name_1 VS name_2' to name the axes
        # plot on ax the dict of probabilities p(a,b) with 'title' on ax_title
        
        cm, (k1,k2) = self.dict2d_to_array(
            p_ab, with_keys=True,dtype='float32', avgbysample=avgbysample)
        if adapt_max:
            vmin=0
            vmax=np.max(cm)
        else:
            vmin, vmax = (0,1)
        self.plot_heatmap(
            cm, k1, k2, name_dist,
            ax, with_cbar=cbar, with_labels=with_labels, vmin=vmin, vmax=vmax,
            xtick_step=xtick_step, ytick_step=ytick_step, linewidths=linewidths)
        ax.set_title(title)
    
    def vals_avgbysample(self, b_heights):
        proceed = True
        try:
            b_heights = [val['mean'] for val in b_heights]
            if np.nanmax(b_heights)==np.inf:
                print("inf value for means {}".format(b_heights))
                proceed = False
            b_errs = [val['std'] for val in b_heights]
            if np.nanmax(b_errs)==np.inf:
                print("inf value for stds {}".format(b_errs))
                proceed = False
            vmax = np.nanmax(np.asarray(b_heights)+np.asarray(b_errs))
        except:
            b_heights = b_heights
            b_errs = None
            vmax = np.nanmax(b_heights)
            if np.nanmax(b_heights)==np.inf:
                print("inf value for b_heights {}".format(b_heights))
                proceed = False
        assert proceed, "error: values with np.inf"
        return (b_heights, b_errs, vmax)
    
    def plot_measures(self, set_measures, ax, title=None, fontsize='x-small', 
                      avgbysample=False):
        # plot the set_measures on axis ax eventually with title
        # set_measure has a special structure:
        #    list[tuple(xlabel, (name1,value1), list[(name11,value11),(name12,value12)..]),
        #                               list[(name1a,value1a),(name1b,value1b)..]), ..]
        # where for each tuple in the list:
        #     (name1, value1) is the name and a measurment shared and overlapping 
        #     or bellow the following list of measurments. 11, 12 values are 
        #     overlapping. 1a, 1b are above. All mesurements are plotted as bars
        #     (name, value) can be =None, list can be empty
        # if avgbysample: values are {'mean':, 'std':} else just means floats
        color_dict = {'H': ('c',1),# for entropy cyan
                      'KL': ('y',1),# for KL-divergence yellow
                      'I': ('m',0.5),# for mutual information magenta
                      'other': ('b',1)}# for others
        default_width_bar = .8
        nb_measures = [max(1, len(m[1:][1])+len(m[1:][2])) for m in set_measures]
        idx_measures = [sum(nb_measures[:idx]) for idx in range(len(nb_measures))]
        tot_bar = sum(nb_measures)
        
        zero = {'mean':0, 'std':0} if avgbysample else 0
        # plot bars (name11,value11),(name12,value12)
        zero_height = lambda x: (x[0],x[1], [(None, zero)]*len(x[2]))# to put every (name1a,value1a) to (None, 0)
        bars = [zero_height(m[1:]) for m in set_measures]
        zero_height = lambda x: x[1:] if (x[0] is not None or len(x[1])+len(x[2])!=0) else ([(None,zero)],[]) # to deal with empty measures (space between)
        bars = [mmm for m in bars for mm in zero_height(m)for mmm in mm] 
        b_names, b_heights = zip(*bars)
        try:
            b_heights, b_errs, vmax = self.vals_avgbysample(b_heights)
        except:
            print('avgbysample', avgbysample)
            print('title', title)
            print('heights', b_heights)
            assert False
        x_pos = range(tot_bar)
        y_vmin = [1e-2, 1e-1][int(avgbysample)]
        self.plot_bars_withnames(ax, x_pos, b_heights, b_errs, b_names, 
                                 color_dict=color_dict, 
                                 width_bar=default_width_bar, y_vmin = y_vmin)
        
        # plot overlapping bars (name1,value1),(name2,value2)
        # list[(x_position, width)]
        params_over = [(idxm+(nbm-1)/2, nbm-1+default_width_bar) for nbm, idxm in zip(nb_measures, idx_measures)]
        zero_height = lambda x: x[0] if (x[0] is not None) else (None, zero) # empty bars to 0 height and no name
        bars = [zero_height(m[1:]) for m in set_measures]
        b_names, b_heights = zip(*bars)
        try:
            b_heights, b_errs, vmaxx = self.vals_avgbysample(b_heights)
        except:
            print('avgbysample', avgbysample)
            print('title', title)
            print('heights', b_heights)
            assert False
        # print('bar_names', b_names)
        x_pos, width_bar = zip(*params_over)
        vmax = np.nanmax([vmax, vmaxx])
        self.plot_bars_withnames(ax, x_pos, b_heights, b_errs, b_names, 
                                 color_dict=color_dict, width_bar=width_bar,
                                 y_vmin = y_vmin)
        
        # plot bars over (name1a,value1a),(name1b,value1b)
        bottom = [[bhb]*nbm for bhb,nbm in zip(b_heights, nb_measures)]
        bottom = [bb for b in bottom for bb in b]
        zero_height = lambda x: (x[0], [(None, zero)]*len(x[1]),x[2])# to put every (name11,value11) to (None, 0)
        bars = [zero_height(m[1:]) for m in set_measures]
        zero_height = lambda x: x[1:] if (x[0] is not None or len(x[1])+len(x[2])!=0) else ([(None,zero)],[]) # to deal with empty measures (space between)
        bars = [mmm for m in bars for mm in zero_height(m) for mmm in mm] 
        x_pos = range(tot_bar)
        b_names, b_heights = zip(*bars)
        try:
            b_heights, b_errs, vmaxx = self.vals_avgbysample(b_heights)
        except:
            print('avgbysample', avgbysample)
            print('title', title)
            print('heights', b_heights)
            assert False
        # print('bar_names', b_names)
        if b_errs is None:
            vmax = np.nanmax([vmax]+[b+h for b,h in zip(bottom, b_heights)])

        else:
            vmax = np.nanmax([vmax]+[b+h+e for b,h,e in zip(bottom, b_heights, b_errs)])
        self.plot_bars_withnames(ax, x_pos, b_heights, b_errs, b_names, 
                                 bottom=bottom, color_dict=color_dict, 
                                 width_bar=default_width_bar, y_vmin = y_vmin)
        ax.set_yticks([])
        vmax=[np.ceil(2*vmax/10**(np.floor(np.log10(vmax))))/2*10**(np.floor(np.log10(vmax))), vmax][int(vmax==0)]
        step=vmax/5
        self.format_axis(ax, vmin=0, vmax=vmax, step=step, 
                         axis='y', ax_label='bits', type_labels='%.1e', 
                         minor=False, margin=[0,0], fontsize=fontsize)#, trace=True)
        ax.set_yticks([], minor=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([v[0] for v in params_over], [m[0] for m in set_measures])
        ax.set_title(title, fontsize=fontsize)
    
    def plot_bars_withnames(self, ax, x_pos, b_heights, b_errs, b_names, 
                            bottom=None, color_dict=None, width_bar=None, 
                            fontsize='x-small', logscale=True, y_vmin=0.1):
        default_color = ('b',1)
        if color_dict is None:
            color_dict = {'other':default_color}
        none_color = lambda nname, name: color_dict[nname] if name is not None else default_color
        color_from_names = lambda x: [
            none_color(nn, n) for n in x for nn in color_dict if (n is None or all(
                nnn not in n for nnn in color_dict)) and nn==list(color_dict.keys())[0] or n is not None and nn in n]
        b_colors = color_from_names(b_names)
        b_colors, alpha = zip(*b_colors)
        alpha = np.asarray(alpha)
        for al in set(alpha):
            al_x_pos = list(np.asarray(x_pos)[alpha==al])
            al_b_heights = list(np.asarray(b_heights)[alpha==al])
            al_b_errs = None if b_errs is None else list(np.asarray(b_errs)[alpha==al])
            al_b_colors = list(np.asarray(b_colors)[alpha==al]) if b_colors is not None else default_color[0]
            al_bottom = list(np.asarray(bottom)[alpha==al]) if bottom is not None else None
            al_b_names = list(np.asarray(b_names)[alpha==al]) if b_names is not None else None
            al_width_bar = list(np.asarray(width_bar)[alpha==al]) if type(width_bar) in [tuple, list, np.ndarray] else width_bar
            bars = ax.bar(
                [x for x,h in zip(al_x_pos,al_b_heights) if h>y_vmin],
                [h for h in al_b_heights if h>y_vmin],
                yerr= None if al_b_errs is None else [e for e,h in zip(al_b_errs,al_b_heights) if h>y_vmin], 
                ecolor='gray', 
                color = None if al_b_colors is None else [c for c,h in zip(al_b_colors,al_b_heights) if h>y_vmin], 
                width = [w for w,h in zip(al_width_bar,al_b_heights) if h>y_vmin] if isinstance(al_width_bar, list) else al_width_bar,
                alpha=al, 
                bottom= None if al_bottom is None else [b for b,h in zip(al_bottom,al_b_heights) if h>y_vmin],
                error_kw=dict(
                    lw=[w/4 for w,h in zip(al_width_bar,al_b_heights) if h>y_vmin] if isinstance(al_width_bar, list) else al_width_bar/4,
                    capsize=[w/4 for w,h in zip(al_width_bar,al_b_heights) if h>y_vmin] if isinstance(al_width_bar, list) else al_width_bar/4,
                    capthick=[w/3 for w,h in zip(al_width_bar,al_b_heights) if h>y_vmin] if isinstance(al_width_bar, list) else al_width_bar/3))
            if al_bottom is None:
                al_bottom = 0
            if not(isinstance(al_bottom,list)):
                al_bottom = [al_bottom]*len(al_x_pos)
            if logscale:
                ax.set_yscale('log')
                if al_b_errs is not None and len(al_b_heights)>0:
                    assert len(al_b_heights)==len(al_b_errs)
                    min_val = np.nanmin([[h+b-e, np.nan][int(h+b-e<=0)] for h,b,e in zip(
                        al_b_heights, al_bottom, al_b_errs) if (
                            h is not None and e is not None)])
                    if np.isnan(min_val):
                        min_y = y_vmin
                    else:
                        min_y = max(y_vmin,min(y_vmin,10**np.floor(np.log(min_val/2))))
                else:
                    min_y = max(y_vmin,min(y_vmin,10**np.floor(np.log(np.nanmin(
                        np.asarray(al_b_heights)+np.asarray(al_bottom))/2))))
                ax.set_ylim([min_y, None])
            else:
                min_y = 0
            
            # position name of bars at their centers
            bar_heights = [bar.get_height() for bar in bars]
            for i_text, (x,bh,bt,bn) in enumerate(natsorted(zip(al_x_pos, al_b_heights, al_bottom, al_b_names))):
                if bt+bh>min_y:
                    ax.text(x, [bt+bh/2,np.sqrt([bt*(bh+bt),bh*min_y][int(bt==0)])][int(logscale)], bn, #transform=None, 
                         ha='center', va='center', fontsize=fontsize)
                elif bn is not None:
                    nott = "{:.0e}".format(bt+bh).split('e')
                    nott = 'e'.join([nott[0], str(int(nott[1]))])
                    ax.text(x, (ax.get_ylim()[1]+29*ax.get_ylim()[0])/30+(-1)**i_text*(ax.get_ylim()[1]+29*ax.get_ylim()[0])/50, 
                            '{}={}'.format(bn,nott), #transform=None, 
                         ha='center', va='center', fontsize=fontsize)
    
    def plot_centers(self, axs, pio_centers, stride_info=1, avgbysample=False):
        # if avgbysample: bar plots on centers will include std
        # if not(avgbysample): simple bar plot
        # axs is a tuple of of axis:
        #    (ax_pc0, ax_pc1, ax_pc2, ax_pc1c2, ax_info)
        #    ax_pc0 and ax_info can be Nones
        # pio_centers is a dict of informations to be ploted:
        #    {'centers':{'c0':{'name:value},
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':{'kl-div':float, 
        #                  'kl_proportion':float,
        #                  'entropies':(float, float)},
        #     'info_c0c2':{same with kl-div}}
        # stride_info is to control the distance between plots about the info 
        # on the data
        if avgbysample:
            try:
                rmax = max(*tuple(max([
                    pio_centers['centers'][center]['mean'][kkey]+
                    pio_centers['centers'][center]['std'][kkey] for kkey in 
                    pio_centers['centers'][center]['mean'].keys()]) for center in ['c1','c2']))
            except:
                for center in ['c1','c2']:
                    print('pio_centers means', pio_centers['centers'][center]['mean'])
                    print('pio_centers stds', pio_centers['centers'][center]['std'])
                assert False
        else:
            rmax = max(max(pio_centers['centers']['c1'].values()),
                       max(pio_centers['centers']['c2'].values()))
        if axs[0] is not None:
            assert pio_centers['centers']['c0'] is not None
            if avgbysample:
                rmax = max(rmax, max([
                    pio_centers['centers']['c0']['mean'][kkey]+
                    pio_centers['centers']['c0']['std'][kkey] for kkey in 
                    pio_centers['centers']['c0']['mean'].keys()]))
            else:
                rmax = max(rmax, max(pio_centers['centers']['c0'].values()))
            self.plot_proba(
                pio_centers['centers']['c0'], axs[0], title='prior centers', 
                rmax=rmax, bottom=0.1, count_ceil=0.1, avgbysample=avgbysample)
        self.plot_proba(
            pio_centers['centers']['c1'], axs[1], title='genuine centers', 
            rmax=rmax, bottom=0.1, count_ceil=0.1, avgbysample=avgbysample)
        self.plot_proba(
            pio_centers['centers']['c2'], axs[2], title='pred centers', 
            rmax=rmax, bottom=0.1, count_ceil=0.1, avgbysample=avgbysample)
        self.plot_jointproba(
            pio_centers['centers']['c1c2'], 'in centers VS out centers', axs[3], 
            title='centers flip', cbar=True, with_labels=False,
            xtick_step=5, ytick_step=5, linewidths=.5, adapt_max = True, 
            avgbysample=avgbysample)
        if axs[4] is not None:
            if pio_centers['info_c0c1'] is not None:
                assert pio_centers['info_c0c2'] is not None
                set_measures = [
                    ("prior", None, [('H(c0)',pio_centers['info_c0c1']['entropies'][0])],[]),
                    *(((None, None, [], []),)*stride_info),
                    ("genuine", ('H(c0)',pio_centers['info_c0c1']['entropies'][0]), [],
                     [('KL(c0||c1)',pio_centers['info_c0c1']['kl-div'])]),
                    *(((None, None, [], []),)*stride_info),
                    ("prediction", ('H(c0)',pio_centers['info_c0c2']['entropies'][0]), [],
                     [('KL(c0||c2)',pio_centers['info_c0c2']['kl-div'])])]
            else:
                set_measures = [
                    *(((None, None, [],[]),)*(3+2*stride_info))]
            set_measures.extend([
                *(((None, None, [], []),)*stride_info),
                ("prediction\nVS. genuine", ('I(c1;c2)',pio_centers['info_c1c2']['mutual_info']),
                 [('H(c1)',pio_centers['info_c1c2']['entropies'][0]),
                  ('H(c2)',pio_centers['info_c1c2']['entropies'][1])], []),
                ])
            self.plot_measures(set_measures, axs[4], title=None, 
                               avgbysample=avgbysample)
        
    def plot_classes(self, axs, pio_classes, avgbysample=False):
        # axs is a tuple of of axis:
        #    ((ax_pk1, ax_pk2, ax_pk1k2, ax_info))
        #    all can be Nones
        # pio_classes is a dict of informations to be ploted:
        #    {'classes':{'k1k2':{'name:value},
        #                'k1':, 'k2'},
        #     'infoK':{'mutual_info':float, 
        #               'mi_proportions':(float, float),
        #               'entropies':(float, float)}}
        if pio_classes['classes']['k1'] is not None:
            rmax = max(pio_classes['classes']['k1'].values())
        else:
            rmax = 0
        if pio_classes['classes']['k2'] is not None:
            rmax = max(rmax,*(pio_classes['classes']['k2'].values()))
        
        if axs[0] is not None and pio_classes['classes']['k1'] is not None:
            self.plot_proba(
                pio_classes['classes']['k1'], axs[0], title='genuine classif', 
                rmax=rmax, bottom=0.1, count_ceil=0.1)
        if axs[1] is not None and pio_classes['classes']['k2'] is not None:
            self.plot_proba(
                pio_classes['classes']['k2'], axs[1], title='output classif', 
                rmax=rmax, bottom=0.1, count_ceil=0.1)
        if axs[2] is not None and pio_classes['classes']['k1k2'] is not None:
            self.plot_jointproba(
                pio_classes['classes']['k1k2'], 'genuine classif VS pred classif', axs[2], 
                title='classes flip', cbar=True, with_labels=True,
                xtick_step=1, ytick_step=1, linewidths=1, 
                adapt_max=True)
        if axs[3] is not None and pio_classes['infoK'] is not None:
            set_measures = ([
                ("prediction\nVS. genuine", ('I(k1;k2)',pio_classes['infoK']['mutual_info']),
                 [('H(k1)',pio_classes['infoK']['entropies'][0]),
                  ('H(k2)',pio_classes['infoK']['entropies'][1])], []),
                ])
            self.plot_measures(set_measures, axs[3], title=None, 
                               avgbysample=avgbysample)
    
    def plot_title(self, ax, title, update=False, fontsize='large', posxy=None):
        if not(update):
            if posxy is None:
                ax.text(
                    ax.get_xlim()[1]/2,
                    ax.get_ylim()[1]/2,
                    title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=fontsize,
                    # fontvariant='small-caps',
                    fontweight='bold')
            else:
                ax.text(
                    *posxy,
                    title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=fontsize,
                    # fontvariant='small-caps',
                    fontweight='bold')
            ax.set_axis_off()
    
    def plot_avgsample(self, axs, pio_results, title=None):
        # plot the relevant results on the set of axis axs for the global statistics
        # or conditionned by input weak labels
        # similar to plot_1orno, but it will plot the averaged prio/input/output distributions obtained over allsamples and BN info of these averaged distributions
        # axs is a tuple (ax_frame,
        #                 ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, 
        #                 ax_center_info, ax_classes_title, ax_k1, ax_k2, 
        #                 ax_k1k2, ax_classe_info)
        # pio_results is a dict of informations to be ploted:
        #    {'centers':{'c0':{'name:value},
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':{'kl-div':float, 
        #                  'kl_proportion':float,
        #                  'entropies':(float, float)},
        #     'info_c0c2':{same with kl-div},
        #     'classes':{'k1k2':{'name:value},
        #                'k1':, 'k2'},
        #     'infoK':{'mutual_info':float, 
        #               'mi_proportions':(float, float),
        #               'entropies':(float, float)},
        #     'sample_info':{'p_c0':dict{(count,{mean, std})}},
        #                    'p_c1':dict{"},
        #                    'p_c2':dict{"},
        #                    'p_c1c2':dict{dict{"}},
        #                    'KL(c0||c1)':{"} or dict{"},
        #                    'KL(c0||c2)':{"} or dict{"},
        #                    'I(c1||c2)':{"} or dict{"},
        #                    'H(c0)':{"} or dict{"},
        #                    'H(c1)':{"} or dict{"},
        #                    'H(c2)':{"} or dict{"}}}
        count = pio_results['sample_info']['p_c1'][0]
        assert all(count==pio_results['sample_info'][ckey][0] for ckey in pio_results['sample_info'].keys()), "Count should be the same for every sample_info keys"
        axs[0].set_title(title + " "*5+"Count = {}".format(count), loc='left', fontsize='xx-large')
        self.plot_title(axs[1], "CENTERS")
        assert all(np.nanmin(pio_results['sample_info'][h][1].values())!=np.inf for h in ['H(c0)', 'H(c1)', 'H(c2)']), "entropy can't be inf for " + title
        try:
            self.plot_centers((axs[2], axs[3], axs[4], axs[5], axs[6]), 
                              {'centers': {
                                  'c0':pio_results['sample_info']['p_c0'][1],
                                  'c1':pio_results['sample_info']['p_c1'][1],
                                  'c2':pio_results['sample_info']['p_c2'][1],
                                  'c1c2':pio_results['sample_info']['p_c1c2'][1]},
                               'info_c1c2': {
                                   'mutual_info':pio_results['sample_info']['I(c1||c2)'][1], 
                                   'mi_proportions':None,
                                   'entropies':(
                                       pio_results['sample_info']['H(c1)'][1], 
                                       pio_results['sample_info']['H(c2)'][1])},
                               'info_c0c1':{
                                   'kl-div':pio_results['sample_info']['KL(c0||c1)'][1], 
                                   'kl_proportion':None,
                                   'entropies':(
                                       pio_results['sample_info']['H(c0)'][1], 
                                       pio_results['sample_info']['H(c1)'][1])},
                               'info_c0c2':{
                                   'kl-div':pio_results['sample_info']['KL(c0||c2)'][1], 
                                   'kl_proportion':None,
                                   'entropies':(
                                       pio_results['sample_info']['H(c0)'][1], 
                                       pio_results['sample_info']['H(c2)'][1])}},
                              stride_info=4, avgbysample=True)
        except:
            print('title', title)
            assert False, "failed, may be due to OOM.. retry!"
        self.plot_title(axs[7], "\nCLASSES")
        self.plot_classes((axs[8], axs[9], axs[10], axs[11]), 
                          {'classes': pio_results['classes'],
                           'infoK': pio_results['infoK']}, avgbysample=True)
    
    def plot_1orno(self, axs, pio_results, title=None):
        # plot the relevant results on the set of axis axs for the global statistics
        # or conditionned by input weak labels
        # similar to plot_avgsample, but it will plot the non-averaged prio/input/output distributions computed over all samples and BN info on these non-averaged distributions
        # axs is a tuple (ax_frame,
        #                 ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, 
        #                 ax_center_info, ax_classes_title, ax_k1, ax_k2, 
        #                 ax_k1k2, ax_classe_info)
        # pio_results is a dict of informations to be ploted:
        #    {'centers':{'c0':{'name:value},
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':{'kl-div':float, 
        #                  'kl_proportion':float,
        #                  'entropies':(float, float)},
        #     'info_c0c2':{same with kl-div},
        #     'classes':{'k1k2':{'name:value},
        #                'k1':, 'k2'},
        #     'infoK':{'mutual_info':float, 
        #               'mi_proportions':(float, float),
        #               'entropies':(float, float)}}
        count = pio_results['sample_info']['p_c1'][0]
        assert all(count==pio_results['sample_info'][ckey][0] for ckey in pio_results['sample_info'].keys()), "Count should be the same for every sample_info keys"
        axs[0].set_title(title + " "*5+"Count = {}".format(count), loc='left', fontsize='xx-large')
        # axs[0].set_title(title, loc='left', fontsize='xx-large')
        self.plot_title(axs[1], "CENTERS")
        self.plot_centers((axs[2], axs[3], axs[4], axs[5], axs[6]), 
                          {'centers': pio_results['centers'],
                           'info_c1c2': pio_results['info_c1c2'],
                           'info_c0c1': pio_results['info_c0c1'],
                           'info_c0c2': pio_results['info_c0c2']}, 
                          stride_info=5)
        self.plot_title(axs[7], "\nCLASSES")
        self.plot_classes((axs[8], axs[9], axs[10], axs[11]), 
                          {'classes': pio_results['classes'],
                           'infoK': pio_results['infoK']})  
    
    def plot_1_2(self, axs, pio_results, title=None):
        # plot the relevant results on the set of axis axs for the global statistics
        # or conditionned by input weak labels
        # axs is a tuple (ax_frame,
        #                 ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, 
        #                 ax_center_info, ax_classes_title, ax_k1, ax_k2, 
        #                 ax_k1k2, ax_classe_info)
        # pio_results is a dict of informations to be ploted:
        #    {'centers':{'c0':None,
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':None,
        #     'info_c0c2':None,
        #     'classes':{'k1k2':None,
        #                'k1':None, 'k2':},
        #     'infoK':None}
        count = pio_results['sample_info']['p_c1'][0]
        assert all(count==pio_results['sample_info'][ckey][0] for ckey in pio_results['sample_info'].keys()), "Count should be the same for every sample_info keys"
        axs[0].set_title(title + " "*5+"Count = {}".format(count), loc='left', fontsize='xx-large')
        # axs[0].set_title(title, loc='left', fontsize='xx-large')
        self.plot_title(axs[1], "CENTERS")
        # print("pio_results", pio_results)
        self.plot_centers((None, axs[3], axs[4], axs[5], axs[6]), 
                          {'centers': pio_results['centers'],
                           'info_c1c2': pio_results['info_c1c2'],
                           'info_c0c1': None,
                           'info_c0c2': None}, 
                          stride_info=5)
        self.plot_title(axs[7], "\nCLASSES")
        self.plot_classes((None, axs[9], None, None), 
                          {'classes': pio_results['classes'],
                           'infoK': pio_results['infoK']})
            
    def plot_1_3(self, axs, pio_results, title=None):
        # plot the relevant results on the set of axis axs for the global statistics
        # or conditionned by input weak labels
        # axs is a tuple (ax_frame,
        #                 ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, 
        #                 ax_center_info, ax_classes_title, ax_k1, ax_k2, 
        #                 ax_k1k2, ax_classe_info)
        # pio_results is a dict of informations to be ploted:
        #    {'centers':{'c0':None,
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':None,
        #     'info_c0c2':None,
        #     'classes':{'k1k2':None,
        #                'k1':, 'k2':None},
        #     'infoK':None}
        count = pio_results['sample_info']['p_c1'][0]
        assert all(count==pio_results['sample_info'][ckey][0] for ckey in pio_results['sample_info'].keys()), "Count should be the same for every sample_info keys"
        axs[0].set_title(title + " "*5+"Count = {}".format(count), loc='left', fontsize='xx-large')
        # axs[0].set_title(title, loc='left', fontsize='xx-large')
        self.plot_title(axs[1], "CENTERS")
        self.plot_centers((None, axs[3], axs[4], axs[5], axs[6]), 
                          {'centers': pio_results['centers'],
                           'info_c1c2': pio_results['info_c1c2'],
                           'info_c0c1': None,
                           'info_c0c2': None}, 
                          stride_info=5)
        self.plot_title(axs[7], "\nCLASSES")
        self.plot_classes((axs[8], None, None, None), 
                          {'classes': pio_results['classes'],
                           'infoK': pio_results['infoK']})
            
    def plot_1_2_3(self, axs, pio_results, title=None):
        # plot the relevant results on the set of axis axs for the global statistics
        # or conditionned by input weak labels
        # axs is a tuple (ax_frame,
        #                 ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, 
        #                 ax_center_info, ax_classes_title, ax_k1, ax_k2, 
        #                 ax_k1k2, ax_classe_info)
        # pio_results is a dict of informations to be ploted:
        #    {'centers':{'c0':None,
        #                'c1c2':, 'c1':, 'c2'},
        #     'info_c1c2':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #     'info_c0c1':None,
        #     'info_c0c2':None,
        #     'classes':None,
        #     'infoK':None}
        count = pio_results['sample_info']['p_c1'][0]
        assert all(count==pio_results['sample_info'][ckey][0] for ckey in pio_results['sample_info'].keys()), "Count should be the same for every sample_info keys"
        axs[0].set_title(title + " "*5+"Count = {}".format(count), loc='left', fontsize='xx-large')
        # axs[0].set_title(title, loc='left', fontsize='xx-large')
        self.plot_title(axs[1], "CENTERS")
        self.plot_centers((None, axs[3], axs[4], axs[5], axs[6]), 
                          {'centers': pio_results['centers'],
                           'info_c1c2': pio_results['info_c1c2'],
                           'info_c0c1': None,
                           'info_c0c2': None}, 
                          stride_info=5)
    
    def create_onefig_testmeasures(self, parms, pio_results, key_pltfn, 
                                   save_name, overwrite=False):
        # parms is tuple(labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height)
        # pio_results contains measurements defined in self.plot_clsres_one_label
        # key_pltfn is a dict {key: function}
        # where for each key from pio_results, it gives the plotting function to apply
        fname = os.path.join(self.results_dir,'.'.join(save_name))
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        fig = plt.figure(figsize=(width, height))
        # to extract the relevant info for freqdicts
        # the given key k is reached at the 3rd level of dict of dict
        # the 3rd level can be a tuple of dict or a dict
        # data in which extraction is applied:
        # dict{'centers/..': {'c0/..': data to extract}}} (the first key 'cond_?' is already selcted before applying this function)
        handle_none = lambda x, f: f(x) if x is not None else x
        tuple_extract = lambda x,k: tuple(xx[k] for xx in x) if isinstance(x,tuple) else x[k]
        # extract_1_2 = lambda x,k,t: {kk: tuple_extract(xx,k) for kk, xx in x.items()} if t==2 else tuple_extract(x,k)
        # if t==1 extract the key at 3rd level, if t==2 at 4th level
        extract_key = lambda x,k: {
            kk: handle_none(
                xx, lambda v: {kkk: handle_none(
                    xxx, lambda vv: tuple_extract(vv,k)) for kkk, xxx in v.items()}) for kk, xx in x.items()}
        
        for key, fn in key_pltfn.items():
            # create axes and plot
            fn(fig, parms, pio_results, extract_key)
        
        # fig.tight_layout()
        
        if not(manual_mode):
            self.savefig_autodpi(fname,
                bbox_inches=None, overwrite=False)
                # bbox_inches='tight')
            plt.close()
        else:
            return fname
    
    def plot_fn_no(self, fn, fig, parms):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        frame_dims = (w_sep/2,height-h_box-h_sep/2,w_box,h_box)
        axs = self.define_axes_ck(fig, frame_dims, cond=0)
        if self.backg_color:
            axs[0].set_facecolor('thistle')
        fn(axs)
    
    def plot_cond_no(self, fig, parms, pio_results, extract_key):
        self.plot_fn_no(
            lambda x: self.plot_1orno(
                x, pio_results['cond_no'], title="Global cond stats"),
            fig, parms)
    
    def plot_by_no(self, fig, parms, pio_results, extract_key):
        self.plot_fn_no(
            lambda x: self.plot_avgsample(
                x, pio_results['cond_no'], title="Global avg timestats"),
            fig, parms)
    
    def plot_fn_1_alll(self, fn, fig, parms, label):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        for i_box, lbl in enumerate(labels):
            # cond_1 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, height-h_box-h_sep/2, w_box, h_box)
            axs = self.define_axes_ck(fig, frame_dims, cond=0)
            if label == lbl and self.backg_color:
                axs[0].set_facecolor('peachpuff')
            fn(axs, lbl)
    
    def plot_cond_1_alll(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_alll(
            lambda x,y: self.plot_1orno(
                x, extract_key(pio_results['cond_1'],y), 
                title="Cond Label {}".format(y)),
            fig, parms, label)
    
    def plot_by_1_alll(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_alll(
            lambda x,y: self.plot_avgsample(
                x, extract_key(pio_results['cond_1'],y), 
                title="Avg Label {}".format(y)),
            fig, parms, label)
    
    def plot_fn_1_onel(self, fn, fig, parms):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        frame_dims = (w_sep/2, height-2*(h_box+h_sep)+h_sep/2, w_box, h_box)
        axs = self.define_axes_ck(fig, frame_dims, cond=0)
        if self.backg_color:
            axs[0].set_facecolor('peachpuff')
        fn(axs)
    
    def plot_cond_1_onel(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_onel(
            lambda x: self.plot_1orno(
                x, extract_key(pio_results['cond_1'],label), 
                title="Cond Label {}".format(label)),
            fig, parms)
    
    def plot_by_1_onel(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_onel(
            lambda x: self.plot_avgsample(
                x, extract_key(pio_results['cond_1'],label), 
                title="Avg Label {}".format(label)),
            fig, parms)
    
    def plot_cond_1(self, fig, parms, pio_results, extract_key, label):
        self.plot_cond_1_alll(fig, parms, pio_results, extract_key, label)
        self.plot_cond_1_onel(fig, parms, pio_results, extract_key, label)
    
    def plot_by_1(self, fig, parms, pio_results, extract_key, label):
        self.plot_by_1_alll(fig, parms, pio_results, extract_key, label)
        self.plot_by_1_onel(fig, parms, pio_results, extract_key, label)
    
    def plot_fn_1_2(self, fn, fig, parms, cond, none_axs=None):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        for i_box, clsn in enumerate(classes1):
            # cond_1_2 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, height-2*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            axs = self.define_axes_ck(fig, frame_dims, cond=cond)
            # (ax_frame,
            #  ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, ax_centers_info, 
            #  ax_classes_title, ax_k1, ax_k2, ax_k1k2, ax_classes_info)
            if self.backg_color:
                axs[0].set_facecolor('palegreen')
            if none_axs is not None:
                for nax in none_axs:
                    axs[nax].set_axis_off()
            fn(axs, clsn)
    
    def plot_cond_1_2(self, fig, parms, pio_results, extract_key, label):
        labres_cond_1_2 = extract_key(pio_results['cond_1_2'],label)
        self.plot_fn_1_2(
            lambda x, y: self.plot_1_2(
                x, extract_key(labres_cond_1_2,y), 
                title="Cond Input Class {}".format(y)),
            fig, parms, cond=2)
    
    def plot_by_1_2(self, fig, parms, pio_results, extract_key, label):
        labres_cond_1_2 = extract_key(pio_results['cond_1_2'],label)
        self.plot_fn_1_2(
            lambda x, y: self.plot_avgsample(
                x, extract_key(labres_cond_1_2,y), 
                title="Avg Input Class {}".format(y)),
            fig, parms, cond=0, none_axs=[8, 10, 11])
    
    def plot_fn_1_3(self, fn, fig, parms, cond, none_axs=None):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        for i_box, clsn in enumerate(classes2):
            # cond_1_3 results
            frame_dims = (w_sep/2, height-(i_box+3)*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            axs = self.define_axes_ck(fig, frame_dims, cond=cond)
            # (ax_frame,
            #  ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, ax_centers_info, 
            #  ax_classes_title, ax_k1, ax_k2, ax_k1k2, ax_classes_info)
            if self.backg_color:
                axs[0].set_facecolor('palegoldenrod')
            if none_axs is not None:
                for nax in none_axs:
                    axs[nax].set_axis_off()
            fn(axs, clsn)
    
    def plot_cond_1_3(self, fig, parms, pio_results, extract_key, label):
        labres_cond_1_3 = extract_key(pio_results['cond_1_3'],label)
        self.plot_fn_1_3(
            lambda x, y: self.plot_1_3(
                x, extract_key(labres_cond_1_3,y), 
                title="Cond Output Class {}".format(y)),
            fig, parms, cond=1)
    
    def plot_by_1_3(self, fig, parms, pio_results, extract_key, label):
        labres_cond_1_3 = extract_key(pio_results['cond_1_3'],label)
        self.plot_fn_1_3(
            lambda x, y: self.plot_avgsample(
                x, extract_key(labres_cond_1_3,y), 
                title="Avg Output Class {}".format(y)),
            fig, parms, cond=0, none_axs=[9, 10, 11])
    
    def plot_fn_1_2_3(self, fn, pio_results, label, fig, parms, extract_key, 
                      title_format, cond, none_axs=None):
        labels, classes1, classes2, w_sep, h_sep, w_box, h_box, width, height = parms
        labres_cond_1_2_3 = extract_key(pio_results['cond_1_2_3'],label)
        for i_box1, clsn1 in enumerate(classes1):
            labclres_cond_1_2_3 = extract_key(labres_cond_1_2_3,clsn1)
            for i_box2, clsn2 in enumerate(classes2):
                # cond_1_2_3 results
                frame_dims = ((i_box1+1)*(w_box + w_sep)+w_sep/2, 
                              height-(i_box2+3)*(h_box+h_sep)+h_sep/2,
                              w_box, h_box)
                axs = self.define_axes_ck(fig, frame_dims, cond=cond)
                # (ax_frame,
                #  ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, ax_centers_info, 
                #  ax_classes_title, ax_k1, ax_k2, ax_k1k2, ax_classes_info)
                if none_axs is not None:
                    for nax in none_axs:
                        axs[nax].set_axis_off()
                fn(axs, extract_key(labclres_cond_1_2_3,clsn2), 
                   title=title_format.format(clsn1, clsn2))
    
    def plot_cond_1_2_3(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_2_3(
            self.plot_1_2_3, pio_results, label, fig, parms, extract_key, 
            "Cond Input class {} Output class {}", cond=3)
    
    def plot_by_1_2_3(self, fig, parms, pio_results, extract_key, label):
        self.plot_fn_1_2_3(
            self.plot_avgsample, pio_results, label, fig, parms, extract_key, 
            "Avg Input class {} Output class {}", cond=0, none_axs=[7,8,9,10,11])
    
    def plot_clsres_one_label(self, pio_results, label, 
                              save_name='Stats.png'):
        # Plots results of prediction with measurements made by a Mg center counter
        # and a classifier. It will output 2 files, one for prior/input and output
        # distributions over all samples, and another for prior/input and output
        # distributions over corresponding times and averaged for all samples.
        # The distributions plotted in the first case are global distributions (cond_no), 
        # and distributions conditioned by for a given instance 'label' of cond_1:
        #     * each label (cond_1)
        #     * each label and class 1 (cond_1_2)
        #     * each label and class 2 (cond_1_3)
        #     * each label and classes 1 and 2 (cond_1_2_3)
        # The distributions plotted for the 2nd case are averaged distributions
        # for the prior/input and output samples, the averaging follows the same 
        # structure as the conditions used in the first case.
        # Figures are constructed following a specific layout
        # pio_results is a dict of informations to be ploted:
        # {dist:{'centers':{'c0':{'name:value},
        #                   'c1c2':, 'c1':, 'c2'},
        #        'info_c1c2':{'mutual_info':float, 
        #                     'mi_proportions':(float, float),
        #                     'entropies':(float, float)},
        #        'info_c0c1':{'kl-div':float, 
        #                     'kl_proportion':float,
        #                     'entropies':(float, float)},
        #        'info_c0c2':{same with kl-div},
        #        'classes':{'k1k2':{'name:value},
        #                   'k1':, 'k2'},
        #        'infoK':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)},
        #        'sample_info':{'p_c0':dict{(count,{mean, std})}},
        #                    'p_c1':dict{"},
        #                    'p_c2':dict{"},
        #                    'p_c1c2':dict{dict{"}},
        #                    'KL(c0||c1)':{"} or dict{"},
        #                    'KL(c0||c2)':{"} or dict{"},
        #                    'I(c1||c2)':{"} or dict{"},
        #                    'H(c0)':{"} or dict{"},
        #                    'H(c1)':{"} or dict{"},
        #                    'H(c2)':{"} or dict{"}}}
        # where 'dist' is 'cond_no', 'cond_1', or 'cond_1_2', or...
        # create the params for all the axis
        dict_temp = pio_results['cond_1']['classes']['k1k2']
        labels = list(dict_temp.keys())
        nlabels = len(labels)
        dict_temp = dict_temp[labels[0]]
        classes1 = list(dict_temp.keys()) # keys as k1
        nk1 = len(classes1)
        dict_temp = dict_temp[classes1[0]]
        classes2 = list(dict_temp.keys()) # keys are k2
        nk2 = len(classes2)
        # create figure
        w_sep = .05 # horizontal separation in %
        h_sep = .05 # vertical separation in %
        w_box = 15 # width for one box of result
        h_box = 15 # height for one box of result
        previous_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*5/6)})
        width = w_box*(1+max(nlabels, nk1))
        height = h_box*(2+nk2)
        w_sep *= width
        h_sep *= height
        w_box = (width-w_sep)/(1+max(nlabels, nk1))
        h_box = (height-h_sep)/(2+nk2)
        w_sep /= (1+max(nlabels, nk1))
        h_sep /= (2+nk2)
        # print("width height w_sep h_box w_box h_box", )
        # fig = plt.figure(figsize=(width, height))
        
        parms = (labels, classes1, classes2, 
                 w_sep, h_sep, w_box, h_box, width, height)
        
        save_name = save_name.split('.')
        save_name_globdist = save_name[:-2]+[save_name[-2]+'_GlobDist_label_{}_polar-{}'.format(
            label, self.show_dist_polar)]+save_name[-1:]
        self.create_onefig_testmeasures(
            parms, pio_results, 
            {'cond_no': lambda fig, parms, pio_results, extract_key: self.plot_cond_no(
                 fig, parms, pio_results, extract_key),
             'cond_1': lambda fig, parms, pio_results, extract_key: self.plot_cond_1(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_2': lambda fig, parms, pio_results, extract_key: self.plot_cond_1_2(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_3': lambda fig, parms, pio_results, extract_key: self.plot_cond_1_3(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_2_3': lambda fig, parms, pio_results, extract_key: self.plot_cond_1_2_3(
                 fig, parms, pio_results, extract_key, label)},
            save_name_globdist)
        
        save_name_samptimedist = save_name[:-2]+[save_name[-2]+'_AvgSample_label_{}_polar-{}'.format(
            label, self.show_dist_polar)]+save_name[-1:]
        self.create_onefig_testmeasures(
            parms, pio_results, 
            {'cond_no': lambda fig, parms, pio_results, extract_key: self.plot_by_no(
                 fig, parms, pio_results, extract_key),
             'cond_1': lambda fig, parms, pio_results, extract_key: self.plot_by_1(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_2': lambda fig, parms, pio_results, extract_key: self.plot_by_1_2(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_3': lambda fig, parms, pio_results, extract_key: self.plot_by_1_3(
                 fig, parms, pio_results, extract_key, label),
             'cond_1_2_3': lambda fig, parms, pio_results, extract_key: self.plot_by_1_2_3(
                 fig, parms, pio_results, extract_key, label)},
            save_name_samptimedist)
        
        plt.rcParams.update({'font.size': int(previous_font_size)})
    
    def define_axes_ck(self, fig, frame_dims, cond=0):
        # cond = 0 for full version
        # = 1 for cond_1_3
        # = 2 for cond_1_2
        # = 3 for cond_1_2_3
        ratio_pad_fn = lambda ratio_for_padplot: lambda x,y,w,h:(
            x+(1-ratio_for_padplot)/2*w,
            y+(1-ratio_for_padplot)/2*h,
            ratio_for_padplot*w,
            ratio_for_padplot*h)
        # print('frame_dims',frame_dims)
        (x, y, w, h) = ratio_pad_fn(0.95)(*frame_dims)
        # print('(x,y,w,h)', (x,y,w,h))
        # draw the frame box
        ax_frame = self.add_axes_inches(fig, (x, y, w, h))
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        if not(self.frame_res):
            ax_frame.set_axis_off()
        ratio_pad = ratio_pad_fn(0.8)
        delta = 3 # number of subrows for a plot
        n_rows_full = 4
        n_cols_full = 4
        parms = lambda x:(x, # number of rows (correspond to height for plots)
                          x*delta, # number of subrows (correspond to height for titles)
                          (n_rows_full-x)*delta) # padding before the first row in terms of subrows
        # if cond<3:
        n_rows, n_subrows, pad_subrows = parms(n_rows_full) 
        # elif cond==3:
        #     n_rows, n_subrows, pad_subrows = parms(n_rows_full-1-2/3) 
        # else:
        #     n_rows, n_subrows, pad_subrows = parms(n_rows_full-1) 
        # print('rows, subrows, pad_sr',(n_rows, n_subrows, pad_subrows))
        pos_center_title = (
            x, y+(n_rows_full*delta-pad_subrows-1)*h/(n_rows_full*delta), 
            w, h/(n_rows_full*delta))
        # print('pos_center_title',pos_center_title)
        
        ax_center_title = self.add_axes_inches(fig, pos_center_title)
        w_pad = pos_center_title[2]/n_cols_full
        if cond==0:
            ax_c0 = self.add_axes_inches(fig, ratio_pad(*(
                pos_center_title[0], pos_center_title[1]-h/n_rows_full,
                w_pad, h/n_rows_full)), projection=[None,'polar'][int(self.show_dist_polar)])
        else:
            ax_c0 = None
        w_stride = 0
        w_col = pos_center_title[2]/n_cols_full
        ax_c1 = self.add_axes_inches(fig, ratio_pad(*(
            pos_center_title[0]+w_pad, pos_center_title[1]-h/n_rows_full,
            w_col, h/n_rows_full)), projection=[None,'polar'][int(self.show_dist_polar)])
        # print('ax_c1', ratio_pad(*(
        #     pos_center_title[0]+w_pad, pos_center_title[1]-h/n_rows_full,
        #     w_col, h/n_rows_full)))
        ax_c2 = self.add_axes_inches(fig, ratio_pad(*(
            pos_center_title[0]+w_pad+w_col+w_stride,
            pos_center_title[1]-h/n_rows_full,
            w_col, h/n_rows_full)), projection=[None,'polar'][int(self.show_dist_polar)])
        # print('ax_c2',ratio_pad(*(
        #     pos_center_title[0]+w_pad+w_col+w_stride,
        #     pos_center_title[1]-h/n_rows_full,
        #     w_col, h/n_rows_full)))
        ax_c1c2 = self.add_axes_inches(fig, ratio_pad(*(
            pos_center_title[0]+w_pad+w_col+2*w_stride+w_col,
            pos_center_title[1]-h/n_rows_full,
            w_col, h/n_rows_full)))
        # print('ax_c1c2',ratio_pad(*(
        #     pos_center_title[0]+w_pad+w_col+2*w_stride+w_col,
        #     pos_center_title[1]-h/n_rows_full,
        #     w_col, h/n_rows_full)))
        ax_centers_info = self.add_axes_inches(fig, ratio_pad(*(
            pos_center_title[0]+w_pad/3,
            pos_center_title[1]-2*h/n_rows_full,
            pos_center_title[2]-2*w_pad/3, h/n_rows_full)))
        # print('ax_cI',ratio_pad(*(
        #     pos_center_title[0]+w_pad/3,
        #     pos_center_title[1]-2*h/n_rows_full,
        #     pos_center_title[2]-2*w_pad/3, h/n_rows_full)))
        if cond<3:
            ax_classes_title = self.add_axes_inches(fig, ratio_pad(*(
                pos_center_title[0],
                pos_center_title[1]-2*h/n_rows_full-h/(n_rows_full*delta)/2,
                pos_center_title[2], pos_center_title[3]/2)))
            # print('ax_clt',ratio_pad(*(
            #     pos_center_title[0],
            #     pos_center_title[1]-2*h/n_rows_full-2*h/(n_rows_full*delta),
            #     pos_center_title[2], pos_center_title[3])))
            if cond in [0,1]:
                # if cond == 0:
                w_pad = 0
                # elif cond == 1:
                #     w_pad = (pos_center_title[0]-pos_center_title[2]/n_cols_full)/2
                ax_k1 = self.add_axes_inches(fig, ratio_pad(*(
                    pos_center_title[0]+w_pad, 
                    pos_center_title[1]-3*h/n_rows_full-h/(n_rows_full*delta),
                    pos_center_title[2]/n_cols_full, h/n_rows_full)), projection=[None,'polar'][int(self.show_dist_polar)])
                # print('ax_k1',ratio_pad(*(
                #     pos_center_title[0]+w_pad, 
                #     pos_center_title[1]-3*h/n_rows_full-2*h/(n_rows_full*delta),
                #     pos_center_title[2]/n_cols_full, h/n_rows_full)))
            else:
                ax_k1 = None
            if cond in [0,2]:
                # if cond == 0:
                w_pad = pos_center_title[2]/n_cols_full
                # elif cond == 2:
                #     w_pad = (pos_center_title[0]-pos_center_title[2]/n_cols_full)/2
                ax_k2 = self.add_axes_inches(fig, ratio_pad(*(
                    pos_center_title[0]+w_pad, 
                    pos_center_title[1]-3*h/n_rows_full-h/(n_rows_full*delta),
                    pos_center_title[2]/n_cols_full, h/n_rows_full)), projection=[None,'polar'][int(self.show_dist_polar)])
                # print('ax_k2',ratio_pad(*(
                #     pos_center_title[0]+w_pad, 
                #     pos_center_title[1]-3*h/n_rows_full-2*h/(n_rows_full*delta),
                #     pos_center_title[2]/n_cols_full, h/n_rows_full)))
            else:
                ax_k2 = None
            if cond == 0:
                ax_k1k2 = self.add_axes_inches(fig, ratio_pad_fn(0.70)(*(
                    pos_center_title[0]+w_pad*2, 
                    pos_center_title[1]-3*h/n_rows_full-h/(n_rows_full*delta),
                    pos_center_title[2]/n_cols_full, h/n_rows_full)))
                # print('ax_k1k2',ratio_pad(*(
                #     pos_center_title[0]+w_pad*2, 
                #     pos_center_title[1]-3*h/n_rows_full-2*h/(n_rows_full*delta),
                #     pos_center_title[2]/n_cols_full, h/n_rows_full)))
                ax_classes_info = self.add_axes_inches(fig, ratio_pad_fn(0.70)(*(
                    pos_center_title[0]+w_pad*3, 
                    pos_center_title[1]-3*h/n_rows_full-h/(n_rows_full*delta),
                    pos_center_title[2]/n_cols_full, h/n_rows_full)))
                # print('ax_kI',ratio_pad(*(
                #     pos_center_title[0]+w_pad*3, 
                #     pos_center_title[1]-3*h/n_rows_full-2*h/(n_rows_full*delta),
                #     pos_center_title[2]/n_cols_full, h/n_rows_full)))
            else:
                ax_k1k2 = None
                ax_classes_info = None
        else:
            ax_classes_title = None
            ax_k1 = None
            ax_k2 = None
            ax_k1k2 = None
            ax_classes_info = None
        return (ax_frame,
                ax_center_title, ax_c0, ax_c1, ax_c2, ax_c1c2, ax_centers_info, 
                ax_classes_title, ax_k1, ax_k2, ax_k1k2, ax_classes_info)
    
    def plot_frame(self, fig, dims):
        ax = self.add_axes_inches(fig, dims)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_1simp_pred_centers(self, pio_results, label, 
                                 glob_class_in, glob_class_out,
                                 save_name='Centercount.png'):
        # Plots center results of a simple prediction by global distributions
        # and distributions conditioned by a given instance 'label' of cond_1:
        #     * each class 1 (cond_1_2)
        #     * each class 2 (cond_1_3)
        #     * each classes 1 and 2 (cond_1_2_3)
        # using a specific layout
        # pio_results ise dics of information to be ploted:
        # {dist:{'centers':{'c0':{'name:value},
        #                   'c1c2':, 'c1':, 'c2'},
        #        'info_c1c2':{'mutual_info':float, 
        #                     'mi_proportions':(float, float),
        #                     'entropies':(float, float)},
        #        'info_c0c1':{'kl-div':float, 
        #                     'kl_proportion':float,
        #                     'entropies':(float, float)},
        #        'info_c0c2':{same with kl-div},
        #        'classes':{'k1k2':{'name:value},
        #                   'k1':, 'k2'},
        #        'infoK':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)}}}
        # where 'dist' is 'cond_no', 'cond_1', or 'cond_1_2', or...
        # create the params for all the axis
        # labels = [label]
        # nlabels = len(labels)
        # dict_temp = pio_results['classes']['k1k2']
        # classes1 = natsorted(list(dict_temp.keys())) # keys as k1
        # nk1 = len(classes1)
        # dict_temp = dict_temp[classes1[0]]
        # classes2 = natsorted(list(dict_temp.keys())) # keys are k2
        # nk2 = len(classes2)
        # create figure
        w_sep = .05 # horizontal separation in %
        h_sep = .05 # vertical separation in %
        w_box = 15 # width for one box of result
        h_box = 15 # height for one box of result
        previous_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*5/6)})
        width = w_box#*(1+max(nlabels, nk1))
        height = h_box#*(2+nk2)
        w_sep *= width
        h_sep *= height
        w_box = width-w_sep#(width-w_sep)/(1+max(nlabels, nk1))
        h_box = height-h_sep#(height-h_sep)/(2+nk2)
        fig = plt.figure(figsize=(width, height))
        # create axes and plot
        # global values:
        # frame_dims = (0,height-h_box,w_box,h_box)
        # axs = self.define_axes_ck(fig, frame_dims, cond=0)
        # self.plot_1orno(
        #     axs, globpio_results, 
        #     title="Global prediction ClassIN: {} ClassOUT: {}".format(
        #         glob_class_in, glob_class_out))
        
        # to extract the relevant info for freqdicts
        # the given key k is reached at the 3rd level of dict of dict
        # the 3rd level can be a tuple of dict or a dict
        handle_none = lambda x, f: f(x) if x is not None else x
        tuple_extract = lambda x,k: tuple(xx[k] for xx in x) if isinstance(x,tuple) else x[k]
        extract_key = lambda x,k: {
            kk: handle_none(
                xx, lambda v: {kkk: handle_none(
                    xxx, lambda vv: tuple_extract(vv,k)) for kkk, xxx in v.items()}) for kk, xx in x.items()}
        
        # for i_box, lbl in enumerate(labels):
        #     # cond_1 results
        #     frame_dims = ((i_box+1)*w_box + w_sep, height-h_box, w_box, h_box)
        #     axs = self.define_axes_ck(fig, frame_dims, cond=1)
        #     self.plot_1orno(axs, extract_key(pio_results['cond_1'],lbl), 
        #                          title="Label {}".format(lbl))
        
        frame_dims = (w_sep/2, h_sep/2, w_box, h_box)
        axs = self.define_axes_ck(fig, frame_dims, cond=0)
        if self.backg_color:
            axs[0].set_facecolor('thistle')
        self.plot_1orno(axs, extract_key(pio_results,label), 
                             title="Marginals Label {}".format(label))
        
        # labres_cond_1_2 = extract_key(pio_results['cond_1_2'],label)
        # for i_box, clsn in enumerate(classes1):
        #     # cond_1_2 results
        #     frame_dims = ((i_box+1)*w_box + w_sep, height-2*h_box-h_sep,
        #                   w_box, h_box)
        #     axs = self.define_axes_ck(fig, frame_dims, cond=2)
        #     self.plot_cond_1_2(axs, extract_key(labres_cond_1_2,clsn), 
        #                          title="Input class {}".format(clsn))
        
        # labres_cond_1_3 = extract_key(pio_results['cond_1_3'],label)
        # for i_box, clsn in enumerate(classes2):
        #     # cond_1_3 results
        #     frame_dims = (0, height-(i_box+3)*h_box-h_sep,
        #                   w_box, h_box)
        #     axs = self.define_axes_ck(fig, frame_dims, cond=1)
        #     self.plot_cond_1_3(axs, extract_key(labres_cond_1_3,clsn), 
        #                          title="Output class {}".format(clsn))
        
        # labres_cond_1_2_3 = extract_key(pio_results['cond_1_2_3'],label)
        # for i_box1, clsn1 in enumerate(classes1):
        #     labclres_cond_1_2_3 = extract_key(labres_cond_1_2_3,clsn1)
        #     for i_box2, clsn2 in enumerate(classes2):
        #         # cond_1_2_3 results
        #         frame_dims = ((i_box1+1)*w_box + w_sep, 
        #                       height-(i_box2+3)*h_box-h_sep,
        #                       w_box, h_box)
        #         axs = self.define_axes_ck(fig, frame_dims, cond=3)
        #         self.plot_cond_1_2_3(
        #             axs, extract_key(labclres_cond_1_2_3,clsn2), 
        #             title="Input class {} Output class {}".format(clsn1, clsn2))
        
        # fig.tight_layout()
        
        self.savefig_autodpi(os.path.join(
            self.results_dir, save_name),
            bbox_inches='tight')
        plt.close()
        
        plt.rcParams.update({'font.size': int(previous_font_size)})
    
    def plot_1long_pred_centers(self, pio_results, globpio_results, label, 
                                glob_class_in, glob_class_out,
                                save_name='Centercount.png'):
        # Plots center results of a long prediction by global distributions (cond_no),
        #     * by global prediction on averaging classes of all slides
        #     * by marginal distributions on the slides results
        # and distributions conditioned by a given instance 'label' of cond_1:
        #     * each class 1 (cond_1_2)
        #     * each class 2 (cond_1_3)
        #     * each classes 1 and 2 (cond_1_2_3)
        # using a specific layout
        # pio_results/globpio_results are dicts of informations to be ploted:
        # {dist:{'centers':{'c0':{'name:value},
        #                   'c1c2':, 'c1':, 'c2'},
        #        'info_c1c2':{'mutual_info':float, 
        #                     'mi_proportions':(float, float),
        #                     'entropies':(float, float)},
        #        'info_c0c1':{'kl-div':float, 
        #                     'kl_proportion':float,
        #                     'entropies':(float, float)},
        #        'info_c0c2':{same with kl-div},
        #        'classes':{'k1k2':{'name:value},
        #                   'k1':, 'k2'},
        #        'infoK':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)}}}
        # where 'dist' is 'cond_no', 'cond_1', or 'cond_1_2', or...
        # create the params for all the axis
        labels = [label]
        nlabels = len(labels)
        dict_temp = globpio_results['classes']['k1k2']
        classes1 = natsorted(list(dict_temp.keys())) # keys as k1
        nk1 = len(classes1)
        dict_temp = dict_temp[classes1[0]]
        classes2 = natsorted(list(dict_temp.keys())) # keys are k2
        nk2 = len(classes2)
        # create figure
        w_sep = .05 # horizontal separation in %
        h_sep = .05 # vertical separation in 4
        w_box = 15 # width for one box of result
        h_box = 15 # height for one box of result
        previous_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': int(previous_font_size*w_box/30*5/6)})
        width = w_box*(1+max(nlabels, nk1))
        height = h_box*(2+nk2)
        w_sep *= width
        h_sep *= height
        w_box = (width-w_sep)/(1+max(nlabels, nk1))
        h_box = (height-h_sep)/(2+nk2)
        w_sep /= (1+max(nlabels, nk1))
        h_sep /= (2+nk2)
        fig = plt.figure(figsize=(width, height))
        print('def_fig size {}'.format(fig.get_size_inches()))
        print('def_fig dpi {}'.format(fig.dpi))
        # create axes and plot
        # global values:
        frame_dims = (w_sep/2,height-h_box-h_sep/2,w_box,h_box)
        axs = self.define_axes_ck(fig, frame_dims, cond=0)
        if self.backg_color:
            axs[0].set_facecolor('thistle')
        self.plot_1orno(
            axs, globpio_results, 
            title="Global prediction ClassIN: {} ClassOUT: {}".format(
                glob_class_in, glob_class_out))
        
        # to extract the relevant info for freqdicts
        # the given key k is reached at the 3rd level of dict of dict
        # the 3rd level can be a tuple of dict or a dict
        handle_none = lambda x, f: f(x) if x is not None else x
        tuple_extract = lambda x,k: tuple(xx[k] for xx in x) if isinstance(x,tuple) else x[k]
        extract_key = lambda x,k: {
            kk: handle_none(
                xx, lambda v: {kkk: handle_none(
                    xxx, lambda vv: tuple_extract(vv,k)) for kkk, xxx in v.items()}) for kk, xx in x.items()}
        
        # for i_box, lbl in enumerate(labels):
        #     # cond_1 results
        #     frame_dims = ((i_box+1)*w_box + w_sep, height-h_box, w_box, h_box)
        #     axs = self.define_axes_ck(fig, frame_dims, cond=1)
        #     self.plot_1orno(axs, extract_key(pio_results['cond_1'],lbl), 
        #                          title="Label {}".format(lbl))
        
        frame_dims = (w_sep/2, height-2*(h_box+h_sep)+h_sep/2, w_box, h_box)
        axs = self.define_axes_ck(fig, frame_dims, cond=0)
        if self.backg_color:
            axs[0].set_facecolor('peachpuff')
        self.plot_1orno(axs, extract_key(pio_results['cond_1'],label), 
                             title="Global marginals Label {}".format(label))
        
        labres_cond_1_2 = extract_key(pio_results['cond_1_2'],label)
        for i_box, clsn in enumerate(classes1):
            # cond_1_2 results
            frame_dims = ((i_box+1)*(w_box + w_sep)+w_sep/2, height-2*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            axs = self.define_axes_ck(fig, frame_dims, cond=2)
            if self.backg_color:
                axs[0].set_facecolor('palegreen')
            self.plot_1_2(axs, extract_key(labres_cond_1_2,clsn), 
                          title="Slides Input class {}".format(clsn))
        
        labres_cond_1_3 = extract_key(pio_results['cond_1_3'],label)
        for i_box, clsn in enumerate(classes2):
            # cond_1_3 results
            frame_dims = (w_sep/2, height-(i_box+3)*(h_box+h_sep)+h_sep/2,
                          w_box, h_box)
            axs = self.define_axes_ck(fig, frame_dims, cond=1)
            if self.backg_color:
                axs[0].set_facecolor('palegoldenrod')
            self.plot_1_3(axs, extract_key(labres_cond_1_3,clsn), 
                          title="Slides Output class {}".format(clsn))
        
        labres_cond_1_2_3 = extract_key(pio_results['cond_1_2_3'],label)
        for i_box1, clsn1 in enumerate(classes1):
            labclres_cond_1_2_3 = extract_key(labres_cond_1_2_3,clsn1)
            for i_box2, clsn2 in enumerate(classes2):
                # cond_1_2_3 results
                frame_dims = ((i_box1+1)*(w_box + w_sep)+w_sep/2, 
                              height-(i_box2+3)*(h_box+h_sep)+h_sep/2,
                              w_box, h_box)
                axs = self.define_axes_ck(fig, frame_dims, cond=3)
                self.plot_1_2_3(
                    axs, extract_key(labclres_cond_1_2_3,clsn2), 
                    title="Slides Input class {} Output class {}".format(clsn1, clsn2))
        
        # fig.tight_layout()
        print('end_fig size {}'.format(fig.get_size_inches()))
        print('end_fig dpi {}'.format(fig.dpi))

        self.savefig_autodpi(os.path.join(
            self.results_dir, save_name),
            bbox_inches='tight')
        plt.close()
        
        plt.rcParams.update({'font.size': int(previous_font_size)})
    
    def savefig_autodpi(self, fname, bbox_inches='tight', max_pix=2**16, 
                        dpi_request=200, overwrite=True):
        if self.fig_form is not None:
            fname = fname.split('.')
            if fname[-1] in [v for v in ['png', 'ps', 'pdf', 'svg'] if v!=self.fig_form]:
                fname = fname[:-1]+[self.fig_form]
            elif fname[-1]!=self.fig_form:
                fname = fname+[self.fig_form]
        fname = '.'.join(fname)
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        gc.collect()
        time.sleep(2)
        fig = plt.gcf()
        try:
            # print('fig_dpi', fig.dpi)
            # print('fig_inches', fig.get_size_inches())
            # bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # print('fig_pixels', (bbox.width*fig.dpi, bbox.height*fig.dpi))
            # print('May adapt with fig_dpi', (2**16-1)/max(fig.get_size_inches()))
            size_pix = max(fig.get_size_inches()*dpi_request)
            if size_pix>max_pix:
                dpi_request = int(dpi_request*(max_pix-1)/size_pix)
            plt.savefig(
                fname,
                bbox_inches=bbox_inches,
                dpi = dpi_request)
        except:
            try:
                try:
                    new_dpi_request = 0.9*dpi_request
                    print('name', fname)
                    print('1-Adapt dpi to be able to save current figure')
                    print('current figsize {} (may be larger if some plots were made out of the frame)'.format(fig.get_size_inches()))
                    print('current dpi:  {}'.format(fig.dpi))
                    print('used dpi:  {}'.format(new_dpi_request))
                    assert new_dpi_request>25, "dpi too low"
                    time.sleep(1)
                    self.savefig_autodpi(fname, bbox_inches=bbox_inches, 
                                         max_pix=max_pix, 
                                         dpi_request=new_dpi_request)
                except:
                    print('name', fname)
                    print('2-Adapt dpi to be able to save current figure')
                    print('current figsize {} (may be larger if some plots were made out of the frame)'.format(fig.get_size_inches()))
                    print('current dpi:  {}'.format(fig.dpi))
                    print('used dpi:  {}'.format(min(fig.dpi, max(int(dpi_request/2),100))))
                    time.sleep(1)
                    size_pix = max(fig.get_size_inches()*dpi_request)
                    if size_pix>max_pix:
                        dpi_request = int(dpi_request*(max_pix-1)/size_pix)
                    plt.savefig(
                        fname,
                        bbox_inches=bbox_inches,
                        dpi = min(fig.dpi, max(int(dpi_request/2),100)))
            except:
                time.sleep(1)
                print('name', fname)
                print('3-Adapt dpi to be able to save current figure')
                print('current dpi:  {}'.format(fig.dpi))
                # print('fig_inches', fig.get_size_inches())
                # bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                bbox = fig.get_tightbbox(fig.canvas.get_renderer())
                bbox = [max(bbox.x1,bbox.xmax)-min(bbox.x0,bbox.xmin),
                        max(bbox.y1,bbox.ymax)-min(bbox.y0,bbox.ymin)]
                dpi = int((max_pix-1)/max(bbox))
                print('used dpi:  {}'.format(dpi))
                # print('fig_pixels', (bbox.width*fig.dpi, bbox.height*fig.dpi))
                #fig = plt.gcf()
                # May be useful to use:
                # fig.get_tightbbox(fig.canvas.get_renderer()) for the right sizes instead of get_size_inches()
                size_pix = max(fig.get_size_inches()*dpi_request)
                if size_pix>max_pix:
                    dpi = min(dpi,int(dpi_request*(max_pix-1)/size_pix))
                plt.savefig(
                    fname,
                    bbox_inches=bbox_inches,
                    dpi = dpi)
    
    def plotdetailed_center_counts(self, prior_centers, in_centers, out_centers, 
                                   save_name='Centercount.png'):
        labs = list(in_centers.keys())
        clas = list(in_centers[labs[0]].keys())
        # cent = list(in_centers[labs[0]][clas[0]].keys())
        fig = plt.figure(constrained_layout=True, 
                         figsize=(40*len(clas), 11*len(labs)))
        widths, heights = [[1]*4*len(clas), [0.1,1]*len(labs)]
        spec = gridspec.GridSpec(nrows=2*len(labs), ncols=4*len(clas), figure=fig, 
                                 width_ratios=widths, 
                                 height_ratios=heights)
        axes = {}
        for row in range(len(labs)):
            for col in range(len(clas)):
                axes[2*row, 3*col] = fig.add_subplot(spec[2*row, 2*col:2*col+3])
                axes[2*row+1, 3*col] = fig.add_subplot(spec[2*row+1, 3*col], polar=self.show_dist_polar)
                axes[2*row+1, 3*col+1] = fig.add_subplot(spec[2*row+1, 3*col+1], polar=self.show_dist_polar)
                axes[2*row+1, 3*col+2] = fig.add_subplot(spec[2*row+1, 3*col+2], polar=self.show_dist_polar)
                axes[2*row+1, 3*col+3] = fig.add_subplot(spec[2*row+1, 3*col+3], polar=self.show_dist_polar)
                self.polar_center_counts( axes[2*row, 3*col], 
                                    axes[2*row+1, 3*col],
                                    axes[2*row+1, 3*col+1], 
                                    axes[2*row+1, 3*col+2], 
                                    axes[2*row+1, 3*col+3], 
                                    'Label %s  & Precicted Class %s'%(labs[row], clas[col]), 
                                    prior_centers[labs[row]][clas[col]], 
                                    in_centers[labs[row]][clas[col]], 
                                    out_centers[labs[row]][clas[col]])
        
        spec.tight_layout(fig)
        
        if not(manual_mode):
            self.savefig_autodpi(os.path.join(
                self.results_dir,save_name),
                bbox_inches=None)
                # bbox_inches='tight')
            plt.close()
        else:
            return os.path.join(
                self.results_dir,save_name)
        
    def plotglobal_center_counts(self, prior_centers, in_centers, out_centers, 
                                 save_name='Centercount.png'):
        # Plots center results of a long prediction by global distribution (cond_no),
        #     * by global prediction on averaging classes of all slides
        #     * by marginal distributions on the slides results
        # using a specific layout
        # pio_results/globpio_results are dicts of informations to be ploted:
        # {dist:{'centers':{'c0':{'name:value},
        #                   'c1c2':, 'c1':, 'c2'},
        #        'info_c1c2':{'mutual_info':float, 
        #                     'mi_proportions':(float, float),
        #                     'entropies':(float, float)},
        #        'info_c0c1':{'kl-div':float, 
        #                     'kl_proportion':float,
        #                     'entropies':(float, float)},
        #        'info_c0c2':{same with kl-div},
        #        'classes':{'k1k2':{'name:value},
        #                   'k1':, 'k2'},
        #        'infoK':{'mutual_info':float, 
        #                  'mi_proportions':(float, float),
        #                  'entropies':(float, float)}}}
        # where 'dist' is 'cond_no', 'cond_1', or 'cond_1_2', or...
        # create the params for all the axis      
        labs = list(in_centers.keys())
        clas = list(in_centers[labs[0]].keys())
        cent = list(in_centers[labs[0]][clas[0]].keys())
        prior_gcent = {c: np.sum([[prior_centers[l][cl][c] for cl in clas] for l in labs]) for c in cent}
        in_gcent = {c: np.sum([[in_centers[l][cl][c] for cl in clas] for l in labs]) for c in cent}
        out_gcent = {c: np.sum([[out_centers[l][cl][c] for cl in clas] for l in labs]) for c in cent}
        # cent = list(in_centers[labs[0]][clas[0]].keys())
        fig = plt.figure(constrained_layout=True, 
                         figsize=(40, 11))
        widths, heights = [[1]*4, [.1,1]]
        spec = gridspec.GridSpec(nrows=2, ncols=4, figure=fig, 
                                 width_ratios=widths, 
                                 height_ratios=heights)
        axes = {}
        
        axes[0, 0] = fig.add_subplot(spec[0, :])
        axes[1, 0] = fig.add_subplot(spec[1, 0], polar=self.show_dist_polar)
        axes[1, 1] = fig.add_subplot(spec[1, 1], polar=self.show_dist_polar)
        axes[1, 2] = fig.add_subplot(spec[1, 2], polar=self.show_dist_polar)
        axes[1, 3] = fig.add_subplot(spec[1, 3], polar=self.show_dist_polar)
        self.polar_center_counts(axes[0, 0],
                                 axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3],
                                 'Global center distributions',
                                 prior_gcent, in_gcent, out_gcent)
        
        spec.tight_layout(fig)
        
        if not(manual_mode):
            self.savefig_autodpi(os.path.join(
                self.results_dir,save_name),
                bbox_inches=None)
                # bbox_inches='tight')
            plt.close()
        else:
            return os.path.join(
                self.results_dir,save_name)
    
    def plot_classmetrics(self, class_results_in, class_results_out,
                          class_IOchange, class_TIchange, class_TOchange,
                          confusion_classes, adapt_max=True, 
                          xtick_step=1, ytick_step=1,
                          save_name='ConfusionM_Data.png',
                          overwrite=True):
        fname = os.path.join(self.results_dir,save_name)
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        # Plot and save results
        # number of metrics
        nm = len(class_results_in)
        ncols = np.lcm(3,nm)
        widths, heights = [[3/ncols]*ncols, [.3,.07*(3+len(confusion_classes)),1]]
        fig = plt.figure(constrained_layout=True, 
                         figsize=(int(10*sum(widths)), int(10*sum(heights))))
        spec = gridspec.GridSpec(nrows=len(heights), ncols=len(widths), figure=fig, 
                                 width_ratios=widths, 
                                 height_ratios=heights)
        axes = {}
        
        # Metrics values
        for i, inout in enumerate(['Input', 'Output']):
            axes[0,i] = fig.add_subplot(spec[0,i*int(ncols/nm):(i+1)*int(ncols/nm)])
            axes[0,i].text(
                axes[0,i].get_xlim()[1]/2,
                axes[0,i].get_ylim()[1]/2,
                '\n'.join([['%s : %.3f','%s : %.1f%%'][int('ccuracy' in m_name)]%(
                    m_name,
                    [([1,100][int('ccuracy' in m_name)])*class_results_in[m_name],
                     ([1,100][int('ccuracy' in m_name)])*class_results_out[m_name]][
                         int(inout=='Output')]) for m_name in class_results_in.keys()]),
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[0,i].transAxes)
            axes[0,i].set_axis_off()
            axes[0,i].set_title(inout)
        
        # CM accuracies
        # Confusion matrices
        for i, (cm, name) in enumerate(zip(
                [class_IOchange, class_TIchange, class_TOchange],
                ['ClassifiedInput VS ClassifiedOutput', 'TrueWeakLabel VS ClassifiedInput', 'TrueWeakLabel VS ClassifiedOutput'])):
            axes[1,i] = fig.add_subplot(spec[1,i*int(ncols/3):(i+1)*int(ncols/3)])
            th = tss_hss_all(cm, confusion_classes)
            axes[1,i].text(
                axes[1,i].get_xlim()[1]/2,
                axes[1,i].get_ylim()[1]/2,
                name+' :\n'+'\n'.join(['Label %s : TSS %.3f - HSS %.3f'%(
                    lbl, th[0][lbl], th[1][lbl]) for lbl in th[0].keys()]),
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[1,i].transAxes)
            axes[1,i].set_axis_off()
            axes[1,i].set_title(inout)
            
            axes[2,i] = fig.add_subplot(spec[2,i*int(ncols/3):(i+1)*int(ncols/3)])
			
            if adapt_max:
                vmin=0
                vmax=np.max(cm)
            else:
                vmin, vmax = (0,1)
            self.plot_heatmap(cm, confusion_classes, confusion_classes, name, 
                              axes[2,i], with_cbar=False, with_labels=True,
                              xtick_step=xtick_step, ytick_step=ytick_step,
                              vmin=vmin, vmax=vmax)
        
        # spec.tight_layout(fig)
        if not(manual_mode):
            self.savefig_autodpi(
                os.path.join(self.results_dir,save_name),
                bbox_inches=None)
                # bbox_inches='tight')
            plt.close()
        else:
            return os.path.join(self.results_dir,save_name)
    
    def plot_heatmap(self, cm, xticklabels, yticklabels, name, ax, 
                     with_cbar=False, with_labels=True, vmin=0, vmax=1, 
                     xtick_step=5, ytick_step=5, linewidths=.5):
        # name is "name_y VS name_x" eg "True VS Pred"
        group_counts = ["{0:d}".format(int(value)) for value in
                        cm.flatten()]
        group_percentages = ["{0:.0%}".format(value) for value in
                             cm.flatten()/np.sum(cm)]
        if with_labels:
            labels = [f"{v1}\n{v2}" for v1, v2 in
                      zip(group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(*cm.shape)
        else:
            labels = None
        htmp = sns.heatmap(cm, annot=labels, fmt='', ax=ax, 
                    cmap='Blues', cbar=with_cbar,
                    annot_kws={'fontsize':'small'},
                    vmin=vmin, vmax=vmax,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    linewidths=linewidths, linecolor='black')
        ax.set_xticklabels(
            [[None,tl[1]][itl%xtick_step==0] for itl,tl in enumerate(
                natsorted([(tll._x, tll) for tll in ax.get_xticklabels()]))],
            fontsize='small')
        ax.set_yticklabels(
            [[None,tl[1]][itl%ytick_step==0] for itl,tl in enumerate(
                natsorted([(tll._y, tll) for tll in ax.get_yticklabels()]))],
            fontsize='small')
        ax.set_title(name, fontsize='small')
        ax.set_ylabel(name.split(' VS ')[0], fontsize='small')
        ax.set_xlabel(name.split(' VS ')[1], fontsize='small')
        if with_cbar:
            cbar = htmp.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize='small')
    
    def create_minor(self, major):
        # major is the list of 'get_xmajorticklabels()' for the major tick labels
        # returns the positions for the new minor tick labels in-between each major ticks
        pos = [v._x for v in major]
        pos = natsorted(pos)
        pos = [2*pos[0]-pos[1]]+pos+[2*pos[-1]-pos[-2]]
        return [(pos[i+1]+pos[i])/2 for i in range(len(pos)-1)]
    
    def geterr_features_sequence(self, ori_seq, pred_seq):
        """Functions that takes the original and the predicted sequence (n_seq, time)
        and returns the error sequences of physical features (absolute and relative).
        """
        
        ori_feats = feature_transform(ori_seq, ori_seq).transpose() # WARNING: both are 2C
        pred_feats = feature_transform(pred_seq, pred_seq).transpose() # WARNING: both are 2C
        
        assert ori_feats.shape == pred_feats.shape, "Original and predicted features must have the same shape"
        
        abs_err = np.abs(pred_feats - ori_feats)
        rel_err = np.abs(pred_feats - ori_feats) / ori_feats
        #rel_err[np.isnan(rel_err)] = 0
        
        return abs_err, rel_err # absolute and relative error
    
    def ploterr_features(self, err_feats, legends, save_name='err_feat.png',
                         update=False, lc=None, overwrite=False, 
                         portrait=False, fontsize=None):
        if lc == None:
            label = 'model?'
            color = 'b'
            alpha = 1
        else:
            try:
                label, (color, alpha) = lc
            except:
                label, color = lc
                alpha = 1
        fname = os.path.join(self.results_dir,save_name)
        if os.path.isfile(fname) and not(overwrite):
            print("did not overwrite file {}".format(fname))
            return
        
        previous_font_size = plt.rcParams['font.size']
        if fontsize is None:
            plt.rcParams.update({'font.size': 12})
        else:
            plt.rcParams.update({'font.size': fontsize})
        
        if not(portrait):
            cols = 4
            rows = 3
        else:
            cols = 2
            rows = 6
        
        if update:
            axes = {}
            i_fig = 0
            for row in range(rows):
                for col in range(cols):
                    axes[row, col] = plt.gcf().get_axes()[i_fig]
                    i_fig += 1
            ax_leg0 = plt.gcf().get_axes()[-2]
            ax_leg1 = plt.gcf().get_axes()[-1]
            # print('update leg ax', ax_leg.get_position())
        else:
            fig = plt.figure(constrained_layout=True, figsize=(20*cols, 10*rows))
            subfig_w = int(err_feats.shape[1] / 20)
            widths, heights = [[subfig_w]*cols, [.2]+[1]*rows+[.2]]
            spec = gridspec.GridSpec(nrows=rows+2, ncols=cols, figure=fig, 
                                     width_ratios=widths, 
                                     height_ratios=heights)
            axes = {}
            for row in range(rows):
                for col in range(cols):
                    axes[row, col] = fig.add_subplot(spec[row+1, col])
                    print('plt ax', axes[row, col].get_position())
            # ax_leg = fig.add_axes((1/2-1/18,1/24,
            #                        1/2,1/2))
            ax_leg0 = fig.add_subplot(spec[0, :])
            ax_leg0.axis('off')
            ax_leg1 = fig.add_subplot(spec[-1, :])
            ax_leg1.axis('off')
            # print('leg ax', ax_leg.get_position())
        
        i_fig = 0
        
        for i_fig in range(len(err_feats)):
            row = i_fig % rows
            col = i_fig // rows
            x_indices = np.arange(1, err_feats.shape[1]+1, 1)
            
            axes[row, col].plot(x_indices, err_feats[i_fig], color=color, 
                                alpha=alpha, linestyle='-', 
                                label=label)
                                # label='by time'+[' ',''][int(label=='')]+label)
            # axes[row, col].legend(loc='upper center', fontsize='xx-small')
            axes[row, col].set_title(legends[i_fig][0], fontsize='small')
            self.format_axis(axes[row, col], vmin=1, vmax=err_feats.shape[1], 
                             step=np.ceil(int(err_feats.shape[1]/6)/5)*5, 
                             axis='x', ax_label='time', type_labels='int', 
                             minor=False)
            
            vmin=np.nanmin(err_feats[i_fig])
            vmax=np.nanmax(err_feats[i_fig])
            step = (vmax-vmin)/2
            self.format_axis(axes[row, col], vmin=vmin, vmax=vmax, step=step, 
                             axis='y', ax_label=None, type_labels=legends[i_fig][1], 
                             minor=False, margin=[1,1], more2_ticks=False)
        
        patches = []
        for leg in zip(*axes[0,0].get_legend_handles_labels()):
            patches.append(matplotlib.patches.Patch(
                color=leg[0].get_color(), label=leg[1]))
            
        # print(patches)
        ax_leg0.legend(handles=patches, loc='center', fontsize='xx-small', 
                       ncol=6)
        ax_leg1.legend(handles=patches, loc='center', fontsize='xx-small', 
                       ncol=6)
        
        if not(manual_mode):
            self.savefig_autodpi(os.path.join(
                self.results_dir,save_name),
                bbox_inches='tight')
            plt.close()
        else:
            plt.rcParams.update({'font.size': previous_font_size})
            return os.path.join(
                self.results_dir,save_name)
        
        plt.rcParams.update({'font.size': previous_font_size})
        
        return
    
    def sum_extendshape_ifnecess(self, mean_err, err, count):
        """ input of err should be (batch_size, features_nb, seq_length), 
            there could be some np.nan values for err, in this case they will
            be ignored for the mean.
            input of mean_err should be (features_nb, seq_length).
            count : (features_nb, seq_length)
        """
        assert count.shape == mean_err.shape, "The count and mean_err arrays should have compatible dimensions"
        err[np.isinf(err)] = np.nan
        current_count = np.sum(1 - np.isnan(err), axis = 0)
        err = np.nansum(err, axis = 0) # sum on the batch dim
        assert err.shape[0] == mean_err.shape[0], "The feature shape (1) of err should be the same as the number of physical features, here: %i and %i"%(err.shape[0], len(self.feat_legends))
        
        assert err.shape[0] == mean_err.shape[0], "The err and mean_err arrays should have the same number of features"
        if err.shape[1] > mean_err.shape[1]:
            temp_err = 0 * err
            temp_err[:, :mean_err.shape[1]] = mean_err
            temp_count =0 * current_count
            temp_count[:, :count.shape[1]] = count
            return temp_err + err, current_count + temp_count
        elif err.shape[1] < mean_err.shape[1]:
            temp_err = 0 * mean_err
            temp_err[:, :err.shape[1]] = err
            temp_count =0 * count
            temp_count[:, :current_count.shape[1]] = current_count
            return mean_err + temp_err, count + temp_count
        else:
            return mean_err + err, count + current_count
    
    def npRagged(self, list_arr, ref=0):
        # ref if for the index in the list where to base the reference dims
        list_arrays = [deepcopy(arr) for arr in list_arr]
        ref_shape = list_arrays[ref].shape
        if all(arr.shape==ref_shape for arr in list_arrays):
            return np.asarray(list_arrays)
        second = int((ref+1)%2)
        second_shape0 = list_arrays[second].shape[0]
        list_arrays[second] = np.tile(
            list_arrays[second],
            [[1,2][int(
                ref_shape[0]==second_shape0)
                ]]+[1]*(list_arrays[second].ndim-1))
        ragged = np.asarray(list_arrays, dtype='object')
        ragged[second] = ragged[second][:second_shape0]
        return ragged
    
    def save_raw(self, last_raw_saving, ds, makedir,
                 np_metrics_in, np_metrics_out, 
                 # class_results_in, class_results_out,
                 # class_count_in, class_count_out, 
                 # mts_results, pio_centers,
                 mean_abs_err, mean_rel_err,
                 abs_length, rel_length,
                 psnrs, errors, errors5, kcenter,
                 class_IOchange, class_TIchange, class_TOchange,
                 confusion_classes, end = False):
        if self.add_classifier:
            np_metrics_insave = {}
            np_metrics_outsave = {}
            for clsn in list(self.classifier.keys()):
                np_metrics_insave[clsn] = {
                    metric.name:{
                        'total':metric.total,
                        'count':metric.count
                        } for metric in np_metrics_in[clsn]}
                np_metrics_outsave[clsn] = {
                    metric.name:{
                        'total':metric.total,
                        'count':metric.count
                        } for metric in np_metrics_out[clsn]}
                # Collect results
                # class_results_in[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_in[clsn]}
                # class_count_in[clsn][ds] = {metric.name:metric.count for metric in np_metrics_in[clsn]}
                # print('Classifier metrics IN:\n', class_results_in[clsn][ds])
                # class_results_out[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_out[clsn]}
                # class_count_out[clsn][ds] = {metric.name:metric.count for metric in np_metrics_out[clsn]}
                # print('Classifier metrics OUT:\n', class_results_out[clsn][ds])
        else:
            np_metrics_insave = {'noclassifier': None}
            np_metrics_outsave = {'noclassifier': None}
            # class_results_in = {'noclassifier': None}
            # class_count_in = {'noclassifier': None}
            # class_results_out = {'noclassifier': None}
            # class_count_out = {'noclassifier': None}
               
        mts_metrics_save = {}
        for i_img in list(makedir.keys()):
            # Collect results
            mts_metrics_save[i_img] = {
                 nm: vm.return_save()
                 for nm,vm in self.mts_metrics[i_img].metrics_dict.items()}
            # mts_results[i_img][ds] = {
            #     'by_no':self.mts_metrics[i_img].result_by_no(),
            #     'by_1':self.mts_metrics[i_img].result_by_1(),
            #     'by_1_2':self.mts_metrics[i_img].result_by_1_2(),
            #     'by_1_3':self.mts_metrics[i_img].result_by_1_3(),
            #     'by_1_2_3':self.mts_metrics[i_img].result_by_1_2_3()}
        
        if self.add_centercount:
            pio_metrics_save = {}
            for i_img in list(makedir.keys()):
                # Collect results
                pio_metrics_save[i_img] = {
                    'prior': {
                        'total':self.center_counter_pio[i_img].freqdict_prior.total,
                        'count':self.center_counter_pio[i_img].freqdict_prior.count
                        },
                    'io': {
                        'total':self.center_counter_pio[i_img].freqdict_io.total,
                        'count':self.center_counter_pio[i_img].freqdict_io.count
                        },
                    'pio': {
                        'total':self.center_counter_pio[i_img].batchinfo_pio.total,
                        'count':self.center_counter_pio[i_img].batchinfo_pio.count
                        }
                    }
                # pio_centers[i_img][ds] = {
                #     'cond_no':self.center_counter_pio[i_img].result_cond_no(),
                #     'cond_1':self.center_counter_pio[i_img].result_cond_1(),
                #     'cond_1_2':self.center_counter_pio[i_img].result_cond_1_2(),
                #     'cond_1_3':self.center_counter_pio[i_img].result_cond_1_3(),
                #     'cond_1_2_3':self.center_counter_pio[i_img].result_cond_1_2_3()}
        # else:
        #     pio_metrics_save[i_img] = {'noclassifier': None}
        #     pio_centers['noclassifier'][ds] = [None]
        
        # Outputs or test before plots:
        # mean_abs_err
        # mean_rel_err
        # psnrs
        # errors
        # errors5
        # class_IOchange
        # class_TIchange
        # class_TOchange
        # class_results_in
        # class_results_out
        # confusion_classes
        # mts_results
        # pio_centers
        
        # npRagged Trick to overcome a bug with np.asarray dtype='object' 
        # to store a ragged array when the first dim have same length 
        # but the next ones may differ
        kwargs = {
            'end' : end,
            'last_raw_saving' : last_raw_saving,
            'mean_abs_err' : mean_abs_err[ds],
            'mean_rel_err' : mean_rel_err[ds],
            'abs_length' : abs_length[ds],
            'rel_length' : rel_length[ds],
            'psnrs': psnrs,
            'errors': self.npRagged(errors, ref=0),
            'errors5': self.npRagged(errors5, ref=0),
            'kcenter': kcenter,
            # 'mts_results' : {clsn:mts_results[clsn][ds] for clsn in mts_results.keys()}}
            'mts_metrics_save' : mts_metrics_save}
        if self.add_centercount:
            kwargs = {
                **kwargs,
                # **{'pio_centers' : {clsn:pio_centers[clsn][ds] for clsn in pio_centers.keys()}}}
                **{'pio_metrics_save' : pio_metrics_save}}
        if self.add_classifier:
            kwargs = {
                **kwargs,
                **{'np_metrics_insave' : np_metrics_insave,
                   'np_metrics_outsave' : np_metrics_outsave,
                   # 'class_count_in' : {clsn:class_count_in[clsn][ds] for clsn in class_count_in.keys()},
                   # 'class_results_in' : {clsn:class_results_in[clsn][ds] for clsn in class_results_in.keys()},
                   # 'class_count_out' : {clsn:class_count_out[clsn][ds] for clsn in class_count_out.keys()},
                   # 'class_results_out' : {clsn:class_results_out[clsn][ds] for clsn in class_results_out.keys()},
                   'class_IOchange' : {clsn:class_IOchange[clsn][ds] for clsn in class_IOchange.keys()},
                   'class_TIchange' : {clsn:class_TIchange[clsn][ds] for clsn in class_TIchange.keys()},
                   'class_TOchange' : {clsn:class_TOchange[clsn][ds] for clsn in class_TOchange.keys()},
                   'confusion_classes' : confusion_classes}}
        
        if os.path.isfile(os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds))):
            os.rename(os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds)), 
                      os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds)))
        if os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))):
            os.rename(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), 
                      os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds)))
        np.savez(os.path.join(self.results_dir, 'test_RAW_{}'.format(ds)),
                 **kwargs)
        if os.path.isfile(os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds))):
            os.remove(os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds)))
    
    def tests(self):
        if not os.path.exists(os.path.join(self.results_dir, 'testing')) and not(manual_mode):
            os.makedirs(os.path.join(self.results_dir, 'testing'))
        if not(manual_mode):
            if self.model_type in ['LSTM', 'LSTMS', 'GRU', 'GRUS', 'NBeats']:
                print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(True, True))
                self.model_instance(True, True)
            else:
                print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(False, True))
                self.model_instance(False, True)
            print('[%s - START] Testing..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
        if os.path.isfile(os.path.join(self.results_dir, 'test_results.npz')) and not(manual_mode):
            results = np.load(os.path.join(self.results_dir, 'test_results.npz'), allow_pickle = True)
            means = results['means'].all()
            stds = results['stds'].all()
            means_ssim = results['means_ssim'].all()
            stds_ssim = results['stds_ssim'].all()
            means_kcenter = results['means_kcenter'].all()
            stds_kcenter = results['stds_kcenter'].all()
            mean_abs_err = results['mean_abs_err'].all()
            mean_rel_err = results['mean_rel_err'].all()
            abs_length = results['abs_length'].all()
            rel_length = results['rel_length'].all()
            mts_results = results['mts_results'].all()
            if self.add_classifier:
                class_count_in = {}
                class_results_in = {}
                class_count_out = {}
                class_results_out = {}
                class_IOchange = {}
                class_TIchange = {}
                class_TOchange = {}
                confusion_classes = {}
                for clsn in list(self.classifier.keys()):
                    if os.path.isfile(os.path.join(
                            self.results_dir, 
                            'test_classresults_%s.npz'%clsn)):
                        classresults =  np.load(os.path.join(
                            self.results_dir, 
                            'test_classresults_%s.npz'%clsn), 
                            allow_pickle = True)
                        class_count_in[clsn] = classresults['class_count_in'].all()
                        class_results_in[clsn] = classresults['class_results_in'].all()
                        class_count_out[clsn] = classresults['class_count_out'].all()
                        class_results_out[clsn] = classresults['class_results_out'].all()
                        class_IOchange[clsn] = classresults['class_IOchange'].all()
                        class_TIchange[clsn] = classresults['class_TIchange'].all()
                        class_TOchange[clsn] = classresults['class_TOchange'].all()
                        confusion_classes[clsn] = classresults['confusion_classes']
                        classresults.close()
                    else:
                        class_count_in[clsn] = {}
                        class_results_in[clsn] = {}
                        class_count_out[clsn] = {}
                        class_results_out[clsn] = {}
                        class_IOchange[clsn] = {}
                        class_TIchange[clsn] = {}
                        class_TOchange[clsn] = {}
                        confusion_classes[clsn] = [None]
            else:
                class_count_in = {'noclassifier' : {}}
                class_results_in = {'noclassifier' : {}}
                class_count_out = {'noclassifier' : {}}
                class_results_out = {'noclassifier' : {}}
                class_IOchange = {'noclassifier' : {}}
                class_TIchange = {'noclassifier' : {}}
                class_TOchange = {'noclassifier' : {}}
                confusion_classes = {'noclassifier': [None]}
            if self.add_centercount:
                pio_centers = results['pio_centers'].all()
            results.close()
        else:
            means, stds = {}, {}
            means_ssim, stds_ssim = {}, {}
            means_kcenter, stds_kcenter = {}, {}
            mean_abs_err, mean_rel_err = {},{}
            abs_length, rel_length = {}, {}
            mts_results = {clsn:{} for clsn in list(self.classifier.keys())}
            if self.add_classifier:
                class_count_in = {}
                class_results_in = {}
                class_count_out = {}
                class_results_out = {}
                class_IOchange = {}
                class_TIchange = {}
                class_TOchange = {}
                confusion_classes = {}
                for clsn in list(self.classifier.keys()):
                    class_count_in[clsn] = {}
                    class_results_in[clsn] = {}
                    class_count_out[clsn] = {}
                    class_results_out[clsn] = {}
                    class_IOchange[clsn] = {}
                    class_TIchange[clsn] = {}
                    class_TOchange[clsn] = {}
                    confusion_classes[clsn] = [None]
            else:
                class_count_in = {'noclassifier' : {}}
                class_results_in = {'noclassifier' : {}}
                class_count_out = {'noclassifier' : {}}
                class_results_out = {'noclassifier' : {}}
                class_IOchange = {'noclassifier' : {}}
                class_TIchange = {'noclassifier' : {}}
                class_TOchange = {'noclassifier' : {}}
                confusion_classes = {'noclassifier': [None]}
            if self.add_centercount:
                if self.add_classifier:
                    pio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
                else:
                    pio_centers = {'noclassifier':{}}
        
        test_ds = self.test_ds
        for ds in test_ds:
            if 'TEL' in ds:
                test_ds.remove(ds)
                test_ds += [ds]
        for ds in test_ds:
            if 'TEL' not in ds:
                gen = self.test_generators[ds][0]
                length = self.test_generators[ds][1]
                print('[Testing] %s Samples:'%ds)
            else:
                kwargs = {
                    'means' : means,
                    'stds' : stds,
                    'means_ssim' : means_ssim,
                    'stds_ssim' : stds_ssim,
                    'means_kcenter' : means_kcenter,
                    'stds_kcenter' : stds_kcenter,
                    'mean_abs_err' : mean_abs_err,
                    'mean_rel_err' : mean_rel_err,
                    'abs_length' : abs_length,
                    'rel_length' : rel_length,
                    'mts_results' : mts_results}
                if self.add_centercount:
                    kwargs = {
                        **kwargs,
                        **{'pio_centers' : pio_centers}}
                    
                np.savez(os.path.join(self.results_dir, 'test_results'),
                         **kwargs)
                
                if self.add_classifier:
                    for clsn in list(self.classifier.keys()):
                        np.savez(os.path.join(self.results_dir, 
                                              'test_classresults_%s.npz'%clsn),
                                 **{'class_count_in' : class_count_in[clsn],
                                    'class_results_in' : class_results_in[clsn],
                                    'class_count_out' : class_count_out[clsn],
                                    'class_results_out' : class_results_out[clsn],
                                    'class_IOchange' : class_IOchange[clsn],
                                    'class_TIchange' : class_TIchange[clsn],
                                    'class_TOchange' : class_TOchange[clsn],
                                    'confusion_classes' : confusion_classes[clsn]})
                
                self.long_test()
                
                results = np.load(os.path.join(self.results_dir, 'test_results.npz'), allow_pickle = True)
                means = results['means'].all()
                stds = results['stds'].all()
                means_ssim = results['means_ssim'].all()
                stds_ssim = results['stds_ssim'].all()
                means_kcenter = results['means_kcenter'].all()
                stds_kcenter = results['stds_kcenter'].all()
                mean_abs_err = results['mean_abs_err'].all()
                mean_rel_err = results['mean_rel_err'].all()
                abs_length = results['abs_length'].all()
                rel_length = results['rel_length'].all()
                mts_results = results['mts_results'].all()
                if self.add_classifier:
                    class_count_in = {}
                    class_results_in = {}
                    class_count_out = {}
                    class_results_out = {}
                    class_IOchange = {}
                    class_TIchange = {}
                    class_TOchange = {}
                    confusion_classes = {}
                    for clsn, clsfier in self.classifier.items():
                        if os.path.isfile(os.path.join(
                                self.results_dir, 
                                'test_classresults_%s.npz'%clsn)):
                            classresults =  np.load(os.path.join(
                                self.results_dir, 
                                'test_classresults_%s.npz'%clsn), 
                                allow_pickle = True)
                            class_count_in[clsn] = classresults['class_count_in'].all()
                            class_results_in[clsn] = classresults['class_results_in'].all()
                            class_count_out[clsn] = classresults['class_count_out'].all()
                            class_results_out[clsn] = classresults['class_results_out'].all()
                            class_IOchange[clsn] = classresults['class_IOchange'].all()
                            class_TIchange[clsn] = classresults['class_TIchange'].all()
                            class_TOchange[clsn] = classresults['class_TOchange'].all()
                            confusion_classes[clsn] = classresults['confusion_classes']
                            classresults.close()
                        else:
                            class_count_in[clsn] = {}
                            class_results_in[clsn] = {}
                            class_count_out[clsn] = {}
                            class_results_out[clsn] = {}
                            class_IOchange[clsn] = {}
                            class_TIchange[clsn] = {}
                            class_TOchange[clsn] = {}
                if self.add_centercount:
                    pio_centers = results['pio_centers'].all()
                results.close()
                break
            
            print('# of tests :', length)
            
            psnrs = []
            errors = []
            errors5 = []
            kcenter = {}
            for i in range(1,7):
                kcenter[i] = []
            n = 0
            
            mean_abs_err[ds] = np.zeros((len(self.feat_legends), 1))
            mean_rel_err[ds] = np.zeros((len(self.feat_legends), 1))
            abs_length[ds] = np.zeros((len(self.feat_legends), 1))
            rel_length[ds] = np.zeros((len(self.feat_legends), 1))
            
            if self.add_classifier:
                np_all_metrics = [
                    NP_CategoricalCrossentropy, 
                    NP_BinaryCrossentropy, 
                    NP_CategoricalAccuracy, 
                    NP_BinaryAccuracy]
                np_metrics_in = {}
                np_metrics_out = {}
                makedir = {}
                for clsn, clsfier in self.classifier.items():
                    np_metrics_in[clsn] = [m for m in np_all_metrics if any(
                        [mm.name in m().name for mm in clsfier.model.model.compiled_metrics._metrics])]
                    np_metrics_in[clsn] = [mm(**a) for mm, a in zip(
                        np_metrics_in[clsn], 
                        [[{}, {'class_assign_fn':clsfier.model.np_assign_class}][int(
                            tf.keras.metrics.CategoricalAccuracy().name == m().name)] for m in np_metrics_in[clsn]])]
                    np_metrics_out[clsn] = [m for m in np_all_metrics if any(
                        [mm.name in m().name for mm in clsfier.model.model.compiled_metrics._metrics])]
                    np_metrics_out[clsn] = [mm(**a) for mm, a in zip(
                        np_metrics_out[clsn], 
                        [[{}, {'class_assign_fn':clsfier.model.np_assign_class}][int(
                            tf.keras.metrics.CategoricalAccuracy().name == m().name)] for m in np_metrics_out[clsn]])]
                    for metric in np_metrics_in[clsn]:
                        metric.reset_states()
                    for metric in np_metrics_out[clsn]:
                        metric.reset_states()
                    # confusion matrix IN OUT classification
                    if clsfier.noclass is not None:
                        class_IOchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                        class_TIchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                        class_TOchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                        confusion_classes[clsn] = clsfier.classes + [clsfier.noclass]
                    else:
                        assert clsfier.noclass is None, "In and Out classifiers should have the same 'noclass'"
                        class_IOchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                        class_TIchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                        class_TOchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                        confusion_classes[clsn] = clsfier.classes
                    makedir[clsn] = clsn
                    # print('clsn', clsn)
                    # print('confusion_classes', confusion_classes[clsn])
            else:
                makedir = {'noclassifier':'noclassifier'}
                np_metrics_in = {'noclassifier':None}
                np_metrics_out = {'noclassifier':None}
            
            for _, mkd in makedir.items():
                if not os.path.exists(os.path.join(self.results_dir, 'testing', mkd)):
                    os.makedirs(os.path.join(self.results_dir, 'testing', mkd))
            
            for mtsm in self.mts_metrics.values():
                mtsm.reset()
            
            for clsn in list(self.classifier.keys()):
                self.mts_metrics[clsn].reset()
            
            if self.add_centercount:
                for clsn in list(self.classifier.keys()):
                    self.center_counter_pio[clsn].reset()
            
            # Loop through ds
            # for fast debug and memory purposes
            end_raw = False
            if os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))):
                
                results = np.load(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), allow_pickle = True)
                if 'end' in results.keys():
                    end_raw = results['end']
                else:
                    end_raw = True
                last_raw_saving = results['last_raw_saving']
                print("loading previous raw results: last saved index is {}".format(last_raw_saving))
                mean_abs_err[ds] = results['mean_abs_err']
                mean_rel_err[ds] = results['mean_rel_err']
                abs_length[ds] = results['abs_length']
                rel_length[ds] = results['rel_length']
                psnrs = list(results['psnrs'])
                errors = list(results['errors'])
                errors5 = list(results['errors5'])
                kcenter = results['kcenter'].all()
                if self.add_classifier:
                    np_metrics_insave = results['np_metrics_insave'].all()
                    np_metrics_outsave = results['np_metrics_outsave'].all()
                    for clsn in list(self.classifier.keys()):
                        for metric in np_metrics_in[clsn]:
                            metric.total = np_metrics_insave[clsn][
                                metric.name]['total']
                            metric.count = np_metrics_insave[clsn][
                                metric.name]['count']
                        for metric in np_metrics_out[clsn]:
                            metric.total = np_metrics_outsave[clsn][
                                metric.name]['total']
                            metric.count = np_metrics_outsave[clsn][
                                metric.name]['count']
                        class_IOchange[clsn][ds] = results['class_IOchange'].all()[clsn]
                        class_TIchange[clsn][ds] = results['class_TIchange'].all()[clsn]
                        class_TOchange[clsn][ds] = results['class_TOchange'].all()[clsn]
                        confusion_classes[clsn] = results['confusion_classes'].all()[clsn]
                       
                mts_metrics_save = results['mts_metrics_save'].all()
                for clsn in list(self.classifier.keys()):
                    for nm,vm in self.mts_metrics[clsn].metrics_dict.items():
                        vm.from_saved(mts_metrics_save[clsn][nm])
                
                if self.add_centercount:
                    pio_metrics_save = results['pio_metrics_save'].all()
                    for clsn in list(self.classifier.keys()):
                        self.center_counter_pio[
                            clsn].freqdict_prior.total = pio_metrics_save[
                                clsn]['prior']['total']
                        self.center_counter_pio[
                            clsn].freqdict_prior.count = pio_metrics_save[
                                clsn]['prior']['count']
                        self.center_counter_pio[
                            clsn].freqdict_io.total = pio_metrics_save[
                                clsn]['io']['total']
                        self.center_counter_pio[
                            clsn].freqdict_io.count = pio_metrics_save[
                                clsn]['io']['count']
                        self.center_counter_pio[
                            clsn].batchinfo_pio.total = pio_metrics_save[
                                clsn]['pio']['total']
                        self.center_counter_pio[
                            clsn].batchinfo_pio.count = pio_metrics_save[
                                clsn]['pio']['count']
                results.close()
            else:
                last_raw_saving = 0
            
            if not(end_raw) and  last_raw_saving < length:
                start_timer = time.time()
                for (masked, mask, pos), ori in tqdm(gen):
                    
                    # masked = self.to_rgb(masked)
                    # mask = self.to_rgb(mask)
                    # ori = self.to_rgb(ori)
                    
                    n += ori.shape[0]
                    if n<=last_raw_saving:
                        print("skipping step # {}, will skip all steps until step # {}".format(n, last_raw_saving))
                        continue
                    
                    # Run prediction on image & mask
                    pred = self.model.predict([ori, mask])
                    
                    lab = [self.parse_lab_pos(pos, i, self.test_generators[ds][2])[0] for i in range(len(pos))]
                    if self.add_classifier:
                        class_in = {}
                        class_out = {}
                        for clsn, clsfier in self.classifier.items():
                            # get classifier tested
                            class_mask = create_class_mask(ori, 
                                                           self.mask_ratio, 
                                                           self.random_ratio)
                            catlab = clsfier.catlabels_from_idxpos(
                                pos, dict_poslab=self.test_generators[ds][2], 
                                predictonly=False)
                            class_true = clsfier.model.np_assign_class(catlab)
                            pci = clsfier.model.predict([ori, class_mask])
                            for metric in np_metrics_in[clsn]:
                                metric.update_state(catlab,pci)
                            class_in[clsn] = clsfier.model.np_assign_class(pci)
                            pco = clsfier.model.predict([pred, class_mask])
                            for metric in np_metrics_out[clsn]:
                                metric.update_state(catlab,pco)
                            class_out[clsn] = clsfier.model.np_assign_class(pco)
                            class_IOchange[clsn][ds] += confusion_matrix(
                                class_in[clsn], class_out[clsn],
                                labels=confusion_classes[clsn])
                            class_TIchange[clsn][ds] += confusion_matrix(
                                class_true, class_in[clsn],
                                labels=confusion_classes[clsn])
                            class_TOchange[clsn][ds] += confusion_matrix(
                                class_true, class_out[clsn],
                                labels=confusion_classes[clsn])
                    else:
                        class_in = {'noclassifier':['NoClass']*len(lab)}
                        class_out = {'noclassifier':['NoClass']*len(lab)}
                        class_IOchange[clsn][ds] = None
                        class_TIchange[clsn][ds] = None
                        class_TOchange[clsn][ds] = None
                    
                    for clsn in self.mts_metrics.keys():
                        self.mts_metrics[clsn].update(
                            (lab, class_in[clsn], class_out[clsn]),
                            (ori[:,-int(ori.shape[1]*self.mask_ratio):],
                             pred[:,-int(ori.shape[1]*self.mask_ratio):]))
                    
                    for clsn in list(self.classifier.keys()):
                        self.mts_metrics[clsn].update(
                            (lab, class_in[clsn], class_out[clsn]),
                            (ori[:,-int(ori.shape[1]*self.mask_ratio):],
                            pred[:,-int(ori.shape[1]*self.mask_ratio):]))
                    
                    if self.add_centercount:
                        for clsn in list(self.classifier.keys()):
                            self.center_counter_pio[clsn].fit_batch(
                                lab, (class_in[clsn], class_out[clsn]),
                                (ori[:,:-int(ori.shape[1]*self.mask_ratio)],
                                 ori[:,-int(ori.shape[1]*self.mask_ratio):],
                                 pred[:,-int(ori.shape[1]*self.mask_ratio):]))
                            # results_no = self.center_counter_pio[clsn].result_cond_no()['sample_info']['p_c1c2']
                            # print('pio_results_no', results_no)
                            # results_1 = self.center_counter_pio[clsn].result_cond_1()['sample_info']['p_c1c2']
                            # print('pio_results_no', results_1)
                            # results_1_2 = self.center_counter_pio[clsn].result_cond_1_2()['sample_info']['p_c1c2']
                            # print('pio_results_no', results_1_2)
                            # results_1_3 = self.center_counter_pio[clsn].result_cond_1_3()['sample_info']['p_c1c2']
                            # print('pio_results_no', results_1_3)
                            # results_1_2_3 = self.center_counter_pio[clsn].result_cond_1_2_3()['sample_info']['p_c1c2']
                            # print('pio_results_no', results_1_2_3)
                            # assert False, "STOP HERE"
                    print('ori', ori.shape)
                    print('pred', pred.shape)
                    print('mr', self.mask_ratio)
                    err, err5 = onebatchpredict_errors(ori, pred, self.mask_ratio)
                    
                    errors.append(err)
                    errors5.append(err5)
                    for i in range(1,7):
                        kcenter[i] = list(kcenter[i])
                        kcenter[i].append(np.asarray([kcentroids_equal(to_kcentroid_seq(p.squeeze()[-int(o.squeeze().shape[0] * self.mask_ratio):,:], k=i)[1], to_kcentroid_seq(o.squeeze()[-int(o.squeeze().shape[0] * self.mask_ratio):,:], k=i)[1]) for o, p in zip(ori, pred)]))
                    
                    # Calculate PSNR
                    psnrs.append(-10.0 * np.log10(np.mean(np.square(pred[:,-int(ori.shape[1] * self.mask_ratio):,:] - ori[:,-int(ori.shape[1] * self.mask_ratio):,:]))))
                    
                    if self.with_features:
                        abs_err, rel_err = zip(*[self.geterr_features_sequence(o.squeeze()[-int(ori.shape[1] * self.mask_ratio):, :], p.squeeze()[-int(ori.shape[1] * self.mask_ratio):, :]) for o,p in zip(ori, pred)])
                        abs_err = np.stack(abs_err)
                        rel_err = np.stack(rel_err)
                        
                        mean_abs_err[ds], abs_length[ds] = self.sum_extendshape_ifnecess(mean_abs_err[ds], abs_err, abs_length[ds])
                        mean_rel_err[ds], rel_length[ds] = self.sum_extendshape_ifnecess(mean_rel_err[ds], rel_err, rel_length[ds])
                    else:
                        mean_abs_err[ds], abs_length[ds] = [None, None]
                        mean_rel_err[ds], rel_length[ds] = [None, None]
                    
                    if time.time()-start_timer > 3600:
                        start_timer = time.time()
                        self.save_raw(n, ds, makedir,
                                     np_metrics_in, np_metrics_out, 
                                     # class_results_in, class_results_out,
                                     # class_count_in, class_count_out, 
                                     # mts_results, pio_centers,
                                     mean_abs_err, mean_rel_err,
                                     abs_length, rel_length,
                                     psnrs, errors, errors5, kcenter,
                                     class_IOchange, class_TIchange, class_TOchange,
                                     confusion_classes)
                    
                    if n > length or self.debug:
                    # if n > 0:
                        break
                
                if not(end_raw):
                    if self.with_features:
                        mean_abs_err[ds] /= abs_length[ds]
                        mean_rel_err[ds] /= rel_length[ds]
                    
                    self.save_raw(n, ds, makedir,
                                 np_metrics_in, np_metrics_out, 
                                 # class_results_in, class_results_out,
                                 # class_count_in, class_count_out, 
                                 # mts_results, pio_centers,
                                 mean_abs_err, mean_rel_err,
                                 abs_length, rel_length,
                                 psnrs, errors, errors5, kcenter,
                                 class_IOchange, class_TIchange, class_TOchange,
                                 confusion_classes, end=True)
            
            if self.add_classifier:
                for clsn in list(self.classifier.keys()):
                    # Collect results
                    class_results_in[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_in[clsn]}
                    class_count_in[clsn][ds] = {metric.name:metric.count for metric in np_metrics_in[clsn]}
                    print('Classifier metrics IN:\n', class_results_in[clsn][ds])
                    class_results_out[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_out[clsn]}
                    class_count_out[clsn][ds] = {metric.name:metric.count for metric in np_metrics_out[clsn]}
                    print('Classifier metrics OUT:\n', class_results_out[clsn][ds])
            else:
                class_results_in = {'noclassifier': None}
                class_count_in = {'noclassifier': None}
                class_results_out = {'noclassifier': None}
                class_count_out = {'noclassifier': None}
                   
            for i_img in list(makedir.keys()):
                # Collect results
                mts_results[i_img][ds] = {
                    'by_no':self.mts_metrics[i_img].result_by_no(),
                    'by_1':self.mts_metrics[i_img].result_by_1(),
                    'by_1_2':self.mts_metrics[i_img].result_by_1_2(),
                    'by_1_3':self.mts_metrics[i_img].result_by_1_3(),
                    'by_1_2_3':self.mts_metrics[i_img].result_by_1_2_3()}
            
            if self.add_centercount:
                for i_img in list(makedir.keys()):
                    # Collect results
                    pio_centers[i_img][ds] = {
                        'cond_no':self.center_counter_pio[i_img].result_cond_no(),
                        'cond_1':self.center_counter_pio[i_img].result_cond_1(),
                        'cond_1_2':self.center_counter_pio[i_img].result_cond_1_2(),
                        'cond_1_3':self.center_counter_pio[i_img].result_cond_1_3(),
                        'cond_1_2_3':self.center_counter_pio[i_img].result_cond_1_2_3()}
            # else:
            #     pio_centers['noclassifier'][ds] = [None]
            
            # DISPLAY RESULTS
            if self.with_features:
                for i in range(len(self.feat_legends)):
                    print('Mean-err ABS feature-%s = %.2f'%(self.feat_legends[i][0], np.mean(mean_abs_err[ds])))
                    print('Mean-err REL feature-%s = %.2f'%(self.feat_legends[i][0], np.mean(mean_rel_err[ds])))
            
            means[ds] = [np.array(psnrs).mean()]
            stds[ds] = np.array(psnrs).std()
            print('Mean PSNR = %.2f'%means[ds][0])
            print('Std PSNR = %.2f'%stds[ds])
            
            psnr1m = np.mean(-10.0 * np.log10(np.concatenate(errors, axis = 1)), axis = 1)[0,:]
            psnr5m = np.mean(-10.0 * np.log10(np.concatenate(errors5, axis = 1)), axis = 1)[0,:]
            errors = np.mean(np.concatenate(errors, axis = 1), axis = 1) # mse, ssim
            errors5 = np.mean(np.concatenate(errors5, axis = 1), axis = 1)
            kcenterm = {}
            for i in range(1,7):
                kcenterm[i] = np.mean(np.concatenate(kcenter[i], axis = 0), axis = 0)
            
            means_ssim[ds] = [np.array(errors[1,:]).mean()]
            stds_ssim[ds] = np.array(errors[1,:]).std()
            print('Mean SSIM = %.2f'%means_ssim[ds][0])
            print('Std SSIM = %.2f'%stds_ssim[ds])
            means_kcenter[ds] = {}
            stds_kcenter[ds] = {}
            for i in range(1,7):
                means_kcenter[ds][i] = [np.array(kcenterm[i]).mean()]
                stds_kcenter[ds][i] = np.array(kcenterm[i]).std()
                print('Mean K-CENTERS-%i = %.2f'%(i, means_kcenter[ds][i][0]))
                print('Std K-CENTERS-%i = %.2f'%(i, stds_kcenter[ds][i]))
            
            for i in range(1,7):
                assert len(kcenterm[i]) == len(psnr1m), "accuracy lengthes should be the same for kcenter%i and psnr1m, here %i and %i"%(i, len(kcenterm[i]),len(psnr1m))
            
            means[ds].append(psnr1m)
            means[ds].append(psnr5m)
            means_ssim[ds].append(errors[1, :])
            means_ssim[ds].append(errors5[1,:])
            for i in range(1,7):
                means_kcenter[ds][i].append(kcenterm[i])
            
            if not('_' in ds):
                if self.add_classifier:
                    for clsn in list(self.classifier.keys()):
                        self.plot_classmetrics(
                            class_results_in[clsn][ds], class_results_out[clsn][ds],
                            class_IOchange[clsn][ds], class_TIchange[clsn][ds], class_TOchange[clsn][ds],
                            confusion_classes[clsn],
                            save_name=os.path.join('testing', makedir[clsn], 
                                'Data-{}_confusionM.png'.format(ds)))
                
                for i_img in list(makedir.keys()):
                    # Collect results
                    self.plot_mtsres(
                        mts_results[i_img][ds], 
                        save_name=os.path.join(
                            'testing', makedir[i_img],
                            'Data-{}_mts_results.png'.format(ds)))
                
                if self.add_centercount:
                    for i_img in list(makedir.keys()):
                        for lbl in pio_centers[i_img][ds]['cond_1']['centers']['c0'].keys():
                            self.plot_clsres_one_label(
                                pio_centers[i_img][ds], lbl, 
                                save_name=os.path.join(
                                    'testing', makedir[i_img],
                                    'Data-{}_detailedcentercount.png'.format(ds)))
            
            if self.with_features:
                self.ploterr_features(mean_abs_err[ds], self.feat_legends, 
                                      os.path.join(
                                          'testing','Err{}_feats_{}.png'.format('ABS', ds)))
                self.ploterr_features(mean_rel_err[ds], self.feat_legends,
                                      os.path.join(
                                          'testing','Err{}_feats_{}.png'.format('REL', ds)))
            
            fname = os.path.join(
                self.results_dir,'testing','Data-{}.png'.format(ds))
            if os.path.isfile(fname):
                print("did not overwrite file {}".format(fname))
            else:
                fig = plt.figure(constrained_layout=True, figsize=(80, 40))
                widths, heights = [[2, 2, 2, 2], [1, 1, 1, 1]]
                spec = gridspec.GridSpec(nrows=4, ncols=4, figure=fig, 
                                         width_ratios=widths, 
                                         height_ratios=heights)
                axes = {}
                for row in range(4):
                    for col in range(4):
                        axes[row, col] = fig.add_subplot(spec[row, col])
                
                # Plotting Raw CV errors (PSNR, SSIM)
                axes[0, 0].plot(range(1, len(psnr1m)+1), 
                                psnr1m, label='PSNR')
                self.format_axis(axes[0, 0], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[0, 0], vmin=0, vmax=len(psnr1m), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[0, 0], legend_loc='upper center', title='time predictions', fontsize='x-small')
                
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=psnr5m, tofit=psnr1m)
                axes[1, 0].plot(np.arange(vstart, vend, vstep), 
                                psnr5m, label='PSNR')
                self.format_axis(axes[1, 0], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[1, 0], vmin=vmin, vmax=len(psnr5m), lmin=0, lmax=len(psnr1m), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[1, 0], legend_loc='upper center', title='avg each 5% time predictions', fontsize='x-small')
                
                axes[2, 0].plot(range(1, len(errors[1])+1), 
                                errors[1], label='SSIM')
                axes[2, 0].plot(range(1, len(errors[1])+1), 
                                np.ones_like(errors[1]), label='best', linestyle=':', color='g')
                self.format_axis(axes[2, 0], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[2, 0], vmin=0, vmax=len(errors[1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[2, 0], legend_loc='upper center', title='time predictions', fontsize='x-small')
                
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=errors5[1], tofit=errors[1])
                axes[3, 0].plot(np.arange(vstart, vend, vstep), 
                                errors5[1], label='SSIM')
                axes[3, 0].plot(np.arange(vstart, vend, vstep), 
                                np.ones_like(errors5[1]), label='best', linestyle=':', color='g')
                self.format_axis(axes[3, 0], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[3, 0], vmin=vmin, vmax=len(errors5[1]), lmin=0, lmax=len(errors[1]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[3, 0], legend_loc='upper center', title='avg each 5% time predictions', fontsize='x-small')
                
                # Plotting the other columns: Physical errors (centers assignment)
                for i in range(1,7):
                    row = ((i - 1) % 2) * 2
                    col = (i - 1) // 2 + 1
                    axes[row, col].plot(range(1, len(kcenterm[i])+1), 
                                        kcenterm[i], label='%i-NearestCenters'%i)
                    axes[row, col].plot(range(1, len(kcenterm[i])+1), 
                                        [kinter(i) for _ in range(len(kcenterm[i]))], label='%i-RandomBaseground'%i, linestyle=':', color='r')
                    axes[row, col].plot(range(1, len(kcenterm[i])+1), 
                                        np.ones_like(kcenterm[i]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', ax_label='accuracy', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(kcenterm[i]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', title='time predictions', fontsize='x-small')
                    axes[row + 1, col].plot(*forplot_assignement_accuracy(kcenterm[i], bin_size=int(self.label_length * 0.05)), label='%i-NearestCenters'%i)
                    axes[row + 1, col].plot(*forplot_assignement_accuracy([kinter(i) for _ in range(len(kcenterm[i]))], bin_size=int(self.label_length * 0.05)), label='%i-RandomBaseground'%i, linestyle=':', color='r')
                    axes[row + 1, col].plot(*forplot_assignement_accuracy(np.ones_like(kcenterm[i])), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row+1, col], vmin=0, vmax=1, step = 0.2, axis = 'y', ax_label='accuracy', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row+1, col], vmin=0, vmax=len(kcenterm[i]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row+1, col], legend_loc='upper center', title='avg each 5% time predictions', fontsize='x-small')
                
                # spec.tight_layout(fig)
                
                if not(manual_mode):
                    self.savefig_autodpi(os.path.join(
                        self.results_dir,'testing','Data-{}.png'.format(ds)),
                        bbox_inches='tight')
                    plt.close()
            
            fname = os.path.join(
                self.results_dir,'testing','Data-{}_PSNR_SSIM.pdf'.format(ds))
            if os.path.isfile(fname):
                print("did not overwrite file {}".format(fname))
            else:
                fig, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('time', fontsize='small')
                ax1.set_ylabel('PSNR', color=color)
                ax1.plot([v*self.mask_ratio*self.label_length for v in range(1, len(means[ds][1])+1)], 
                         means[ds][1], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(which='major', linestyle=':')
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('SSIM', color=color)
                ax2.plot([v*self.mask_ratio*self.label_length for v in range(1, len(means_ssim[ds][1])+1)], 
                         means_ssim[ds][1], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.grid(which='major', linestyle=':')
                plt.tight_layout()
                if not(manual_mode):
                    self.savefig_autodpi(fname,
                        bbox_inches='tight')
                    plt.close()
            
            fname = os.path.join(
                self.results_dir,'testing','Data-{}_NNk.pdf'.format(ds))
            if os.path.isfile(fname):
                print("did not overwrite file {}".format(fname))
            else:
                plt.figure()
                colors = plt.cm.jet(np.linspace(0,1,4))
                plt.plot(range(1, len(kcenterm[i])+1), 
                         np.ones_like(kcenterm[i]), label='best accuracy', linestyle=':', color='g')
                for i in range(1,5):
                    plt.plot(range(1, len(means_kcenter[ds][i][1])+1), 
                             means_kcenter[ds][i][1], label = 'NN-%i + dotted worst'%i, color=colors[i-1])
                    plt.plot(range(1, len(means_kcenter[ds][i][1])+1), 
                             [kinter(i) for _ in range(len(means_kcenter[ds][i][1]))], linestyle=':', color=colors[i-1])
                plt.legend(loc='lower right', fontsize='xx-small')
                plt.title('Center assign Accuracy')
                plt.xlabel('time', fontsize='small')
                plt.tight_layout()
                if not(manual_mode):
                    self.savefig_autodpi(fname,
                        bbox_inches='tight')
                    plt.close()
        
        kwargs = {
            'means' : means,
            'stds' : stds,
            'means_ssim' : means_ssim,
            'stds_ssim' : stds_ssim,
            'means_kcenter' : means_kcenter,
            'stds_kcenter' : stds_kcenter,
            'mean_abs_err' : mean_abs_err,
            'mean_rel_err' : mean_rel_err,
            'abs_length' : abs_length,
            'rel_length' : rel_length,
            'mts_results' : mts_results}
        if self.add_centercount:
            kwargs = {
                **kwargs,
                **{'pio_centers' : pio_centers}}
            
        np.savez(os.path.join(self.results_dir, 'test_results'),
                 **kwargs)
        
        if self.add_classifier:
            for clsn in list(self.classifier.keys()):
                np.savez(os.path.join(self.results_dir,
                                      'test_classresults_%s.npz'%clsn),
                         **{'class_count_in' : class_count_in[clsn],
                            'class_results_in' : class_results_in[clsn],
                            'class_count_out' : class_count_out[clsn],
                            'class_results_out' : class_results_out[clsn],
                            'class_IOchange' : class_IOchange[clsn],
                            'class_TIchange' : class_TIchange[clsn],
                            'class_TOchange' : class_TOchange[clsn],
                            'confusion_classes' : confusion_classes[clsn]})
        
        print('[%s - END] Testing.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def set_description(self, ax, legend_loc='upper center', title='title', 
                        fontsize='small'):
        ax.legend(loc=legend_loc, fontsize=fontsize, 
                  ncol=int(len(ax.get_legend_handles_labels()[1])//3) + int(
                      len(ax.get_legend_handles_labels()[1])%3!=0))
        ax.set_title(title)
    
    def adjust_tick_labels(self,args, vmin, vmax, lmin, lmax, minor=False, 
                           ftype_label=None, maxm1=True):
        # adjusts the labels by affine transform (vmin, vmax) <-> (lmin, lmax)
        # between coord positions (vmin, vmax) and labels (lmin, lmax)
        e_ftype_label = lambda x: 'e'.join(
            [('%.0e'%x).split('e')[0],
             str(int(('%.0e'%x).split('e')[1]))])
        if len(args['ticks'])==0:
            args['ticks'] = [vmin]
        if len(args['labels'])==0:
            if not(minor):
                args['labels'] = [[ftype_label(lmin), e_ftype_label(lmin)][
                    int(float(ftype_label(lmin))==0)]]
            else:
                args['labels'] = [None]
        if args['ticks'][-1] != vmax:
            args['ticks'] = list(args['ticks'])[
                :len(list(args['ticks']))-1+int(maxm1)]+[vmax]
        if not(minor) and args['labels'][-1]!=[ftype_label(lmax),
                                               e_ftype_label(lmax)][
            int(float(ftype_label(lmax))==0)]:
            args['labels'] = list(args['labels'])[
                :len(list(args['ticks']))-1+int(maxm1)]+[[ftype_label(lmax),
                                                    e_ftype_label(lmax)][
                int(float(ftype_label(lmax))==0)]]
        else:
            if len(args['labels'])<len(args['ticks']):
                # add one missing last label
                args['labels'] = list(args['labels'])+[None]
        return args
            
    def format_axis(self, ax, vmin=0, vmax=40, step=None, lmin=None, lmax=None, 
                    axis='x', ax_label=None, type_labels='int', minor=True, 
                    margin=[0,0], fontsize='small', trace=False, 
                    more2_ticks = True, maxm1=True):
        # type_labels is 'int' or '%.2f' or other amount of decimal points to display floats
        # margin=list[margin_min, margin_max] is whether adding extra axis at the extrema
        # margin_min and margin_max should be ints and correspond to how many 
        # half-steps (step/2) to add
        assert any([value is not None for value in [vmin, vmax]]), "vmin and vmax should not be None"
        if axis=='y' and ax.get_yscale()=='log':
            y_min_log = 10**np.floor(np.log10(ax.get_ylim()[0]))
            if trace:
                print('get_min_y', ax.get_ylim()[0])
                print('y_min_log', y_min_log)
        int_ftype_label = lambda x: int(x)
        e_ftype_label = lambda x: ['e'.join(
            [('%.0e'%x).split('e')[0],
             str(int(('%.0e'%x).split('e')[1]))]), 0][
                 int(float('e'.join(
                     [('%.0e'%x).split('e')[0],
                      str(int(('%.0e'%x).split('e')[1]))]))==0)]
        if type_labels=='int':
            ftype_label = int_ftype_label
        elif 'e' in type_labels:
            ftype_label = lambda x: 'e'.join(
                [(type_labels%x).split('e')[0],
                 str(int((type_labels%x).split('e')[1]))])
        else:
            ftype_label = lambda x: type_labels%x
        if lmin is None:
            lmin=vmin
        if lmax is None:
            lmax=vmax
        assert not(any([np.isnan(value) for value in [vmin, vmax, lmin, lmax]])), "vmin, vmax, lmin and lmax should not be np.nan"
        assert (vmax==vmin) == (lmax==lmin), "vmin==vmax should have the same state than lmin==lmax"
        if step is None:
            step = (lmax-lmin)/4
        # Ensure proper values for step and tstep
        if lmax==lmin or step==0:
            step = 1
            tstep = [np.divide(step,lmin),1][int(lmin==0)]
            margin = [1,1]
        else:
            tstep = (vmax-vmin)/(lmax-lmin)*step
        # print('vmax', vmax)
        # print('vmin', vmin)
        # print('lmax', lmax)
        # print('lmin', lmin)
        args = {'ticks':[vmin-k*tstep/2 for k in range(1, margin[0]+1) if (minor or k%2==0 or k==margin[0])][::-1] + list(
            np.arange(vmin,vmax,tstep)) + [
                vmax+k*tstep/2 for k in range(1,margin[1]+1) if (minor or k%2==0 or k==margin[1])],
                'labels':[
                    [[ftype_label(lmin-k*step/2), e_ftype_label(lmin-k*step/2)][
                        int(float(ftype_label(lmin-k*step/2))==0)] for k in range(1, margin[0]+1) if (k%2==0 or k==margin[0])][::-1], 
                    [None]*margin[0]][int(minor)] + [
                    [ftype_label(v),e_ftype_label(v)][
                        int(float(ftype_label(v))==0)] for v in np.arange(lmin,lmax,step)] + [
                        [[ftype_label(lmax+k*step/2), e_ftype_label(lmax+k*step/2)][
                            int(float(ftype_label(lmax+k*step/2))==0)]for k in range(1,margin[1]+1) if (k%2==0 or k==margin[1])], 
                        [None]*margin[1]][int(minor)]}
        if trace:
            print('args', args)
        assert len([va for va in args['labels'] if va is not None])>1, "Too few labels for {} : {}".format('major', args['labels'])
        args = self.adjust_tick_labels(args, vmin, vmax, lmin, lmax,
                                       ftype_label=ftype_label, maxm1=maxm1)
        assert len([va for va in args['labels'] if va is not None])>1, "Too few labels for {} : {}".format('major', args['labels'])
        if not(more2_ticks):
            notnones = [va is not None for va in args['labels']]
            firstnn = notnones.index(True)
            lastnn = len(notnones)-notnones[::-1].index(True)-1
            assert firstnn!=lastnn, "only one non-None value"
            args = {
                'ticks': args['ticks'],
                'labels': [[None,v][int(iv in [firstnn, lastnn])
                    ] for iv,v in enumerate(args['labels'])]}
        assert len([va for va in args['labels'] if va is not None])>1, "Too few labels for {} : {}".format('major', args['labels'])
        if trace:
            print('args', args)
        if minor:
            args_minor = {'ticks':np.arange(vmin,vmax,tstep/2),
                          'labels':[None]*len(np.arange(vmin,vmax,tstep/2))}
            if trace:
                print('args_minor', args_minor)
            args_minor = self.adjust_tick_labels(args_minor, vmin, vmax, lmin, lmax,
                                                 minor=True, ftype_label=ftype_label,
                                                 maxm1=maxm1)
            if not(more2_ticks):
                args_minor = {
                    'ticks': args_minor['ticks'],
                    'labels': [None]*len(args_minor['labels'])}
            if trace:
                print('args_minor', args_minor)
        
        if axis=='y' and ax.get_yscale()=='log':
            min_y_value = [y_value for y_value in args['ticks'] if y_value>=y_min_log]
            min_y_value = [min_y_value, [y_min_log*10]][int(len(min_y_value)==0)]
            min_y_value = np.min(min_y_value)
            if trace:
                print('min_y_value', min_y_value)
            args = {'ticks': [
                y_min_log*10**exp for exp in np.arange(0, np.log10(min_y_value/y_min_log))] + [
                y_value for i_value, y_value in enumerate(
                    args['ticks']) if y_value>=y_min_log],
                    'labels':[
                        [ftype_label(y_min_log*10**exp), e_ftype_label(y_min_log*10**exp)][
                            int(float(ftype_label(y_min_log*10**exp))==0)] for exp in np.arange(0, np.log10(min_y_value/y_min_log))] + [
                        args['labels'][i_value] for i_value, y_value in enumerate(
                            args['ticks']) if y_value>=y_min_log]}
            if trace:
                print('args', args)
            if minor:
                args_minor = {'ticks': [
                    y_min_log*10**exp for exp in np.arange(0, np.log10(min_y_value/y_min_log))] + [
                    y_value for i_value, y_value in enumerate(
                        args_minor['ticks']) if y_value>=y_min_log],
                        'labels':[
                            [ftype_label(y_min_log*10**exp), e_ftype_label(y_min_log*10**exp)][
                                int(float(ftype_label(y_min_log*10**exp))==0)] for exp in np.arange(0, np.log10(min_y_value/y_min_log))] + [
                            args_minor['labels'][i_value] for i_value, y_value in enumerate(
                                args_minor['ticks']) if y_value>=y_min_log]}
                if trace:
                    print('args_minor', args_minor)
        
        ax.grid(which='major', linestyle='--', axis=axis)
        if minor:
            ax.grid(which='minor', linestyle=':', axis=axis)
        if axis == 'x':
            ax.set_xticks(**args, fontsize=fontsize)
            if minor:
                ax.set_xticks(**args_minor, minor=True, fontsize=fontsize)
            ax.set_xlabel(ax_label, fontsize=fontsize)
        if axis == 'y':
            ax.set_yticks(**args, fontsize=fontsize)
            if minor:
                ax.set_yticks(**args_minor, minor=True, fontsize=fontsize)
            ax.set_ylabel(ax_label, fontsize=fontsize)
    
    def savelong_raw(self,last_raw_saving, ds, 
                     np_metrics_in, np_metrics_out, 
                     # class_results_in, class_results_out,
                     # class_count_in, class_count_out, 
                     # class_acc,
                     # mts_results, glob_mts_results, 
                     # pio_centers, globpio_centers,
                     mean_abs_err, mean_rel_err,
                     abs_length, rel_length,
                     psnrs_df, ssims_df, kcenters_df,
                     class_IOchange, class_TOchange, class_TIchange,
                     confusion_classes, end=False):
        if self.add_classifier:
            np_metrics_insave = {}
            np_metrics_outsave = {}
            atc_metric_save = {}
            for clsn in list(self.classifier.keys()):
                np_metrics_insave[clsn] = {
                    metric.name:{
                        'total':metric.total,
                        'count':metric.count
                        } for metric in np_metrics_in[clsn]}
                np_metrics_outsave[clsn] = {
                    metric.name:{
                        'total':metric.total,
                        'count':metric.count
                        } for metric in np_metrics_out[clsn]}
                atc_metric_save[clsn] = {
                    'count':self.atc_metric[clsn].count
                    }
        else:
            np_metrics_insave = {'noclassifier': None}
            np_metrics_outsave = {'noclassifier': None}
            atc_metric_save = {'noclassifier': None}
               
        mts_metrics_save = {}
        globmts_metrics_save = {}
        for clsn in list(self.classifier.keys()):
            for mn, sn in zip(
                    [self.mts_metrics, self.glob_mts_metrics],
                    [mts_metrics_save, globmts_metrics_save]):
                sn[clsn] = {
                    nm: vm.return_save()
                     for nm,vm in mn[clsn].metrics_dict.items()}
        
        if self.add_centercount:
            pio_metrics_save = {}
            globpio_metrics_save = {}
            for mn, sn in zip(
                    [self.center_counter_pio, self.glob_center_counter_pio],
                    [pio_metrics_save, globpio_metrics_save]):
                for clsn in list(self.classifier.keys()):
                    # Collect results
                    sn[clsn] = {
                        'prior': {
                            'total':mn[clsn].freqdict_prior.total,
                            'count':mn[clsn].freqdict_prior.count
                            },
                        'io': {
                            'total':mn[clsn].freqdict_io.total,
                            'count':mn[clsn].freqdict_io.count
                            },
                        'pio': {
                            'total':mn[clsn].batchinfo_pio.total,
                            'count':mn[clsn].batchinfo_pio.count
                            }
                        }
        # else:
        #     pio_metrics_save[clsn] = {'noclassifier': None}
        #     globpio_metrics_save[clsn] = {'noclassifier': None}
        
        # if self.add_classifier:
        #     # Collect results
        #     for clsn in list(self.classifier.keys()):
        #         class_results_in[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_in[clsn]}
        #         class_count_in[clsn][ds] = {metric.name:metric.count for metric in np_metrics_in[clsn]}
        #         print('Classifier metrics IN:\n', class_results_in[clsn][ds])
        #         class_results_out[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_out[clsn]}
        #         class_count_out[clsn][ds] = {metric.name:metric.count for metric in np_metrics_out[clsn]}
        #         print('Classifier metrics OUT:\n', class_results_out[clsn][ds])
        #         class_acc[clsn][ds] = self.atc_metric[clsn].results()
        # else:
        #     class_results_in = {'noclassifier': None}
        #     class_count_in = {'noclassifier': None}
        #     class_results_out = {'noclassifier': None}
        #     class_count_out = {'noclassifier': None}
        #     class_acc = {'noclassifier': None}
        
        # # mts_metrics
        # for clsn in list(self.classifier.keys()):
        #     mts_results[clsn][ds] = {
        #         'by_no':self.mts_metrics[clsn].result_by_no(),
        #         'by_1':self.mts_metrics[clsn].result_by_1(),
        #         'by_1_2':self.mts_metrics[clsn].result_by_1_2(),
        #         'by_1_3':self.mts_metrics[clsn].result_by_1_3(),
        #         'by_1_2_3':self.mts_metrics[clsn].result_by_1_2_3()}
            
        #     glob_mts_results[clsn][ds] = {
        #         'by_no':self.glob_mts_metrics[clsn].result_by_no(),
        #         'by_1':self.glob_mts_metrics[clsn].result_by_1(),
        #         'by_1_2':self.glob_mts_metrics[clsn].result_by_1_2(),
        #         'by_1_3':self.glob_mts_metrics[clsn].result_by_1_3(),
        #         'by_1_2_3':self.glob_mts_metrics[clsn].result_by_1_2_3()}
        
        # Center counts is freq dict {lab{class{center}}}
        # if self.add_centercount:
        #     for clsn in list(self.classifier.keys()):
        #         pio_centers[clsn][ds] = {
        #             'cond_no':self.center_counter_pio[clsn].result_cond_no(),
        #             'cond_1':self.center_counter_pio[clsn].result_cond_1(),
        #             'cond_1_2':self.center_counter_pio[clsn].result_cond_1_2(),
        #             'cond_1_3':self.center_counter_pio[clsn].result_cond_1_3(),
        #             'cond_1_2_3':self.center_counter_pio[clsn].result_cond_1_2_3()}
                
        #         globpio_centers[clsn][ds] = {
        #             'cond_no':self.glob_center_counter_pio[clsn].result_cond_no(),
        #             'cond_1':self.glob_center_counter_pio[clsn].result_cond_1(),
        #             'cond_1_2':self.glob_center_counter_pio[clsn].result_cond_1_2(),
        #             'cond_1_3':self.glob_center_counter_pio[clsn].result_cond_1_3(),
        #             'cond_1_2_3':self.glob_center_counter_pio[clsn].result_cond_1_2_3()}
        # else:
        #     pio_centers['noclassifier'][ds] = [None]
        #     globpio_centers['noclassifier'][ds] = [None]
        
        # Outputs or test before plots:
        # mean_abs_err
        # mean_rel_err
        # psnrs_df
        # ssims_df
        # kcenters_df
        # class_IOchange
        # class_TIchange
        # class_TOchange
        # class_results_in
        # class_results_out
        # confusion_classes
        # mts_results
        # pio_centers
        
        # npRagged Trick to overcome a bug with np.asarray dtype='object' 
        # to store a ragged array when the first dim have same length 
        # but the next ones may differ
        kwargs = {
            'end': end,
            'last_raw_saving' : last_raw_saving,
            'mean_abs_err' : mean_abs_err[ds],
            'mean_rel_err' : mean_rel_err[ds],
            'abs_length' : abs_length[ds],
            'rel_length' : rel_length[ds],
            'psnrs_df': {'df':psnrs_df},
            'ssims_df': {'df':ssims_df},
            'kcenters_df': kcenters_df,
            'mts_metrics_save' : mts_metrics_save,
            'globmts_metrics_save' : globmts_metrics_save}
        if self.add_centercount:
            kwargs = {
                **kwargs,
                **{'pio_metrics_save' :  pio_metrics_save,
                   'globpio_metrics_save' : globpio_metrics_save}}
                   # 'pio_centers' : {clsn:pio_centers[clsn][ds] for clsn in pio_centers.keys()},
                   # 'globpio_centers' : {clsn:globpio_centers[clsn][ds] for clsn in pio_centers.keys()}}}
        if self.add_classifier:
            kwargs = {
                **kwargs,
                **{'np_metrics_insave' : np_metrics_insave,
                   'np_metrics_outsave' : np_metrics_outsave,
                   'atc_metric_save' : atc_metric_save,
                   # 'class_count_in' : {clsn:class_count_in[clsn][ds] for clsn in class_count_in.keys()},
                   # 'class_results_in' : {clsn:class_results_in[clsn][ds] for clsn in class_results_in.keys()},
                   # 'class_count_out' : {clsn:class_count_out[clsn][ds] for clsn in class_count_out.keys()},
                   # 'class_results_out' : {clsn:class_results_out[clsn][ds] for clsn in class_results_out.keys()},
                   # 'class_acc' : {clsn:class_acc[clsn][ds] for clsn in class_acc.keys()},
                   'class_IOchange' : {clsn:class_IOchange[clsn][ds] for clsn in class_IOchange.keys()},
                   'class_TIchange' : {clsn:class_TIchange[clsn][ds] for clsn in class_TIchange.keys()},
                   'class_TOchange' : {clsn:class_TOchange[clsn][ds] for clsn in class_TOchange.keys()},
                   'confusion_classes' : confusion_classes}}
        
        if os.path.isfile(os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds))):
            os.rename(os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds)), 
                      os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds)))
        if os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))):
            os.rename(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), 
                      os.path.join(self.results_dir, '_test_RAW_{}.npz'.format(ds)))
        np.savez(os.path.join(self.results_dir, 'test_RAW_{}'.format(ds)),
                 **kwargs)
        if os.path.isfile(os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds))):
            os.remove(os.path.join(self.results_dir, '__test_RAW_{}.npz'.format(ds)))
    
    def long_test(self):
        if not os.path.exists(os.path.join(self.results_dir, 'testing_long')):
            os.makedirs(os.path.join(self.results_dir, 'testing_long'))
        if self.model_type in ['LSTM', 'LSTMS', 'GRU', 'GRUS', 'NBeats']:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(True, True))
            self.model_instance(True, True)
        else:
            print('[START] Loading Model for predict - train_bn %s - inference_only %s'%(False, True))
            self.model_instance(False, True)
        # test_ds = self.predict_ds.split('_')
        test_ds = self.test_ds
        if os.path.isfile(os.path.join(self.results_dir + 'test_results.npz')):
            results = np.load(os.path.join(self.results_dir, 'test_results.npz'), allow_pickle = True)
            means = results['means'].all()
            stds = results['stds'].all()
            means_ssim = results['means_ssim'].all()
            stds_ssim = results['stds_ssim'].all()
            means_kcenter = results['means_kcenter'].all()
            stds_kcenter = results['stds_kcenter'].all()
            mean_abs_err = results['mean_abs_err'].all()
            mean_rel_err = results['mean_rel_err'].all()
            abs_length = results['abs_length'].all()
            rel_length = results['rel_length'].all()
            mts_results = results['mts_results'].all()
            glob_mts_results = results['glob_mts_results'].all()
            if self.add_classifier:
                class_count_in = {}
                class_results_in = {}
                class_count_out = {}
                class_results_out = {}
                class_acc = {}
                class_IOchange = {}
                class_TIchange = {}
                class_TOchange = {}
                confusion_classes = {}
                for clsn in list(self.classifier.keys()):
                    if os.path.isfile(os.path.join(
                            self.results_dir, 
                            'test_classresults_%s.npz'%clsn)):
                        classresults =  np.load(os.path.join(
                            self.results_dir, 
                            'test_classresults_%s.npz'%clsn), 
                            allow_pickle = True)
                        class_count_in[clsn] = classresults['class_count_in'].all()
                        class_results_in[clsn] = classresults['class_results_in'].all()
                        class_count_out[clsn] = classresults['class_count_out'].all()
                        class_results_out[clsn] = classresults['class_results_out'].all()
                        class_acc[clsn] = classresults['class_acc'].all()
                        class_IOchange[clsn] = classresults['class_IOchange'].all()
                        class_TIchange[clsn] = classresults['class_TIchange'].all()
                        class_TOchange[clsn] = classresults['class_TOchange'].all()
                        confusion_classes[clsn] = classresults['confusion_classes']
                        classresults.close()
                    else:
                        class_count_in[clsn] = {}
                        class_results_in[clsn] = {}
                        class_count_out[clsn] = {}
                        class_results_out[clsn] = {}
                        class_acc[clsn] = {}
                        class_IOchange[clsn] = {}
                        class_TIchange[clsn] = {}
                        class_TOchange[clsn] = {}
                        confusion_classes[clsn] = [None]
            else:
                class_count_in = {'noclassifier' : {}}
                class_results_in = {'noclassifier' : {}}
                class_count_out = {'noclassifier' : {}}
                class_results_out = {'noclassifier' : {}}
                class_IOchange = {'noclassifier' : {}}
                class_TIchange = {'noclassifier' : {}}
                class_TOchange = {'noclassifier' : {}}
                confusion_classes = {'noclassifier': [None]}
            if self.add_centercount:
                pio_centers = results['pio_centers'].all()
                globpio_centers = results['globpio_centers'].all()
            results.close()
        else:
            means, stds = {}, {}
            means_ssim, stds_ssim = {}, {}
            means_kcenter, stds_kcenter = {}, {}
            mean_abs_err, mean_rel_err = {},{}
            abs_length, rel_length = {}, {}
            mts_results = {clsn:{} for clsn in list(self.classifier.keys())}
            glob_mts_results = {clsn:{} for clsn in list(self.classifier.keys())}
            if self.add_classifier:
                class_count_in = {}
                class_results_in = {}
                class_count_out = {}
                class_results_out = {}
                class_acc = {}
                class_IOchange = {}
                class_TIchange = {}
                class_TOchange = {}
                confusion_classes = {}
                for clsn in list(self.classifier.keys()):
                    class_count_in[clsn] = {}
                    class_results_in[clsn] = {}
                    class_count_out[clsn] = {}
                    class_results_out[clsn] = {}
                    class_acc[clsn] = {}
                    class_IOchange[clsn] = {}
                    class_TIchange[clsn] = {}
                    class_TOchange[clsn] = {}
                    confusion_classes[clsn] = [None]
            else:
                class_count_in = {'noclassifier' : {}}
                class_results_in = {'noclassifier' : {}}
                class_count_out = {'noclassifier' : {}}
                class_results_out = {'noclassifier' : {}}
                class_IOchange = {'noclassifier' : {}}
                class_TIchange = {'noclassifier' : {}}
                class_TOchange = {'noclassifier' : {}}
                confusion_classes = {'noclassifier': [None]}
            if self.add_centercount:
                if self.add_classifier:
                    pio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
                    globpio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
                else:
                    pio_centers = {'noclassifier':{}}
                    globpio_centers = {'noclassifier':{}}
        
        print('test_ds', test_ds)
        if 'TEL' in test_ds:
            test_ds = ['TEL'] + [ds for ds in test_ds if ds!='TEL']
        for ds in test_ds:
            if 'TEL' in ds:
                print('ds', ds)
                print('in data_pack', list(self.data_pack.keys()))
                assert ds.replace('TEL', 'TE') in self.data_pack, "ERROR: problem with the label of data to test, not present in the available data"
                print('[Predicting] Long Testing Samples')
                print('# of tests', len(self.data_pack[ds.replace('TEL', 'TE')][0]))
                print('[%s - START] Testing..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
                
                n = 0
                
                psnrs_df = pd.DataFrame()
                ssims_df =  pd.DataFrame()
                kcenters_df = {}
                for i in range(1, 7):
                    kcenters_df[i] = pd.DataFrame()
                
                mean_abs_err[ds] = np.zeros((len(self.feat_legends), 1))
                mean_rel_err[ds] = np.zeros((len(self.feat_legends), 1))
                abs_length[ds] = np.zeros((len(self.feat_legends), 1))
                rel_length[ds] = np.zeros((len(self.feat_legends), 1))
                
                if self.add_classifier:
                    np_all_metrics = [
                        NP_CategoricalCrossentropy, 
                        NP_BinaryCrossentropy, 
                        NP_CategoricalAccuracy, 
                        NP_BinaryAccuracy]
                    np_metrics_in = {}
                    np_metrics_out = {}
                    makedir = {}
                    for clsn, clsfier in self.classifier.items():
                        np_metrics_in[clsn] = [m for m in np_all_metrics if any(
                            [mm.name in m().name for mm in clsfier.model.model.compiled_metrics._metrics])]
                        np_metrics_in[clsn] = [mm(**a) for mm, a in zip(
                            np_metrics_in[clsn], 
                            [[{}, {'class_assign_fn':clsfier.model.np_assign_class}][int(
                                tf.keras.metrics.CategoricalAccuracy().name == m().name)] for m in np_metrics_in[clsn]])]
                        np_metrics_out[clsn] = [m for m in np_all_metrics if any(
                            [mm.name in m().name for mm in clsfier.model.model.compiled_metrics._metrics])]
                        np_metrics_out[clsn] = [mm(**a) for mm, a in zip(
                            np_metrics_out[clsn], 
                            [[{}, {'class_assign_fn':clsfier.model.np_assign_class}][int(
                                tf.keras.metrics.CategoricalAccuracy().name == m().name)] for m in np_metrics_out[clsn]])]
                        for metric in np_metrics_in[clsn]:
                            metric.reset_states()
                        for metric in np_metrics_out[clsn]:
                            metric.reset_states()
                        # confusion matrix IN OUT classification
                        if clsfier.noclass is not None:
                            class_IOchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                            class_TIchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                            class_TOchange[clsn][ds] = np.zeros([clsfier.nclass+1]*2, np.int64)
                            confusion_classes[clsn] = clsfier.classes + [clsfier.noclass]
                        else:
                            assert clsfier.noclass is None, "In and Out classifiers should have the same 'noclass'"
                            class_IOchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                            class_TIchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                            class_TOchange[clsn][ds] = np.zeros([clsfier.nclass]*2, np.int64)
                            confusion_classes[clsn] = clsfier.classes
                        makedir[clsn] = clsn
                else:
                    makedir = {'noclassifier':'noclassifier'}
                    np_metrics_in = {'noclassifier':None}
                    np_metrics_out = {'noclassifier':None}
                
                for _, mkd in makedir.items():
                    if not os.path.exists(os.path.join(self.results_dir, 'testing_long', mkd)):
                        os.makedirs(os.path.join(self.results_dir, 'testing_long', mkd))
                
                for clsn in list(self.classifier.keys()):
                    self.mts_metrics[clsn].reset()
                    self.glob_mts_metrics[clsn].reset()
                
                if self.add_centercount:
                    for clsn in list(self.classifier.keys()):
                        self.center_counter_pio[clsn].reset()
                        self.glob_center_counter_pio[clsn].reset()
                
                # input time duration is arbitrary self.label_length in this model
                stride = int(self.label_length*(self.mask_ratio))
                end_raw = False
                if os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))):
                    results = np.load(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), allow_pickle = True)
                    if 'end' in results.keys():
                        end_raw = results['end']
                    else:
                        end_raw = False
                    last_raw_saving = results['last_raw_saving']
                    print("loading previous raw results: last saved index is {}".format(last_raw_saving))
                    print("length of test is {}".format(len(self.data_pack[ds.replace('TEL', 'TE')][0])))
                    mean_abs_err[ds] = results['mean_abs_err']
                    mean_rel_err[ds] = results['mean_rel_err']
                    abs_length[ds] = results['abs_length']
                    rel_length[ds] = results['rel_length']
                    psnrs_df = results['psnrs_df'].all()['df']
                    ssims_df = results['ssims_df'].all()['df']
                    kcenters_df = results['kcenters_df'].all()
                    # for clsn in list(self.classifier.keys()):
                    #     mts_results[clsn][ds] = results['mts_results'].all()[clsn]
                    #     glob_mts_results[clsn][ds] = results['glob_mts_results'].all()[clsn]
                    
                    if self.add_classifier:
                        np_metrics_insave = results['np_metrics_insave'].all()
                        np_metrics_outsave = results['np_metrics_outsave'].all()
                        atc_metric_save = results['atc_metric_save'].all()
                        for clsn in list(self.classifier.keys()):
                            for metric in np_metrics_in[clsn]:
                                metric.total = np_metrics_insave[clsn][
                                    metric.name]['total']
                                metric.count = np_metrics_insave[clsn][
                                    metric.name]['count']
                            for metric in np_metrics_out[clsn]:
                                metric.total = np_metrics_outsave[clsn][
                                    metric.name]['total']
                                metric.count = np_metrics_outsave[clsn][
                                    metric.name]['count']
                            self.atc_metric[clsn].count = atc_metric_save[clsn]['count']
                            class_IOchange[clsn][ds] = results['class_IOchange'].all()[clsn]
                            class_TIchange[clsn][ds] = results['class_TIchange'].all()[clsn]
                            class_TOchange[clsn][ds] = results['class_TOchange'].all()[clsn]
                            confusion_classes[clsn] = results['confusion_classes'].all()[clsn]
                           
                    for clsn in list(self.classifier.keys()):
                        for mn, sn in zip(
                                [self.mts_metrics, 
                                 self.glob_mts_metrics],
                                ['mts_metrics_save',
                                 'globmts_metrics_save']):
                            metrics_save = results[sn].all()
                            for nm,vm in mn[clsn].metrics_dict.items():
                                vm.from_saved(metrics_save[clsn][nm])
                    
                    if self.add_centercount:
                        for nm, sn in zip(
                                [self.center_counter_pio, 
                                 self.glob_center_counter_pio],
                                ['pio_metrics_save',
                                 'globpio_metrics_save']):
                            metrics_save = results[sn].all()
                            for clsn in list(self.classifier.keys()):
                                nm[clsn].freqdict_prior.total = metrics_save[
                                        clsn]['prior']['total']
                                nm[clsn].freqdict_prior.count = metrics_save[
                                        clsn]['prior']['count']
                                nm[clsn].freqdict_io.total = metrics_save[
                                        clsn]['io']['total']
                                nm[clsn].freqdict_io.count = metrics_save[
                                        clsn]['io']['count']
                                nm[clsn].batchinfo_pio.total = metrics_save[
                                        clsn]['pio']['total']
                                nm[clsn].batchinfo_pio.count = metrics_save[
                                        clsn]['pio']['count']
                    results.close()
                else:
                    print("did not find any raw saving for this dataset")
                    last_raw_saving = 0
                
                if not(end_raw) and last_raw_saving < len(self.data_pack[ds.replace('TEL', 'TE')][0]):
                    start_timer = time.time()
                    firstsave = True
                    for seq, pos in tqdm(zip(self.data_pack[ds.replace('TEL', 'TE')][0], self.data_pack[ds.replace('TEL', 'TE')][1])):
                        n += 1
                        if n<=last_raw_saving:
                            print("skipping step # {}, will skip all steps until step # {}".format(n, last_raw_saving))
                            continue
                    
                        seq_pred = np.expand_dims(np.zeros(seq.shape), axis=-1)
                        pos_accum = np.zeros((seq_pred.shape[0]))
                        data_chunked, pos_chunked = chunkdata_for_longpredict(seq, pos, self.label_length, self.mask_ratio)
            #                if self.c_dim == 1:
            #                    data_chunked = np.tile(data_chunked, [1, 1, 1, 3])
                        patch_nb = data_chunked.shape[0]
                        idx = 0
                        last = ((seq.shape[0]-self.label_length) % stride) # 0 if no strange patch at the end of the sequence, last resting timevalues otherwise
                        
                        lab = self.label_from_pos(pos[['.' in str(p) for p in pos].index(True)],
                                                  predictonly=False)
                        
                        if self.add_classifier:
                            pred_class_in = {}
                            pred_class_out = {}
                            catlab = {}
                            for clsn, clsfier in self.classifier.items():
                                pred_class_in[clsn] = np.array([0]*clsfier.nclass, np.float32)
                                pred_class_out[clsn] = np.array([0]*clsfier.nclass, np.float32)
                                
                                catlab[clsn] = clsfier.catlabels_from_pos(
                                    [pos[['.' in str(p) for p in pos].index(True)]],
                                    predictonly=False)
                        
                        for data_patch, pos_patch in zip(data_chunked, pos_chunked):
                            
                            data_patch = np.expand_dims(data_patch, axis = 0)
                            assert idx == pos_patch[0], "ERROR in parsing image chunks"
                            mask = np.ones(data_patch.shape)
                            mask[:, -int(mask.shape[1] * self.mask_ratio):, :, :] = 0
                            
                            if idx == 0:
                                masked = deepcopy(data_patch)
                                masked[mask==0] = 1
                                pred_img = self.model.predict([masked, mask])
                                seq_pred[idx * stride : idx * stride + self.label_length, :, :] += pred_img[0]
                                pos_accum[idx * stride : idx * stride + self.label_length] += 1
                            elif idx == patch_nb-1 and last > 0:
                                masked = np.zeros(data_patch.shape)
                                masked[:, :-last, :, :] = pred_img[:, last:, :, :]
                                masked[mask==0] = 1
                                pred_img = self.model.predict([masked, mask])
                                seq_pred[-self.label_length:, :, :] += pred_img[0]
                                pos_accum[-self.label_length:] += 1
                            else:
                                masked = np.zeros(data_patch.shape)
                                masked[:, :-stride, :, :] = pred_img[:, stride:, :, :]
                                masked[mask==0] = 1
                                pred_img = self.model.predict([masked, mask])
                                seq_pred[idx * stride : idx * stride + self.label_length,:,:] += pred_img[0]
                                pos_accum[idx * stride : idx * stride + self.label_length] += 1
                                
                            if self.add_classifier:
                                # class prediction on input and output
                                class_mask = create_class_mask(data_patch, 
                                                                    self.mask_ratio, 
                                                                    self.random_ratio)
                                slclass_in = {}
                                slclass_out = {}
                                for clsn, clsfier in self.classifier.items():
                                    pcii = clsfier.model.predict([data_patch, class_mask])
                                    for metric in np_metrics_in[clsn]:
                                        metric.update_state(catlab[clsn],pcii)
                                    slclass_in[clsn] = clsfier.model.np_assign_class(
                                        pcii)
                                    pcoi = clsfier.model.predict([pred_img, class_mask])
                                    for metric in np_metrics_out[clsn]:
                                        metric.update_state(catlab[clsn],pcoi)
                                    slclass_out[clsn] = clsfier.model.np_assign_class(
                                        pcoi)
                                    self.atc_metric[clsn].update_states(
                                        lab, 
                                        slclass_in[clsn], 
                                        slclass_out[clsn], 
                                        idx)
                                    pred_class_in[clsn] += pcii[0]
                                    pred_class_out[clsn] += pcoi[0]
                            else:
                                slclass_in = {'noclassifier':['NoClass']}
                                slclass_out = {'noclassifier':['NoClass']}
                            
                            for clsn in list(self.classifier.keys()):
                                self.mts_metrics[clsn].update(
                                    ([lab], slclass_in[clsn], slclass_out[clsn]),
                                    (data_patch[:,-int(data_patch.shape[1]*self.mask_ratio):],
                                     pred_img[:,-int(pred_img.shape[1]*self.mask_ratio):]))
                            
                            if self.add_centercount:
                                for clsn in list(self.classifier.keys()):
                                    self.center_counter_pio[clsn].fit_batch(
                                        [lab], (slclass_in[clsn], slclass_out[clsn]),
                                        (data_patch[:,:-int(data_patch.shape[1]*self.mask_ratio)],
                                         data_patch[:,-int(data_patch.shape[1]*self.mask_ratio):],
                                         pred_img[:,-int(pred_img.shape[1]*self.mask_ratio):]))
                            
                            idx += 1
                            
                        seq_pred /= pos_accum[:, None, None]
                        seq_pred = seq_pred.squeeze()
                        
                        if self.add_classifier:
                            class_in = {}
                            class_out = {}
                            for clsn, clsfier in self.classifier.items():
                                pred_class_in[clsn] = pred_class_in[clsn]/idx
                                pred_class_out[clsn] = pred_class_out[clsn]/idx
                                
                                # get classifier tested
                                # If bugs: lookup pos > lab > catlab
                                class_true = clsfier.model.np_assign_class(catlab[clsn])
                                # _ = self.classifier.model.test_on_batch(
                                #     [ori, class_mask, pos], catlab)
                                class_in[clsn] = clsfier.model.np_assign_class([pred_class_in[clsn]])
                                # _ = self.classifier.model.test_on_batch(
                                #     [pred, class_mask, pos], catlab)
                                class_out[clsn] = clsfier.model.np_assign_class([pred_class_out[clsn]])
                                class_IOchange[clsn][ds] += confusion_matrix(
                                    class_in[clsn], class_out[clsn],
                                    labels=confusion_classes[clsn])
                                class_TIchange[clsn][ds] += confusion_matrix(
                                    class_true, class_in[clsn],
                                    labels=confusion_classes[clsn])
                                class_TOchange[clsn][ds] += confusion_matrix(
                                    class_true, class_out[clsn],
                                    labels=confusion_classes[clsn])
                        else:
                            class_in = {'noclassifier':['NoClass']}
                            class_out = {'noclassifier':['NoClass']}
                            class_IOchange[clsn][ds] = None
                            class_TIchange[clsn][ds] = None
                            class_TOchange[clsn][ds] = None
                        
                        for clsn in list(self.classifier.keys()):
                            self.glob_mts_metrics[clsn].update(
                                ([lab], class_in[clsn], class_out[clsn]),
                                (np.expand_dims(seq[self.label_length-int(self.label_length*self.mask_ratio):], axis=0),
                                 np.expand_dims(seq_pred[self.label_length-int(self.label_length*self.mask_ratio):], axis=0)))
                        
                        if self.add_centercount:
                            for clsn in list(self.classifier.keys()):
                                self.glob_center_counter_pio[clsn].fit_batch(
                                    [lab], (class_in[clsn], class_out[clsn]),
                                    (np.expand_dims(seq[:self.label_length-int(self.label_length*self.mask_ratio)], axis=0),
                                     np.expand_dims(seq[self.label_length-int(self.label_length*self.mask_ratio):], axis=0),
                                     np.expand_dims(seq_pred[self.label_length-int(self.label_length*self.mask_ratio):], axis=0)))
                        
                        psnrs = np.zeros(patch_nb)
                        ssims = np.zeros(patch_nb)
                        kcenters = {}
                        for i in range(1, 7):
                            kcenters[i] = np.zeros(patch_nb)
                        for idx in range(patch_nb):
                            # Average psnr, ssim and kcenters accuracies over a time duration of 'stride'
                            if idx == patch_nb-1 and last>0:
                                psnrs[idx] = -10.0 * np.log10(np.mean(np.square(seq_pred[-self.label_length:,:] - seq[-self.label_length:,:])))
                                ssims[idx] = np.mean([ssim(seq_pred[-self.label_length + t,:], seq[-self.label_length + t,:]) for t in range(self.label_length)])
                                for i in range(1, 7):
                                    kcenters[i][idx] = np.mean(kcentroids_equal(to_kcentroid_seq(seq_pred.squeeze()[-self.label_length:,:], k=i)[1], to_kcentroid_seq(seq.squeeze()[-self.label_length:,:], k=i)[1]))
                            else:
                                psnrs[idx] = -10.0 * np.log10(np.mean(np.square(seq_pred[idx * stride : idx * stride + self.label_length,:] - seq[idx * stride : idx * stride + self.label_length,:])))
                                ssims[idx] = np.mean([ssim(seq_pred[idx * stride + t,:], seq[idx * stride + t,:]) for t in range(self.label_length)])
                                for i in range(1, 7):
                                    kcenters[i][idx] = np.mean(kcentroids_equal(to_kcentroid_seq(seq_pred.squeeze()[idx * stride : idx * stride + self.label_length,:], k=i)[1], to_kcentroid_seq(seq.squeeze()[idx * stride : idx * stride + self.label_length,:], k=i)[1]))
                        psnr = -10.0 * np.log10(np.mean(np.square(seq_pred[self.label_length - int(self.label_length * self.mask_ratio):,:] - seq[self.label_length - int(self.label_length * self.mask_ratio):,:])))
                        ssimm = np.mean([ssim(seq_pred[self.label_length - int(self.label_length * self.mask_ratio) + t,:], seq[self.label_length - int(self.label_length * self.mask_ratio) + t,:]) for t in range(len(seq[self.label_length - int(self.label_length * self.mask_ratio) :,:]))])
                        kcenter = {}
                        for i in range(1, 7):
                            kcenter[i] = np.mean(kcentroids_equal(to_kcentroid_seq(seq_pred[self.label_length - int(self.label_length * self.mask_ratio):,:], k=i)[1], to_kcentroid_seq(seq[self.label_length - int(self.label_length * self.mask_ratio):,:], k=i)[1]))
                        psnrs = pd.DataFrame({'PSNR' : np.concatenate([[psnr], psnrs])})
                        psnrs_df = pd.concat([psnrs_df, psnrs], axis=1, join='outer')
                        ssims = pd.DataFrame({'SSIM' : np.concatenate([[ssimm], ssims])})
                        ssims_df = pd.concat([ssims_df, ssims], axis=1, join='outer')
                        for i in range(1, 7):
                            kcenters[i] = pd.DataFrame({'K-CENTERS%i'%i : np.concatenate([[kcenter[i]], kcenters[i]])})
                            kcenters_df[i] = pd.concat([kcenters_df[i], kcenters[i]], axis=1, join='outer')
                        
                        if self.with_features:
                            abs_err, rel_err = self.geterr_features_sequence(seq[self.label_length - int(self.label_length * self.mask_ratio):,:], seq_pred[self.label_length - int(self.label_length * self.mask_ratio):,:])
                            abs_err = np.expand_dims(abs_err, 0)
                            rel_err = np.expand_dims(rel_err, 0)
                            
                            mean_abs_err[ds], abs_length[ds] = self.sum_extendshape_ifnecess(mean_abs_err[ds], abs_err, abs_length[ds])
                            mean_rel_err[ds], rel_length[ds] = self.sum_extendshape_ifnecess(mean_rel_err[ds], rel_err, rel_length[ds])
                        else:
                            mean_abs_err[ds], abs_length[ds] = [None, None]
                            mean_rel_err[ds], rel_length[ds] = [None, None]
                        
                        if time.time()-start_timer > 3600 or firstsave:
                            start_timer = time.time()
                            firstsave = False
                            self.savelong_raw(n, ds, 
                                             np_metrics_in, np_metrics_out, 
                                             # class_results_in, class_results_out,
                                             # class_count_in, class_count_out, 
                                             # class_acc,
                                             # mts_results, glob_mts_results, 
                                             # pio_centers, globpio_centers,
                                             mean_abs_err, mean_rel_err,
                                             abs_length, rel_length,
                                             psnrs_df, ssims_df, kcenters_df,
                                             class_IOchange, class_TOchange, class_TIchange,
                                             confusion_classes)
                        
                        # if n > self.test_length:
                        if n > len(self.data_pack[ds.replace('TEL', 'TE')][0]) or self.debug:
                            print("Analyzed iterations: {}".format(n))
                            print("Iterations to perform: {}".format(
                                len(self.data_pack[ds.replace('TEL', 'TE')][0]
                                    )))
                            yn = input("End analyze? (y/n) ")
                            if yn=='y':
                                print("Analyze stopped")
                                break
                            else:
                                print("Analyze resumed")
                    
                # Post process and save
                if not(end_raw):
                    if self.with_features:
                        mean_abs_err[ds] /= abs_length[ds]
                        mean_rel_err[ds] /= rel_length[ds]
                    
                    self.savelong_raw(n, ds, 
                                     np_metrics_in, np_metrics_out, 
                                     # class_results_in, class_results_out,
                                     # class_count_in, class_count_out, 
                                     # class_acc,
                                     # mts_results, glob_mts_results, 
                                     # pio_centers, globpio_centers,
                                     mean_abs_err, mean_rel_err,
                                     abs_length, rel_length,
                                     psnrs_df, ssims_df, kcenters_df,
                                     class_IOchange, class_TOchange, class_TIchange,
                                     confusion_classes, end=True)
				
                if self.add_classifier:
                    # Collect results
                    for clsn in list(self.classifier.keys()):
                        class_results_in[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_in[clsn]}
                        class_count_in[clsn][ds] = {metric.name:metric.count for metric in np_metrics_in[clsn]}
                        print('Classifier metrics IN:\n', class_results_in[clsn][ds])
                        class_results_out[clsn][ds] = {metric.name:metric.result() for metric in np_metrics_out[clsn]}
                        class_count_out[clsn][ds] = {metric.name:metric.count for metric in np_metrics_out[clsn]}
                        print('Classifier metrics OUT:\n', class_results_out[clsn][ds])
                        class_acc[clsn][ds] = self.atc_metric[clsn].results()
                else:
                    class_results_in = {'noclassifier': None}
                    class_count_in = {'noclassifier': None}
                    class_results_out = {'noclassifier': None}
                    class_count_out = {'noclassifier': None}
                    class_acc = {'noclassifier': None}
                
                # mts_metrics
                for clsn in list(self.classifier.keys()):
                    mts_results[clsn][ds] = {
                        'by_no':self.mts_metrics[clsn].result_by_no(),
                        'by_1':self.mts_metrics[clsn].result_by_1(),
                        'by_1_2':self.mts_metrics[clsn].result_by_1_2(),
                        'by_1_3':self.mts_metrics[clsn].result_by_1_3(),
                        'by_1_2_3':self.mts_metrics[clsn].result_by_1_2_3()}
                    
                    glob_mts_results[clsn][ds] = {
                        'by_no':self.glob_mts_metrics[clsn].result_by_no(),
                        'by_1':self.glob_mts_metrics[clsn].result_by_1(),
                        'by_1_2':self.glob_mts_metrics[clsn].result_by_1_2(),
                        'by_1_3':self.glob_mts_metrics[clsn].result_by_1_3(),
                        'by_1_2_3':self.glob_mts_metrics[clsn].result_by_1_2_3()}
                
                # Center counts is freq dict {lab{class{center}}}
                if self.add_centercount:
                    for clsn in list(self.classifier.keys()):
                        pio_centers[clsn][ds] = {
                            'cond_no':self.center_counter_pio[clsn].result_cond_no(),
                            'cond_1':self.center_counter_pio[clsn].result_cond_1(),
                            'cond_1_2':self.center_counter_pio[clsn].result_cond_1_2(),
                            'cond_1_3':self.center_counter_pio[clsn].result_cond_1_3(),
                            'cond_1_2_3':self.center_counter_pio[clsn].result_cond_1_2_3()}
                        
                        globpio_centers[clsn][ds] = {
                            'cond_no':self.glob_center_counter_pio[clsn].result_cond_no(),
                            'cond_1':self.glob_center_counter_pio[clsn].result_cond_1(),
                            'cond_1_2':self.glob_center_counter_pio[clsn].result_cond_1_2(),
                            'cond_1_3':self.glob_center_counter_pio[clsn].result_cond_1_3(),
                            'cond_1_2_3':self.glob_center_counter_pio[clsn].result_cond_1_2_3()}
                # else:
                #     pio_centers['noclassifier'][ds] = [None]
                #     globpio_centers['noclassifier'][ds] = [None]
                
                # Plots
                print("Analize completed")
                if not('_' in ds):
                    print("Drawing output figures")
                    if self.add_classifier:
                        # Collect results
                        for clsn in list(self.classifier.keys()):
                            self.plot_classmetrics(
                                class_results_in[clsn][ds], class_results_out[clsn][ds],
                                class_IOchange[clsn][ds], class_TIchange[clsn][ds], class_TOchange[clsn][ds],
                                confusion_classes[clsn],
                                save_name=os.path.join('testing_long', makedir[clsn],
                                    'Data-{}_confusionM.png'.format(ds)))
                            self.plot_classacc(class_acc[clsn][ds],
                                               (self.classifier[clsn].nolabel, 
                                                self.classifier[clsn].noclass),
                                               save_name=os.path.join(
                                                   'testing_long', makedir[clsn],
                                                   'Data-{}_class_accuracy.png'.format(ds)))
                    
                    # mts_metrics
                    for clsn in list(self.classifier.keys()):
                        self.plot_mtsres(
                            mts_results[clsn][ds], 
                            save_name=os.path.join(
                                'testing_long', makedir[clsn],
                                'Data-{}_slides_mts_results.png'.format(ds)))
                        self.plot_mtsres(
                            glob_mts_results[clsn][ds], 
                            save_name=os.path.join(
                                'testing_long', makedir[clsn],
                                'Data-{}_global_mts_results.png'.format(ds)))
                    
                    # Center counts is freq dict {lab{class{center}}}
                    if self.add_centercount:
                        for clsn in list(self.classifier.keys()):
                            for lbl in pio_centers[clsn][ds]['cond_1']['centers']['c0'].keys():
                                self.plot_clsres_one_label(
                                    pio_centers[clsn][ds], lbl, 
                                    save_name=os.path.join(
                                        'testing_long', makedir[clsn],
                                        'Data-{}_slidescentercount.png'.format(ds)))
                            for lbl in globpio_centers[clsn][ds]['cond_1']['centers']['c0'].keys():
                                self.plot_clsres_one_label(
                                    globpio_centers[clsn][ds], lbl, 
                                    save_name=os.path.join(
                                        'testing_long', makedir[clsn],
                                        'Data-{}_globalcentercount.png'.format(ds)))
                
                if self.with_features:
                    self.ploterr_features(mean_abs_err[ds], self.feat_legends,
                                          os.path.join(
                                              'testing_long','Err{}_feats_{}.png'.format('ABS', ds)))
                    self.ploterr_features(mean_rel_err[ds], self.feat_legends,
                                          os.path.join(
                                              'testing_long','Err{}_feats_{}.png'.format('REL', ds)))
                
                means[ds] = psnrs_df.mean(1).values
                print('Means PSNR Global and by chunks :')
                print(means[ds])
                means_ssim[ds] = ssims_df.mean(1).values
                print('Means SSIM Global and by chunks :')
                print(means_ssim[ds])
                means_kcenter[ds] = {}
                for i in range(1, 7):
                    means_kcenter[ds][i] = kcenters_df[i].mean(1).values
                    print('Means K-CENTERS-%i Global and by chunks :'%i)
                    print(means_kcenter[ds][i])
                fig = plt.figure(constrained_layout=True, figsize=(40, 40))
                widths, heights = [[2, 2], [1, 1, 1, 1]]
                # spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)
                spec = gridspec.GridSpec(nrows=4, ncols=2, figure=fig, 
                                         width_ratios=widths, 
                                         height_ratios=heights)
                axes = {}
                for row in range(4):
                    for col in range(2):
                        axes[row, col] = fig.add_subplot(spec[row, col])
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=means[ds][1:], 
                    tofit=stride*means[ds][1:]+1)
                axes[0, 0].plot(np.arange(vstart, vend, vstep), 
                                means[ds][1:], label='PSNR')
                self.format_axis(axes[0, 0], vmin=0, vmax=40, step = 10, axis = 'y', type_labels='int')
                self.format_axis(axes[0, 0], vmin=vmin, vmax=len(means[ds][1:]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[0, 0], legend_loc='upper center', title='time chunks predictions', fontsize='x-small')
                vmin, vstart, vstep, vend = self.adjust_xcoord(
                    toshow=means_ssim[ds][1:], 
                    tofit=stride*means_ssim[ds][1:]+1)
                axes[0, 1].plot(np.arange(vstart, vend, vstep), 
                                means_ssim[ds][1:], label='SSIM')
                axes[0, 1].plot(np.arange(vstart, vend, vstep), 
                                np.ones_like(means_ssim[ds][1:]), label='best', linestyle=':', color='g')
                self.format_axis(axes[0, 1], vmin=0, vmax=1, step = 0.2, axis = 'y', type_labels='%.1f', margin=[0,1])
                self.format_axis(axes[0, 1], vmin=vmin, vmax=len(means_ssim[ds][1:]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                self.set_description(axes[0, 1], legend_loc='upper center', title='time chunks predictions', fontsize='x-small')
                for i in range(1,7):
                    row = (i - 1) // 2 + 1
                    col = (i - 1) % 2
                    vmin, vstart, vstep, vend = self.adjust_xcoord(
                        toshow=means_kcenter[ds][i][1:], 
                        tofit=stride*means_kcenter[ds][i][1:]+1)
                    axes[row, col].plot(range(1, len(means_kcenter[ds][i][1:])+1), 
                                        means_kcenter[ds][i][1:], label='%i-NearestCenters'%i)
                    axes[row, col].plot(range(1, len(means_kcenter[ds][i][1:])+1), 
                                        [kinter(i) for _ in range(len(means_kcenter[ds][i][1:]))], label='%i-RandomBaseground'%i, linestyle=':', color='r')
                    axes[row, col].plot(range(1, len(means_kcenter[ds][i][1:])+1), 
                                        np.ones_like(means_kcenter[ds][i][1:]), label='best accuracy', linestyle=':', color='g')
                    self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, axis = 'y', ax_label='accuracy', type_labels='%.1f', margin=[0,1])
                    self.format_axis(axes[row, col], vmin=0, vmax=len(means_kcenter[ds][i][1:]), step = 10, axis = 'x', ax_label='time', type_labels='int')
                    self.set_description(axes[row, col], legend_loc='upper center', title='time chunks predictions', fontsize='x-small')
                # spec.tight_layout(fig)
                
                if not(manual_mode):
                    self.savefig_autodpi(os.path.join(
                        self.results_dir,'testing_long','Data-%s.png'%ds),
                        bbox_inches='tight')
                    plt.close()
                print('[%s - END] Testing.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
                
                kwargs = {
                    'means' : means,
                    'stds' : stds,
                    'means_ssim' : means_ssim,
                    'stds_ssim' : stds_ssim,
                    'means_kcenter' : means_kcenter,
                    'stds_kcenter' : stds_kcenter,
                    'mean_abs_err' : mean_abs_err,
                    'mean_rel_err' : mean_rel_err,
                    'abs_length' : abs_length,
                    'rel_length' : rel_length,
                    'mts_results' : mts_results,
                    'glob_mts_results' : glob_mts_results
                    }
                if self.add_centercount:
                    kwargs = {
                        **kwargs,
                        **{'pio_centers' : pio_centers,
                           'globpio_centers' : globpio_centers}}
                    
                np.savez(os.path.join(self.results_dir, 'test_results'),
                         **kwargs)
                
                if self.add_classifier:
                    for clsn in list(self.classifier.keys()):
                        np.savez(os.path.join(self.results_dir, 
                                              'test_classresults_%s.npz'%clsn),
                                 **{'class_count_in' : class_count_in[clsn],
                                    'class_results_in' : class_results_in[clsn],
                                    'class_count_out' : class_count_out[clsn],
                                    'class_results_out' : class_results_out[clsn],
                                    'class_acc' : class_acc[clsn],
                                    'class_IOchange' : class_IOchange[clsn],
                                    'class_TIchange' : class_TIchange[clsn],
                                    'class_TOchange' : class_TOchange[clsn],
                                    'confusion_classes' : confusion_classes[clsn]})
                
                fname = os.path.join(self.results_dir, 'testing_long', 
                                         'Data-%s_PSNR_SSIM.pdf'%ds)
                if os.path.isfile(fname):
                    print("did not overwrite file {}".format(fname))
                else:
                    fig, ax1 = plt.subplots()
                    color = 'tab:red'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('PSNR', color=color)
                    ax1.plot([v*self.mask_ratio*self.label_length for v in range(1, len(means[ds][1])+1)], 
                             means[ds][1:], color=color)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.grid(which='major', linestyle=':')
                    ax2 = ax1.twinx()
                    color = 'tab:blue'
                    ax2.set_ylabel('SSIM', color=color)
                    ax2.plot([v*self.mask_ratio*self.label_length for v in range(1, len(means_ssim[ds][1])+1)], 
                             means_ssim[ds][1:], color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.grid(which='major', linestyle=':')
                    plt.tight_layout()
                    if not(manual_mode):
                        self.savefig_autodpi(fname,
                                             bbox_inches='tight')
                        plt.close()
                
                fname = os.path.join(self.results_dir, 'testing_long',
                                         'Data-%s_NNk.pdf'%ds)
                if os.path.isfile(fname):
                    print("did not overwrite file {}".format(fname))
                else:
                    plt.figure()
                    colors = plt.cm.jet(np.linspace(0,1,4))
                    for i in range(1,5):
                        plt.plot(range(1, len(means_kcenter['%s'%ds][i][1:])+1), 
                                 means_kcenter['%s'%ds][i][1:], label = 'NN-%i'%i, color=colors[i-1])
                    plt.legend(loc='right', fontsize='xx-small')
                    plt.tight_layout()
                    if not(manual_mode):
                        self.savefig_autodpi(fname,
                                bbox_inches='tight')
                        plt.close()


class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_data(self, seq, position, mask_ratio, random_ratio, *args, **kwargs):
        generator = super().flow((seq, position), *args, **kwargs)
        while True:
            
            # Get augmentend image samples
            ori, position = next(generator)

            # Get masks for each image sample            
            mask = np.ones(ori.shape)
            if random_ratio:
                for i in range(mask.shape[0]):
                    mask_ratio_s = random.uniform(0.03,mask_ratio)
                    mask[i, -int(mask.shape[0] * mask_ratio_s):, :, :] = 0
            else:
                mask[:,-int(mask.shape[1] * mask_ratio):, :, :] = 0

            # Apply masks to all image sample
            masked = deepcopy(ori)
#            masked[mask==0] = np.mean(masked[mask==1]) #value of the maked data is not 1 because the data could be much smaller than the max value of all data (1)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            gc.collect()
            yield [masked, mask, position], ori

def pad_min3(arr):
    # center pads arrays with 0s such that all dims have a min size of 3
    pads = []
    for nd in arr.shape:
        if nd<3:
            pads.append([int((2-nd)//2)+1])
        else:
            pads.append([0])
    return np.pad(arr, pads)

def onebatchpredict_errors(originals, predicted, mask_ratio):
    assert originals.shape == predicted.shape, "Error: the original and predicted images should have the same shape, here: %s and %s" % (str(originals.shape), str(predicted.shape))
    errors = np.zeros((3, originals.shape[0], int(originals.shape[1] * mask_ratio))) # nmse, ssim, ..
    errors5 = np.zeros((3, originals.shape[0], int(np.ceil(int(originals.shape[1] * mask_ratio) / int(originals.shape[1] * 0.05))))) # avg every 5% of prediction
    if len(originals.shape) == 3:
        errors[0,:,:] = np.mean(np.square(originals[:, -int(originals.shape[1] * mask_ratio) :, :] - predicted[:, -int(originals.shape[1] * mask_ratio) :, :]), axis = 2)
        errors[1,:,:] = np.asarray([[ssim(originals[b,-int(originals.shape[1] * mask_ratio) + t,:], predicted[b,-int(originals.shape[1] * mask_ratio) + t,:]) for t in range(int(originals.shape[1] * mask_ratio))] for b in range(originals.shape[0])])
        
        errors5[0,:,:] = np.asarray([[np.mean(np.square(originals[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :] - predicted[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :])) for t in range(int(np.ceil(int(originals.shape[1] * mask_ratio) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
        errors5[1,:,:] = np.asarray([[ssim(pad_min3(originals[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :]), 
                                           pad_min3(predicted[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :]), 
                                           win_size=max(3,min(int((-originals.shape[1]+min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)) + int(originals.shape[1] * mask_ratio) - t * int(originals.shape[1] * 0.05))//2)*2-1,7))
                                           ) for t in range(int(np.ceil(int(originals.shape[1] * mask_ratio) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
    else:
        assert len(originals.shape) == 4, "tensor should be 3D or 4D, here %iD" % len(originals.shape)
        errors[0,:,:] = np.mean(np.square(originals[:, -int(originals.shape[1] * mask_ratio) :, :, :] - predicted[:, -int(originals.shape[1] * mask_ratio) :, :, :]), axis = (2,3))
        errors[1,:,:] = np.asarray([[ssim(originals[b, -int(originals.shape[1] * mask_ratio) + t, :, :], predicted[b, -int(originals.shape[1] * mask_ratio) + t, :, :], multichannel = True) for t in range(int(originals.shape[1] * mask_ratio))] for b in range(originals.shape[0])])
        
        errors5[0,:,:] = np.asarray([[np.mean(np.square(originals[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :, :] - predicted[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :, :])) for t in range(int(np.ceil(int(originals.shape[1] * mask_ratio) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
        errors5[1,:,:] = np.asarray([[ssim(pad_min3(originals[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :, :]), 
                                           pad_min3(predicted[b, -int(originals.shape[1] * mask_ratio) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)), :, :]), 
                                           multichannel = True, 
                                           win_size=max(3,min(int((-originals.shape[1]+min(originals.shape[1], originals.shape[1] - int(originals.shape[1] * mask_ratio) + (t + 1) * int(originals.shape[1] * 0.05)) + int(originals.shape[1] * mask_ratio) - t * int(originals.shape[1] * 0.05))//2)*2-1,7))
                                           ) for t in range(int(np.ceil(int(originals.shape[1] * mask_ratio) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
    return errors, errors5

def onelongbatchpredict_errors(originals, predicted, mask_begins):
    assert originals.shape == predicted.shape, "Error: the original and predicted images should have the same shape, here: %s and %s" % (str(originals.shape), str(predicted.shape))
    errors = np.zeros((3, originals.shape[0], originals.shape[1] - mask_begins)) # nmse, ssim, ..
    errors5 = np.zeros((3, originals.shape[0], int(np.ceil((originals.shape[1] - mask_begins) / int(originals.shape[1] * 0.05))))) # avg every 5% of prediction
    if len(originals.shape) == 3:
        errors[0,:,:] = np.mean(np.square(originals[:, -(originals.shape[1] - mask_begins) :, :] - predicted[:, -(originals.shape[1] - mask_begins) :, :]), axis = 2)
        errors[1,:,:] = np.asarray([[ssim(originals[b,-(originals.shape[1] - mask_begins) + t,:], predicted[b,-(originals.shape[1] - mask_begins) + t,:]) for t in range(originals.shape[1] - mask_begins)] for b in range(originals.shape[0])])
        
        errors5[0,:,:] = np.asarray([[np.mean(np.square(originals[b, -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :] - predicted[b, -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :])) for t in range(int(np.ceil((originals.shape[1] - mask_begins) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
        errors5[1,:,:] = np.asarray([[ssim(pad_min3(originals[b, -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :]), 
                                           pad_min3(predicted[b, -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :]), 
                                           win_size=max(3,min(int((min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05))+(originals.shape[1] - mask_begins) - t * int(originals.shape[1] * 0.05))//2)*2-1,7))
                                           ) for t in range(int(np.ceil((originals.shape[1] - mask_begins) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
    else:
        assert len(originals.shape) == 4, "tensor should be 3D or 4D, here %iD" % len(originals.shape)
        errors[0,:,:] = np.mean(np.square(originals[:, -(originals.shape[1] - mask_begins) :, :, :] - predicted[:, -(originals.shape[1] - mask_begins) :, :, :]), axis = (2,3))
        errors[1,:,:] = np.asarray([[ssim(originals[b, -(originals.shape[1] - mask_begins) + t, :, :], predicted[b, -(originals.shape[1] - mask_begins) + t, :, :], multichannel = True) for t in range(originals.shape[1] - mask_begins)] for b in range(originals.shape[0])])
        
        errors5[0,:,:] = np.asarray([[np.mean(np.square(originals[b, min(-int(originals.shape[1] * 0.05), -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05)) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :, :] - predicted[b, min(-int(originals.shape[1] * 0.05), -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05)) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :, :])) for t in range(int(np.ceil((originals.shape[1] - mask_begins) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
        errors5[1,:,:] = np.asarray([[ssim(pad_min3(originals[b, min(-int(originals.shape[1] * 0.05), -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05)) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :, :]), 
                                           pad_min3(predicted[b, min(-int(originals.shape[1] * 0.05), -(originals.shape[1] - mask_begins) + t * int(originals.shape[1] * 0.05)) : min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05)), :, :]),
                                           multichannel = True, 
                                           win_size=max(3,min(int((min(originals.shape[1], originals.shape[1] - (originals.shape[1] - mask_begins) + (t + 1) * int(originals.shape[1] * 0.05))+(originals.shape[1] - mask_begins) - t * int(originals.shape[1] * 0.05))//2)*2-1,7))
                                           ) for t in range(int(np.ceil((originals.shape[1] - mask_begins) / int(originals.shape[1] * 0.05))))] for b in range(originals.shape[0])])
    return errors, errors5