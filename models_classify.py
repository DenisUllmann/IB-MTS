# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:15:06 2022

@author: Denis
"""

import os
import gc
import datetime
import numpy as np
import random
import itertools
import zipfile
import io

from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from tqdm.keras import TqdmCallback
# from keras_tqdm import TQDMCallback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO libs instead of nn_libs (also in nn_libs)
from libs.class_pconv_model import PConvDense

from dataset.data_process import *

now = datetime.datetime.now
plt.rcParams.update({'font.size': 33})

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

class SP_Conv_Dense(object):
    def __init__(self, config, skip_data=False, c_dim=1):
        self.name = config.name
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.c_dim = c_dim
        self.batch_norm = config.batch_norm
        self.learning_rate_BN = config.learning_rate_BN
        self.learning_rate_FINE = config.learning_rate_FINE
        self.dataset = config.dataset
        self.dataset_address = config.dataset_address
        self.train_ratio = config.train_ratio
        self.test_ratio = config.test_ratio
        self.fulldt_nopart = config.fulldt_nopart
        self.label_length = config.label_length
        self.mask_ratio = config.mask_ratio
        self.random_ratio = config.random_ratio
        self.labels = config.labels.split('_')
        self.nolabel = config.nolabel
        self.classes = config.classes.split('_')
        self.class_inclusions = config.class_inclusions.split('_')
        self.nclass = len(self.classes)
        if self.nclass == 1:
            print("[Classifier: %s] - Override param 'noclass' newly set to value 'noclass' because the Classifier has just one class: it will do basic anomaly detection"%self.name)
            self.noclass = 'noclass'
        else:
            self.noclass = config.noclass
        # dict for classification {train_label: class vector[0 true,1 false,..]}
        # class vector has length nclass
        # true when train_label is in corresponding class from classes (class_inclusions included)
        self.class_dict = {k: [int(k in c or c+'>'+k in self.class_inclusions) for c in self.classes] for k in self.labels}
        # if self.nolabel is not None:
        #     self.class_dict = {
        #         **self.class_dict,
        #         self.nolabel: [0]*len(self.classes)}
        print('class dict', self.class_dict)
        # DONE here: change SP_Conv_dense model
        # DONE here: change loss with the class_dict
        # DONE here and not here: stats of clusters per classes / per prediction: comparison
        self.test_labels = config.test_labels
        self.test_labels = self.test_labels.split('_')
        self.name = config.name
        self.checkpoint_dir = config.checkpoint_dir
        self.logs_dir = config.logs_dir
        self.sample_dir = config.sample_dir
        self.train1 = config.train1
        self.train2 = config.train2
        self.preload_train = config.preload_train
        self.preload_data = config.preload_data
        self.testload_FINE = config.testload_FINE
        self.test = config.test
        self.test_ds = config.test_ds
        self.test_ds = self.test_ds.split('_')
        if 'TE' in self.test_ds:
            self.test_ds = self.test_ds + ['TE_%s'%ds for ds in self.test_labels]
        if 'TEL' in self.test_ds:
            self.test_ds = self.test_ds + ['TEL_%s'%ds for ds in self.test_labels]
        self.predict = config.predict
        self.predict_ds = config.predict_ds
        self.number_predict = config.number_predict
        self.show_res = config.show_res
        self.coef_diff = 5
        self.feat_legends = config.feat_legends
        # labels = self.labels.split('_')
        # test_labels = self.test_labels.split('_')
        self.all_labels = list(set(self.labels+self.test_labels))
        
        if not(skip_data):
            print('[%s - START] Creating the datasets..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
            if self.dataset == 'iris_level_2C' or self.dataset == 'iris_level_2B':
                self.c_dim = 1
            if self.train1 and not(self.preload_train) and not(self.preload_data):
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
                # print("data pack", len(data_pack))
                # print("data pack", len(data_pack[0]))
                # print("data pack", len(data_pack[0][1]))
                # print("data pack", [ee for ee in data_pack[0][1]])
                # print("data pack", any([('QS' in str(ee)) for ee in data_pack[0][1]]))
                self.data_pack = {a: [e for e in self.data_pack if any([a in str(ee) for ee in e[1]])] for a in activities} # {act: [(data, pos)]}
                # print('dtp_ini_data', type(self.data_pack['QS'][0][0]))
                # partition each activities by Train / Test
                # print("data pack", {u: len(v) for u,v in data_pack.items()})
                # print("shape data", data_pack['QS'][0][0].shape)
                # partition by train / test
                # {act_lab: {event: percentage}}
                if self.fulldt_nopart:
                    print('Taking all the available data for training, very few overlapping data for val and test')
                    for a in self.data_pack:
                        self.data_pack[a] = {
                            'TR': self.data_pack[a],
                            'VAL': list(zip(*list(zip(*self.data_pack[a]))[:max(2,min(int(len(self.data_pack)*0.1),5))])),
                            'TE': list(zip(*list(zip(*self.data_pack[a]))[:max(2,min(int(len(self.data_pack)*0.1),5))]))}# {act: {tvt: [(data, pos)]}}
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
                            # print(percentages[a])
                            # print(events[0])
                            # print(events[-1])
                            # print(percentages[a][events[0]])
                            # print(percentages[a][events[-1]])
                            # print(percentages[a][events[0]]<=self.train_ratio)
                            # print(percentages[a][events[-1]]<=self.test_ratio)
                            cumul_left = self.loop_cumul(percentages[a], events, self.train_ratio, attempts==max_attempts-1)
                            cumul_right = self.loop_cumul(percentages[a], events, self.test_ratio, attempts==max_attempts-1, start='right')
                            if cumul_left is None or cumul_right is None:
                                # print('should skip')
                                continue
                            # print('cumul_left', cumul_left)
                            # print('cumul_right', cumul_right)
                            # print('train ratio', self.train_ratio)
                            # print('test ratio', self.test_ratio)
                            # print('events', events)
                            # print('percentages[a]', percentages[a])
                            # print('cumul train', [sum(percentages[a][ee] for ee in events[:events.index(e)+1])<self.train_ratio for e in events])
                            # print('cumul train', [sum(percentages[a][ee] for ee in events[:events.index(e)+1]) for e in events])
                            # print('cumul test', [sum(percentages[a][ee] for ee in events[::-1][events[::-1].index(e):])<self.train_ratio for e in events[::-1]])
                            # print('cumul test', [sum(percentages[a][ee] for ee in events[::-1][events[::-1].index(e):]) for e in events[::-1]])
                            # print('verify cumul', [a+b for a,b in zip([sum(percentages[a][ee] for ee in events[:events.index(e)+1]) for e in events],[sum(percentages[a][ee] for ee in events[::-1][:events[::-1].index(e)+1]) for e in events[::-1]])])
                            # print(cumul_right+cumul_left)
                            # print(len(events))
                            # print(cumul_right+cumul_left<len(events))
                            # print('cumul_left', cumul_left)
                            # print('substract' ,(cumul_right+cumul_left)-len(events))
                            if cumul_right+cumul_left>len(events):
                                cumul_left -= (cumul_right+cumul_left)-len(events)
                            assert cumul_left>0, "problem with the partition, train and test overlap, coding issue or too small amount of events"
                            # print('cumul_left', cumul_left)
                            # print('cumul_right', cumul_right)
                            train_events = events[:cumul_left]
                            test_events = events[::-1][:cumul_right]
                            if cumul_left+cumul_right<len(events):
                                valid_events = [e for e in events if e not in train_events+test_events]
                                assert len(valid_events)!=0, "problem during the partition, see code"
                                # print('train events: ', train_events)
                                # print('validation events: ', valid_events)
                                # print('test events: ', test_events)
                            # when there are no remaining events for validation
                            else:
                                print("Label %s : No separate event for validation dataset, will use an overlapping part of the training data"%a)
                                valid_ratio = 1-self.train_ratio-self.test_ratio
                                # print('valid_ratio', valid_ratio)
                                valid_events = [e for e in train_events if percentages[a][e]<valid_ratio]
                                # print('valid_events', valid_events)
                                if len(valid_events)!=0:
                                    random.shuffle(valid_events)
                                    cumul_left = self.loop_cumul(percentages[a], valid_events, valid_ratio, False)
                                    valid_events = valid_events[:cumul_left]
                                else:
                                    valid_events = [train_events[[percentages[a][e] for e in train_events].index(min([percentages[a][e] for e in train_events]))]]
                                # print(valid_events)
                            self.data_pack[a] = {
                                'TR': [e for e in self.data_pack[a] if any(ee in e[1] for ee in train_events)],
                                'VAL': [e for e in self.data_pack[a] if any(ee in e[1] for ee in valid_events)],
                                'TE': [e for e in self.data_pack[a] if any(ee in e[1] for ee in test_events)]}# {act: {tvt: [(data, pos)]}}
                            shuffle = False
                        
                        # print('train_events', train_events)
                        # print('test_events', test_events)
                        # print('valid_events', valid_events)
                        assert all(len(lev)>0 for lev in [train_events, test_events, valid_events]) and len([e for e in train_events if e in test_events])==0, "Did not find how to partition data"
                        del cumul_right, cumul_left, train_events, test_events, valid_events
                self.data_pack = {dt: {a: self.data_pack[a][dt] for a in self.data_pack} for dt in ['TR','VAL','TE']}# {tvt: {act: [(data, pos), ...]}}
                self.data_pack['TE'] = {
                    **(self.data_pack['TE']),
                    **{a: list(zip(*create_labelines_timeseq_dataset(self.dataset_address, [a], self.label_length))) for a in self.test_labels if a not in self.labels}}
                # print('dtpt', type(data_pack_train[0]))
                # print('dtpt', len(data_pack_train[0]))
                # print('dtpt0', type(data_pack_train[0][0]))
                # print('dtpt1', type(data_pack_train[0][1]))
                if not(self.fulldt_nopart):
                    self.data_pack = {
                        **{'_'.join([dt,a]): self.data_pack[dt][a] for dt in self.data_pack for a in self.data_pack[dt]},
                        'TR': list(itertools.chain(*(self.data_pack['TR'].values()))),
                        'VAL': list(itertools.chain(*(self.data_pack['VAL'].values()))),
                        'TE': list(itertools.chain(*(self.data_pack['TE'].values()))),}# [(data, pos)..]
                else:
                    self.data_pack = {
                        'TR': list(itertools.chain(*(self.data_pack['TR'].values()))),
                        'VAL': list(itertools.chain(*(self.data_pack['VAL'].values())))}#,
                        # 'TE': list(itertools.chain(*(self.data_pack['TE'].values()))),}# [(data, pos)..]
                # separate data and positions {key_of_data: [data, positions]}
                # print('data_pack', {k: type(self.data_pack[k]) for k in self.data_pack})
                # print('data_pack', {k: len(self.data_pack[k]) for k in self.data_pack})
                # print('data_pack', {k: type(self.data_pack[k][0]) for k in self.data_pack})
                # print('data_pack', {k: len(self.data_pack[k][0]) for k in self.data_pack})
                self.data_pack = {k: list(zip(*(self.data_pack[k]))) for k in self.data_pack} # {k: [(data,),(position,)]}
                # print('data_pack', {k: [len(self.data_pack[k][0]), len(self.data_pack[k][1])] for k in self.data_pack})
                # print('data_pack', {k: [self.data_pack[k][0][0].shape, len(self.data_pack[k][1][0])] for k in self.data_pack})
                # print('k', list(self.data_pack.keys())[0])
                # print('dtp', len(self.data_pack[list(self.data_pack.keys())[0]]))
                # print('dtp0', type(self.data_pack[list(self.data_pack.keys())[0]][0]))
                # print('dtp0', len(self.data_pack[list(self.data_pack.keys())[0]][0]))
                # print('dtp0', self.data_pack[list(self.data_pack.keys())[0]][0])
                # print('dtp00', len(self.data_pack[list(self.data_pack.keys())[0]][0][0].shape))
                # convert for training {key_of_data: [data, positions]}
                # random.shuffle(data_pack)
            else:
                print("Loading previous dataset..")
                data_info = np.load(os.path.join(self.checkpoint_dir, 'data_longformat.npz'), allow_pickle = True)
                keys = [k.replace('data_', '') for k in list(data_info.keys()) if 'data_' in k]
                self.data_pack = {k: (data_info['data_'+k], data_info['position_'+k]) for k in keys}
            
            # for k in self.data_pack:
            #     print('k', k)
            #     print('dtpk', len(self.data_pack[k]))
            self.data_train = {k: convertdata_for_training(list(self.data_pack[k][0]), list(self.data_pack[k][1]), self.label_length, self.mask_ratio) for k in self.data_pack}
            self.data_train = {
                **self.data_train,
                'show': [list(e) for e in zip(*list(zip(*self.data_train['TR']))[:self.batch_size])]}
            # if self.c_dim == 1:
            #     # convert to RGB data
            #     self.data_train = {k: [np.tile(self.data_train[k][0], [1, 1, 1, 3]), self.data_train[k][1]] for k in self.data_train}
    
            # print('[%s - END] Datasets created'%now().strftime('%d.%m.%Y - %H:%M:%S'))
            # print('data_pack', {k: [self.data_pack[k][0][0].shape, len(self.data_pack[k][1][0])] for k in self.data_pack})
            # print('data_train', {k: [type(self.data_train[k][0]), type(self.data_train[k][1])] for k in self.data_train})
            # print('data_train', {k: [len(self.data_train[k][0]), len(self.data_train[k][1])] for k in self.data_train})
            # print('data_train', {k: [self.data_train[k][0].shape, len(self.data_train[k][1])] for k in self.data_train})
            
            # Create training generator
            # self.train_generator = {
            #     'TR': AugmentingDataGenerator().flow_from_data(
            #         np.array(self.data_train['TR'][0]),
            #         np.array(list(zip(*(self.data_train['TR'][1])))[0]),
            #         self.mask_ratio, 
            #         self.random_ratio,
            #         batch_size=self.batch_size
            # )}
            # self.train_length = {'TR': len(self.data_train['TR'][0])}
            
            # # Create validation generator
            # self.val_generator = {
            #     'VAL': AugmentingDataGenerator().flow_from_data(
            #         np.array(self.data_train['VAL'][0]), 
            #         np.array(list(zip(*(self.data_train['VAL'][1])))[0]),
            #         self.mask_ratio,
            #         self.random_ratio,
            #         batch_size=self.batch_size
            # )}
            # self.val_length = {'VAL': len(self.data_train['VAL'][0])}
            
            # Create testing generators (as dicts)
            # {data_label : [generator, length]}
            # generators: [masked, mask], ori, position
            # [ori, mask], label(['FL', 'AR']), position([ints])
            # print('pos', list(zip(*(self.data_train['TR'][1])))[0])
            # print('lab', [self.label_from_pos(p) for p in list(
            #     zip(*(self.data_train['TR'][1])))[2]])
            self.generators = {
                k: [AugmentingDataGenerator(dtype='float32').flow_from_data(
                    np.array(self.data_train[k][0]),
                    # np.array(['_'.join([str(p),l]) for p,l in zip(
                    #     list(zip(*(self.data_train[k][1])))[0],
                    #     [self.label_from_pos(p) for p in list(
                    #         zip(*(self.data_train[k][1])))[2]])]),
                    np.array(list(zip(*(self.data_train[k][1])))[0]),
                    np.array([self.class_dict[
                        self.label_from_pos(p, predictonly=False)] for p in list(
                            zip(*(self.data_train[k][1])))[2]]),
                    self.mask_ratio,
                    self.random_ratio,
                    batch_size=self.batch_size
                    ), # 1st is data for model
                    len(self.data_train[k][0]), # 2nd is length of data
                    {p:l for p,l in zip(
                        list(zip(*(self.data_train[k][1])))[0],
                        [self.label_from_pos(p, predictonly=False) for p in list(
                            zip(*(self.data_train[k][1])))[2]])} # 3rd is dict{pos:lab} info
                    ] for k in self.data_train}
            
            if not(self.fulldt_nopart):
                saveCompressed(open(os.path.join(self.checkpoint_dir, 'data_longformat.npz'), 'wb'),
                         **{'data_'+u: v[0] for u,v in self.data_pack.items()},
                         **{'position_'+u: v[1] for u,v in self.data_pack.items()},
                         allow_pickle=True)
    
    def catlabels_from_pos(self, pos, predictonly=True):
        # inputs and outputs are np.array
        # pos = np.1darray[str]
        # output = np.2darray[cat labels defined in self.class_dict]
        return np.stack(np.vectorize(
            lambda x: self.class_dict[self.label_from_pos(x, predictonly)],
            otypes=[np.ndarray])(pos))
    
    def label_from_pos(self, pos, predictonly=True):
        ### pos is str file name
        try:
            # print('pos', pos)
            # print('self.all_labels', self.all_labels)
            return self.all_labels[[l in pos for l in self.all_labels].index(True)]
        except:
            assert predictonly, "No Label can only be used for predictions, not train/eval or test, for pos: {}".format(pos)
            return self.nolabel
    
    def catlabels_from_idxpos(self, pos, dict_poslab=None, predictonly=True):
        # inputs and outputs are np.array
        # pos = np.1darray[ints]
        # output = np.2darray[cat labels defined in self.class_dict]
        # print('class_dict', self.class_dict)
        return np.stack(np.vectorize(
            lambda x: self.class_dict[self.label_from_idxpos(x, dict_poslab, predictonly)],
            otypes=[np.ndarray])(pos))
    
    def label_from_idxpos(self, pos, dict_poslab=None, predictonly=True):
        ### pos is int index
        # print('pos', pos)
        # print('dict_poslab', dict_poslab)
        # print('predictonly', predictonly)
        # print('return label', self.dict_poslab[pos])
        # print('nolabel', self.nolabel)
        try:
            return dict_poslab[pos]
        except:
            assert predictonly, "No Label can only be used for predictions, not train/eval or test, for pos: {}".format(pos)
            return self.nolabel
    
    def to_rgb(self, data):
        if self.c_dim == 1:
            # convert to RGB data
            return np.tile(data, [1, 1, 1, 3])
        return data
    
    def cumul_side(self, percent_dict, sorted_list, threshold):
        return [sum(percent_dict[ee] for ee in sorted_list[:sorted_list.index(e)+1])<threshold for e in sorted_list]
    
    def loop_cumul(self, percent_dict, sorted_list, threshold, max_attempt, start='left'):
        # print('sorted_list', sorted_list)
        # print('per_d', percent_dict[sorted_list[0]])
        # print('thr', threshold)
        if start=='right':
            sorted_list = sorted_list[::-1]
        if percent_dict[sorted_list[0]]<=threshold:
            # print(sum(self.cumul_side(percent_dict, sorted_list, threshold))+1)
            return sum(self.cumul_side(percent_dict, sorted_list, threshold))+1
        elif max_attempt:
            return 1
    
    def model_instance(self, train_bn, inference_only):
        
        # Instantiate the model
        self.model = PConvDense(
            img_rows=self.label_length, img_cols=self.label_length, 
            c_dim=self.c_dim, inference_only=inference_only,
            classes=self.classes, class_dict=self.class_dict, 
            noclass=self.noclass, net_name=self.name, dtype='float32')
        
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'trained_model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        if train_bn:
            learning_rate = self.learning_rate_BN
        else:
            learning_rate = self.learning_rate_FINE
        if inference_only:
            if self.testload_FINE:
                ckpt_name = max(glob(os.path.join(checkpoint_dir,'BN'+str(self.learning_rate_BN)+'FINE'+str(self.learning_rate_FINE)+'*')), key = os.path.getctime)
            else:
                ckpt_name = max(glob(os.path.join(checkpoint_dir,'BN'+str(self.learning_rate_BN)+'w*')), key = os.path.getctime)
            if ckpt_name:
                print(" [*] Checkpoint found")
                print(ckpt_name)
                self.model.load(ckpt_name, train_bn=False)
                print(" [*] Model loaded")
            else:
                assert False, " [*] Failed to find a checkpoint, train a model first"
        elif self.preload_train or not(train_bn): # training from pretrained
            ckpt_name = max(glob(os.path.join(checkpoint_dir,'BN'+str(self.learning_rate_BN)+'*')), key = os.path.getctime)
            if ckpt_name:
                print(" [*] Checkpoint found")
                print(ckpt_name)
                self.model.load(ckpt_name, train_bn=train_bn, lr=learning_rate)
                print(" [*] Model loaded")
            else:
                assert False, " [*] Failed to find a checkpoint, train a model first"
        else:
            #setting model
            self.set_model(train_bn=train_bn, lr=learning_rate)
            print(" [*] Model is set")
    
    def plot_pred(self, orii, maski, labeli, pred_classi, clai, 
                  typ='Img', name='Epoch', meta='ID'):
        # tf.print('plot pred')
        # plt.imshow(tf.transpose(tf.squeeze(tf.boolean_mask(
        #     orii, maski, axis=0))).numpy(), 
        #     cmap = 'gist_heat', vmin = 0, vmax = 1)
        # print('orii', orii.shape)
        # print('maski', maski.shape)
        # print('orimi', (orii.squeeze()[maski.astype(bool).squeeze()[:,0]].T).shape)
        # print('res_plot',{self.classes[i]: pred_classi[i] for i in range(len(self.classes))})
        # print('res_plot',{self.classes[i]: np.squeeze(pred_classi[i]) for i in range(len(self.classes))})
        tsz = plt.rcParams['figure.titlesize']
        plt.rc('figure', titlesize='xx-small')
        plt.imshow(orii.squeeze()[maski.astype(bool).squeeze()[:,0]].T, 
            cmap = 'gist_heat', vmin = 0, vmax = 1)
        plt.title('Label %s\nPrediction: %s\n'%(labeli, clai) + '\n'.join(
            ['%s: %.3f'%(self.classes[i], pred_classi[i]) for i in range(len(self.classes))]))
        plt.tight_layout()
        plt.rc('figure', titlesize=tsz)
        
        plt.savefig(os.path.join(self.sample_dir,'{}__{}__{}.png'.format(typ, name, meta)))
        plt.close()
        
    
    def parse_lab_pos(self, pos, i, dict_poslab):
        """Outputs pos as int and vlabel as str
        pos is the last output of the generator of the type 
        tf.Tensor(['pos_lab])
        """
        # tf.print('parse lab pos')
        # labi = tf.strings.split(tf.expand_dims(pos[i],0), '_').values[1]
        posi = int(pos[i])
        labi = dict_poslab[posi]
        return labi, posi
    
    def plot_callback(self, epoch):
        """Called at the end of each epoch, displaying the previous test images,
        as well as their masked predictions and saving them to disk"""
        
        # Gettings samples to show results at the end of each epochs
        show_samples = next(self.generators['show'][0])
        # Parse samples
        (ori, mask, pos), label = show_samples
        
        # tf.print('ori', ori.shape)
        # tf.print('mask', mask.shape)
        # print('ori', ori.shape)
        # print('mask', mask.shape)
        
        # Get samples & Display them        
        pred_class = self.model.predict([ori, mask], 
                                        batch_size=self.batch_size,
                                        steps=1)
        # tf.print('pred_class', pred_class)
        # print('pred_class np', np.asarray(pred_class))
        # print('pred_class np', np.array(pred_class))
        # print('pred_class np', pred_class.numpy())
        cla = self.model.np_assign_class(pred_class)
        # tf.print('class', cla)
        # print('class', cla)
    
        # Clear current output and display test images
        for i in range(len(ori)):
            # parse lab and pos data
            labi, posi = self.parse_lab_pos(pos, i, self.generators['show'][2])
            # tf.print('labi', labi)
            # tf.print('posi', posi)
            self.plot_pred(ori[i], mask[i], labi,
                           pred_class[i], cla[i],
                           'Img_{}'.format(i),
                           'Epoch_{}'.format(epoch),
                           'ID_{}'.format(posi))
    
    def set_model(self, train_bn=True, lr=0.0002):
    
        # Create Classifier PConv+Dense model
        self.model.model = self.model.build_pconv_dense(train_bn)
        print(self.model.model.summary())
        self.model.compile_pconv_dense(self.model.model, lr) 
    
    # def train(self):
    #     show_samples = next(self.generators['TR'][0])
    def train(self):
        if self.train1:
            self.train_phase(True)
        if self.train2:
            self.train_phase(False)
    
    def set_tqdmcb(self):
        tqdm_cb = TqdmCallback()
        tqdm_cb.on_train_batch_begin = tqdm_cb.on_batch_begin
        tqdm_cb.on_train_batch_end = tqdm_cb.on_batch_end
        tqdm_cb.on_test_begin = lambda y: None
        tqdm_cb.on_test_end = lambda y: None
        tqdm_cb.on_test_batch_begin = lambda x, y: None
        tqdm_cb.on_test_batch_end = lambda x, y: None
        return tqdm_cb
    
    def train_phase(self, train_bn):
        self.model_instance(train_bn, False)
        if train_bn:
            checkpoint_phase = 'BN'+str(self.learning_rate_BN)
        else:
            checkpoint_phase = 'BN'+str(self.learning_rate_BN)+'FINE'+str(self.learning_rate_FINE)
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'trained_model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('[%s - START] Training Phase %i ..'%(now().strftime('%d.%m.%Y - %H:%M:%S'), 2-train_bn))
        # Run training for certain amount of epochs
        self.model.fit_generator(
            self.generators['TR'][0], 
            steps_per_epoch=int((self.generators['TR'][1]//self.batch_size)*1),
            validation_data=self.generators['VAL'][0],
            validation_steps=self.generators['VAL'][1]//self.batch_size,
            epochs=self.epoch,  
            verbose=0,
            callbacks=[
                TensorBoard(
                    log_dir=self.logs_dir,
                    write_graph=False
                ),
                ModelCheckpoint(
                    os.path.join(checkpoint_dir,checkpoint_phase+'weights.{epoch:02d}-{loss:.2f}.h5'),
                    monitor='val_loss', 
                    save_best_only=True, 
                    save_weights_only=True
                ),
                LambdaCallback(
                    on_epoch_end=lambda epoch, logs: self.plot_callback(epoch)
                    # on_epoch_end=lambda epoch, logs: self.plot_callback(self.to_rgb(show_samples), epoch)
                ),
                self.set_tqdmcb()
            ]
        )
        print('[%s - END] Training.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def predicts(self):
        self.model_instance(False, True)
        print('[%s - START] Predicting..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
        predict_ds = self.predict_ds.split('_')
        for ds in predict_ds:
            if 'TEL' in ds:
                self.long_predicts(ds)
            else:
                gen = self.generators[ds][0]
                print('[Predicting] %s Samples'%ds)
            
            n = 0
            for (ori, mask, pos), lab in tqdm(gen, total = self.number_predict):
                
                # masked = self.to_rgb(masked)
                # mask = self.to_rgb(mask)
                # ori = self.to_rgb(ori)
                
                n = self.predict_save_1batch(ori, mask, lab, pos, ds, n)
                
                # Only create predictions for about 100 images
                if n > self.number_predict:
                    break
        print('[%s - END] Predicting.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def predict_save_1batch(self, ori, mask, lab, pos, ds, n):
        # Run predictions for this batch of images
        # prediction sclars for each class
        pred_class = self.model.predict([ori, mask])
        # name of the most probable class
        cla = self.model.assign_class(pred_class)
        for i in range(len(ori)):
            labi, posi = self.parse_lab_pos(pos, i, self.generators[ds][2])
            self.plot_pred(ori[i], mask[i], labi,
                           pred_class[i], cla[i],
                           'Img_{}'.format(i),
                           'PredDS_{}'.format(ds),
                           'ID_{}'.format(posi))
            
            n += 1
            
        return n
    
    def long_predicts(self, ds):
        self.model_instance(False, True)
        # predict_ds = self.predict_ds.split('_')
        print('ds', ds)
        assert 'TEL' in ds, "error in the label os the data to predict"
        # ds = ds.replace('TEL','TE')
        print('[Predicting] Long %s Samples'%ds)
        
        n = 0
        # for seq, pos in tqdm(zip(self.data_test, self.positions_test), total = self.number_predict):
        for seq, pos in tqdm(zip(self.data_pack[ds.replace('TEL','TE')][0], self.data_pack[ds.replace('TEL','TE')][1]), total = self.number_predict):
            mask_seq = np.ones_like(seq)
            mask_seq[:self.label_length-int(self.label_length * self.mask_ratio)] = 0
            pred_class = tf.Variable([0]*len(self.classes), name='long_class_prediction')
            data_chunked, pos_chunked = chunkdata_for_longpredict(seq, pos, self.label_length, self.mask_ratio)
#                if self.c_dim == 1:
#                    data_chunked = np.tile(data_chunked, [1, 1, 1, 3])
            # patch_nb = data_chunked.shape[0]
            idx = 0
            # stride = int(seq.shape[1]*(self.mask_ratio))
            # last = ((seq.shape[0]-self.label_length) % stride) # 0 if no strange patch at the end of the sequence, last resting timevalues otherwise
                
            for data_patch, pos_patch in zip(data_chunked, pos_chunked):
                
                data_patch = np.expand_dims(data_patch, axis = 0).astype(np.float32)
                assert idx == pos_patch[0], "ERROR in parsing image chunks"
                mask = np.ones(data_patch.shape, dtype=np.float32)
                mask[:, :-int(mask.shape[1] * self.mask_ratio), :, :] = 0
                
                pred_class.assign_add(self.model.predict([data_patch, mask]))
                
                idx += 1
            
            # If bugs, find index in pos where name of file from the format data_pack
            lab = self.label_from_pos(
                [pp[['.' in p for p in pp].index(True)] for pp in pos], 
                predictonly=True)
            pred_class = pred_class.assign(pred_class/idx)
            cla = self.model.assign_class(pred_class)
            self.plot_pred(seq, mask_seq, lab,
                           pred_class[0], cla[0],
                           'Class',
                           'LongPredDS_{}'.format(ds),
                           'ID_{}'.format(pos[0]))
            
            n += 1
            # Only create predictions for about self.number_predict images
            if n > self.number_predict - 1:
                break
    
    def tests(self):
        self.model_instance(False, True)
        print('[%s - START] Testing..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
        if os.path.isfile(os.path.join(self.checkpoint_dir, 'results.npz')):
            results = np.load(os.path.join(self.checkpoint_dir, 'results.npz'), allow_pickle = True)
            count = results['count'].all()
            results = results['results'].all()
        else:
            results = {}
            count = {}
        
        test_ds = self.test_ds
        for ds in test_ds:
            if 'TEL' in ds:
                test_ds.remove(ds)
                test_ds += [ds]
        for ds in test_ds:
            if 'TEL' not in ds:
                gen = self.generators[ds][0]
                length = self.generators[ds][1]
                print('[Testing] %s Samples:'%ds)
            else:
                np.savez(os.path.join(self.checkpoint_dir, 'results'),
                         results = results,
                         count = count)
                
                self.long_test()
                
                results = np.load(os.path.join(self.checkpoint_dir , 'results.npz'), allow_pickle = True)
                count = results['count'].all()
                results = results['results'].all()
                break
            
            print('# of tests :', length)
            for metric in self.model.metrics:
                metric.reset_states()
            # Loop through ds
            n = 0
            for (ori, mask, pos), lab in tqdm(gen):
                
                n += ori.shape[0]
                
                _ = self.model.model.test_on_batch([ori, mask, pos], lab)
                
                if n > length:
                    break
            
            results[ds] = {name:metric.result() for name, metric in zip(self.model.model.metrics_names[1:],self.model.model.metrics)}
            count[ds] = {name:metric.count() for name, metric in zip(self.model.model.metrics_names[1:],self.model.model.metrics)}
            for name in results:
                print('{} : {}'.format(name, results[ds][name]))
        
        np.savez(os.path.join(self.checkpoint_dir, 'results'),
                 results = results,
                 count = count)
        
        print('[%s - END] Testing.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
    
    def long_test(self):
        self.model_instance(False, True)
        # test_ds = self.predict_ds.split('_')
        test_ds = self.test_ds
        if os.path.isfile(os.path.join(self.checkpoint_dir, 'results.npz')):
            results = np.load(os.path.join(self.checkpoint_dir, 'results.npz'), allow_pickle = True)
            count = results['count'].all()
            results = results['results'].all()
        else:
            results = {}
            count = {}
        print('test_ds', test_ds)
        for ds in test_ds:
            if 'TEL' in test_ds:
                print('ds', ds)
                print('in data_pack', list(self.data_pack.keys()))
                assert ds.replace('TEL', 'TE') in self.data_pack, "ERROR: problem with the label of data to test, not present in the available data"
                print('[Predicting] Long Testing Samples')
                print('# of tests', len(self.data_pack[ds.replace('TEL', 'TE')][0]))
                print('[%s - START] Testing..'%now().strftime('%d.%m.%Y - %H:%M:%S'))
                
                n = 0
                
                for metric in self.model.model.metrics:
                    metric.reset_states()
                
                for seq, pos in tqdm(zip(self.data_pack[ds.replace('TEL', 'TE')][0], self.data_pack[ds.replace('TEL', 'TE')][1])):
                    mask_seq = np.ones_like(seq)
                    mask_seq[:self.label_length-int(self.label_length * self.mask_ratio)] = 0
                    pred_class = tf.Variable([0]*len(self.classes), name='long_class_prediction')
                    data_chunked, pos_chunked = chunkdata_for_longpredict(seq, pos, self.label_length, self.mask_ratio)
        #                if self.c_dim == 1:
        #                    data_chunked = np.tile(data_chunked, [1, 1, 1, 3])
                    # patch_nb = data_chunked.shape[0]
                    idx = 0
                    # stride = int(seq.shape[1]*(self.mask_ratio))
                    # last = ((seq.shape[0]-self.label_length) % stride) # 0 if no strange patch at the end of the sequence, last resting timevalues otherwise
                        
                    for data_patch, pos_patch in zip(data_chunked, pos_chunked):
                        
                        data_patch = np.expand_dims(data_patch, axis = 0).astype(np.float32)
                        assert idx == pos_patch[0], "ERROR in parsing image chunks"
                        mask = np.ones(data_patch.shape, dtype=np.float32)
                        mask[:, :-int(mask.shape[1] * self.mask_ratio), :, :] = 0
                        
                        pred_class.assign_add(self.model.predict([data_patch, mask]))
                        
                        idx += 1
                    
                    # If bugs, find index in pos where name of file from the format data_pack
                    catlab = self.catlabels_from_pos(
                        [pp[['.' in p for p in pp].index(True)] for pp in pos],
                        predictonly=False)
                    pred_class = pred_class.assign(pred_class/idx)
                    # cla = self.model.assign_class(pred_class)
                    for metric in self.model.model.metrics:
                        metric.update_state(catlab, pred_class)
                    
                    n += 1
                    
                    # if n > self.test_length:
                    if n > len(self.data_pack[ds.replace('TEL', 'TE')][0]):
                        break
                
                results[ds] = {name:metric.result() for name, metric in zip(self.model.model.metrics_names[1:],self.model.model.metrics)}
                count[ds] = {name:metric.count() for name, metric in zip(self.model.model.metrics_names[1:],self.model.model.metrics)}
                for name in results:
                    print('{} : {}'.format(name, results[ds][name]))
                
                print('[%s - END] Testing.'%now().strftime('%d.%m.%Y - %H:%M:%S'))
                np.savez(os.path.join(self.checkpoint_dir, 'results'),
                         results = results,
                         count = count)


class AugmentingDataGenerator(ImageDataGenerator):
    # To define flow_from_data that returns a generator of batched outputs:
    # [ori, mask], label, position
    # ori and mask: [batch,time,lamda,channel] (inputs for models)
    # label: categorical classes tf.Tensor[b, nclass]
    # position: meta string tf.Tensor[b] where values are 'positionIndex_originalLabel'
    def __init__(self, *args, **kwargs):
        super(ImageDataGenerator, self).__init__(*args, **kwargs)
    
    def flow_from_data(self, seq, position, label, mask_ratio, random_ratio,
                       *args, **kwargs):
        seed = np.random.randint(0,100)
        generator_lb = super().flow(seq, label, seed=seed, *args, **kwargs)
        # print(type(position))
        # print(position.dtype)
        # print(position.shape)
        # print(position)
        # print(np.expand_dims(position, axis=[1,2,3]))
        generator_id = super().flow(np.expand_dims(position, axis=[1,2,3]), 
                                    seed=seed, *args, **kwargs)
        while True:
            
            # Get augmentend image samples
            ori, label = next(generator_lb)
            
            # print('try position')
            position = next(generator_id).squeeze()#.astype('str')
            # print('res', position)
            # tf.print('tf_res', position)
            # position = tf.strings.join([position, label], separator='_')
            
            # catmap = CategoricalLookup(class_dict)
            # label = catmap.lookup(tf.convert_to_tensor(label))
            
            # Get masks for each image sample 
            # for prediction, 0 for positions to predict, 
            # for classification (here), 1 for positions to predict
            try:
                dtype = self.dtype.name
            except:
                dtype = self.dtype
            mask = create_class_mask(ori, mask_ratio, random_ratio, dtype=dtype)
            
            # Yield ([ori, mask],  label) training batches
            # print(masked.shape, ori.shape)
            # print('label', label)
            # tf.print('tf_label', label)
            gc.collect()
            yield [ori, mask, position], label

def create_class_mask(ori, mask_ratio, random_ratio, dtype='float32'):
    mask = np.ones(ori.shape, dtype=dtype)
    if random_ratio:
        for i in range(mask.shape[0]):
            mask_ratio_s = random.uniform(0.03,mask_ratio)
            mask[i, :-int(mask.shape[0] * mask_ratio_s), :, :] = 0
    else:
        mask[:,:-int(mask.shape[1] * mask_ratio), :, :] = 0
    return mask
