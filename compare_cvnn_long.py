# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:03:04 2022

@author: Denis
"""
from models import SP_PCUNet

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
from libs.gru_model import GRU
	   
													   

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
previous_font_size = plt.rcParams['font.size']
plt.rcParams.update({'font.size': 66})
plt.rcParams.update({'font.family': 'Cambria'})
manual_mode = True
ds = 'TEL'
label = 'QS'
nclsfier = 2
fname = 'compare_cvnn_long.pdf'
out_img = True

# name of the features studied and type of the label ticks for the graphs
feat_legends = [('intensity','%.1f'),
                ('triplet intensity','%.2f'),
                ('line center','%.1f'),
                ('line width','int'),
                ('line asymmetry','%.2f'),
                ('total_continium','int'),
                ('triplet emission','int'),
                ('k/h ratio integrated','%.2f'),
                ('kh ratio max','%.2f'),
                ('k hight','%.2f'),
                ('peak ratios','int'),
                ('peak separation','int')]

# SETTINGS

# To output and npz file with the physical features onthe data
output_npz_features = False

# To eventualy define only one classifier with settings 
# 'classes' and 'class_inclusions'
#classes_and_inclusions = None

# To define several classifiers list[(classes, inclusions, noclass), ..]
# For model trained with all labels
classes_and_inclusions_addnoclass = list(zip(
    ['_'.join(clss) for clss in [
        ['QS','AR','PF','FL'],
        ['QS','AR','PF','FL'],
        ['QS','AR','PF','FL'],
        ['QS','AR','PF','FL'],
        ['QS'],
        ['AR'],
        ['PF'],
        ['FL']]],
    ['_'.join(incs) for incs in [
        ['QS<AR'],
        ['QS<AR'],
        [''],
        [''],
        [''],
        [''],
        [''],
        ['']]],
    [
      None,
      'noclass',
      None,
      'noclass',
      'noclass',
      'noclass',
      'noclass',
      'noclass']))
# For model trained just with labels not in the inclusions above
# classes_and_inclusions_addnoclass = list(zip(
#     ['_'.join(clss) for clss in [
#         ['QS','AR','PF','FL'],
#         ['QS','AR','PF','FL'],
#         ['QS'],
#         ['AR'],
#         ['PF'],
#         ['FL']]],
#     ['_'.join(incs) for incs in [
#         [''],
#         [''],
#         [''],
#         [''],
#         [''],
#         ['']]],
#     [None,
#      'noclass',
#      'noclass',
#      'noclass',
#      'noclass',
#      'noclass']))

class Flags(object):
    def __init__(self):
        self.model_type = "PCUNet"# name of the model to user ['PCUNet'], ['LSTM'], ['GRU'], ..")
        self.with_centerloss = False# whether to add a term in the total loss optimizing the proximity to the centers
        self.debug = True# True to use debug mode (1 epoch and 1st item of generator for test)")
        self.epoch = 100# Epoch to train [25]")
        self.batch_size = 4# The size of batch images [4]")
        self.batch_norm = True# True for the model with batch_normalzation")
        self.learning_rate_BN = 0.0002# Learning rate of for adam with BN (phase 1) [0.0002]")
        self.learning_rate_FINE = 0.00005# Learning rate of for adam without BN (phase 2 - Fine tuning) [0.00005]")
        self.dataset = "iris_level_2C"# The name of dataset [iris_level_2C, UCSD, mnist]")
        self.dataset_address = 'C:/Users/Denis/ML/MTS_Data/LD2011_2014/data_2c'# The path of dataset")
        self.given_tvt = True# Whether the data is already separated in 'train' 'valid' 'test' (these should appear in the file names)
        self.train_ratio = 0.7# ratio of dataset to use for training [0.7]")
        self.test_ratio = 0.25# ratio of dataset to use for testing [0.25]")
        self.label_length = 240# The length of spectra. [240 (for Mghk), 137 (for al), 370 (for ld)))]")
        self.mask_ratio = 0.25# ending ratio of the timesequences to be masked in time / max ratio is random_ratio in True")
        self.random_ratio = False# True for random ending ratio of the timesequences to be masked in time (with max value = mask_ratio)")
        self.labels = '_'.join(['QS','AR', 'PF', 'FL'])# label for training ['QS','AR','PF','FL'] (these should appear in the filename[:2])")
        self.nolabel = None# allow to sample from unlabeled data and label it eg. 'nolabel'")
        self.test_labels = '_'.join(['QS','AR', 'PF', 'FL'])# label for testing ['QS','AR','PF','FL']")
        self.name = 'model%s%s_B%i_M%i_R%i_%s'%([self.model_type+"%s"%['','star'][int(self.with_centerloss)],"Mghk%s"%['','star'][int(self.with_centerloss)]][int(self.model_type=="PCUNet")], self.dataset[-2:], self.batch_size, int(100*self.mask_ratio), int(self.random_ratio), self.labels)# The name of the model")
        self.checkpoint_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/checkpoint"%(self.dataset,self.name)# Directory name to save the checkpoints [checkpoint]")
        self.logs_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/log"%(self.dataset,self.name)# Directory name to save the log [log]")
        self.results_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/results"%(self.dataset,self.name)# Directory name to save the image samples [samples]")
        self.train1 = True# True for training phase 1 (with BN) [False]")
        self.train2 = True# True for training phase 2 (without BN) : Fine-tuning [False]")
        self.preload_train = True# True for loading a pre-trained model before training, False for testing [False]")
        self.testload_FINE = True# True for loading a trained model with FINE procedure, False for loading a non FINE model [True]")
        self.test = True# True for testing directly at the end of training")
        self.test_ds = '_'.join(['TE','TEL'])# chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
        self.with_features = True# whether features should be investigated
        self.add_classifier = True# True to add classification stats (it will use the params from main_classify.py).")
        self.classes = '_'.join(['QS','AR','PF','FL'])# May be overriden by 'classes_and_inclusions', labels of classification ['QS','AR','PF','FL'] OR ['QS','AR-PF-FL']..")
        self.class_inclusions = '_'.join([''])# inclusions for classification '_'.join(['QS<AR']) OR [QS<AR, QS<PF, QS<FL] OR ['']")
        self.noclass = None# None or name for eventual events not sampling from 'classes' labels (will be assumed to output 0 values for the classifier)")
        self.add_centercount = True# True to add centers stats (it will use the params from main_classify.py).")
        self.predict = False# True for predicting number_predict from each chosen dataset predict_ds")
        self.predict_ds = '_'.join(['TR', 'VAL', 'TE', 'TEL'])# chosen datasets for predictions ['TR', 'VAL', 'TE', 'TEL']")
        self.number_predict = 4# The maximum number of predictions to do")
        self.show_res = True# True for showing results at the end")
        self.cosmic_to_mean = False# True for putting cosmic rays to the mean value") # V2
        self.cosmic_t = 2000# Threshold in DN/s for cosmic rays [2000]")
        self.show_dist_polar = False# Whether to show distribution in a polar way or not")
        self.fig_form = 'pdf'# Format for saved figures in ['png', 'ps', 'pdf', 'svg']")
        self.backg_color = False# Whether to colorize backgrounds or not")
        self.frame_res = False# To frame marginal results in figures
        # self.feat_legends = '__'.join(feat_legends)# list of the features legends")

FLAGS = Flags()

def plot_longcv(means, means_ssim, means_kcenter, stride,
                update=False, lc=None, axes=None, last=False):
    if update:
        axes = {}
        i_ax = 0
        for row in range(4):
            for col in range(2):
                axes[row, col] = plt.gcf().axes[i_ax]
                i_ax += 1
        ax_leg0 = plt.gcf().axes[i_ax]
        ax_leg1 = plt.gcf().axes[i_ax+1]
        ax_leg2 = plt.gcf().axes[i_ax+2]
    else:
        widths, heights = [[2, 2], [.2, 1, .2, 1, 1, 1, .2]]
        fig = plt.figure(constrained_layout=True, figsize=(10*sum(widths), 
                                                           10*sum(heights)))
        # spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)
        spec = gridspec.GridSpec(nrows=len(heights), ncols=len(widths), 
                                 figure=fig, 
                                 width_ratios=widths, 
                                 height_ratios=heights)
        axes = {}
        for row in range(4):
            for col in range(2):
                axes[row, col] = fig.add_subplot(spec[[1,row+2][int(row>0)], col])
        ax_leg0 = fig.add_subplot(spec[0, :])
        ax_leg0.axis('off')
        ax_leg1 = fig.add_subplot(spec[2, :])
        ax_leg1.axis('off')
        ax_leg2 = fig.add_subplot(spec[-1, :])
        ax_leg2.axis('off')
    
    if lc == None:
        nml = 'model?'
        clrl = 'b'
        alpha = 1
    else:
        try:
            nml, (clrl, alpha) = lc
        except:
            nml, clrl = lc
            alpha = 1
    vmin = 0
    vstart = stride
    vstep = stride
    vend = len(means[ds][1:])*stride+1
    # vmin, vstart, vstep, vend = self.adjust_xcoord(
    #     toshow=means[ds][1:], 
    #     tofit=stride*means[ds][1:]+1)
    axes[0, 0].plot(np.arange(vstart, vend, vstep), 
                    means[ds][1:], color = clrl, alpha=alpha, label=nml)
    self.format_axis(axes[0, 0], vmin=20, vmax=40, step = 10, axis = 'y', 
                     type_labels='int')
    self.format_axis(axes[0, 0], vmin=vmin, vmax=len(means[ds][1:])*stride, 
                     step = 100, axis = 'x', ax_label='time', 
                     type_labels='int', maxm1=False)
    axes[0, 0].set_ylabel('PSNR')
    axes[0, 0].set_title('temporal evaluation')
    # self.set_description(axes[0, 0], legend_loc='upper center', 
    #                      title='temporal evaluation', fontsize='x-small')
    vmin = 0
    vstart = stride
    vstep = stride
    vend = len(means_ssim[ds][1:])*stride+1
    # vmin, vstart, vstep, vend = self.adjust_xcoord(
    #     toshow=means_ssim[ds][1:], 
    #     tofit=stride*means_ssim[ds][1:]+1)
    axes[0, 1].plot(np.arange(vstart, vend, vstep), 
                    means_ssim[ds][1:], color = clrl, label=nml)
    if last:
        axes[0, 1].plot(np.arange(vstart, vend, vstep), 
                        np.ones_like(means_ssim[ds][1:]),
                        label='ideal', linestyle=(0, (5, 10)), color='tab:cyan', alpha=alpha)
    self.format_axis(axes[0, 1], vmin=0.5, vmax=1, step = 0.1, axis = 'y', 
                     type_labels='%.1f', margin=[0,1])
    self.format_axis(axes[0, 1], vmin=vmin, vmax=len(means_ssim[ds][1:])*stride, 
                     step = 100, axis = 'x', ax_label='time', type_labels='int',
                     maxm1=False)
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title('temporal evaluation')
    # self.set_description(axes[0, 1], legend_loc='upper center', 
    #                      title='temporal evaluation', fontsize='x-small')
    ax_leg0.legend(*axes[0, 1].get_legend_handles_labels(), loc="center", 
                   fontsize='small', ncol=4, frameon=False)
    for i in range(1,7):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2
        vmin = 0
        vstart = stride
        vstep = stride
        vend = len(means_kcenter[ds][i][1:])*stride+1
        # vmin, vstart, vstep, vend = self.adjust_xcoord(
        #     toshow=means_kcenter[ds][i][1:], 
        #     tofit=stride*means_kcenter[ds][i][1:]+1)
        axes[row, col].plot(np.arange(vstart, vend, vstep), 
                            means_kcenter[ds][i][1:], 
                            color = clrl, label=nml, alpha=alpha)
        if last:
            axes[row, col].plot(np.arange(vstart, vend, vstep), 
                                [kinter(i) for _ in range(
                                    len(means_kcenter[ds][i][1:]))], 
                                label='Rand',
                                linestyle=(0, (5, 10)), color='tab:brown')
            axes[row, col].plot(np.arange(vstart, vend, vstep), 
                                np.ones_like(means_kcenter[ds][i][1:]), 
                                label='Ideal', 
                                linestyle=(0, (5, 10)), color='tab:cyan')

        self.format_axis(axes[row, col], vmin=0, vmax=1, step = 0.2, 
                         axis = 'y', ax_label='accuracy', type_labels='%.1f',
                         margin=[0,1])
        self.format_axis(axes[row, col], vmin=0,
                         vmax=len(means_kcenter[ds][i][1:])*stride, step = 100, 
                         axis = 'x', ax_label='time', type_labels='int',
                         maxm1=False)
        axes[row, col].set_ylabel('%i-NN Accuracy'%i)
        axes[row, col].set_title('temporal evaluation')
    ax_leg1.legend(*axes[row, col].get_legend_handles_labels(), loc="center", 
                   fontsize='small', ncol=4, frameon=False)
    ax_leg2.legend(*axes[row, col].get_legend_handles_labels(), loc="center", 
                   fontsize='small', ncol=4, frameon=False)
    # spec.tight_layout(fig)
    return axes


self = SP_PCUNet(FLAGS, 
        classes_and_inclusions_addnoclass=classes_and_inclusions_addnoclass, 
        feat_legends=feat_legends)

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
if self.add_centercount:
    if self.add_classifier:
        pio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
        globpio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
    else:
        pio_centers = {'noclassifier':{}}
        globpio_centers = {'noclassifier':{}}

test_ds = self.test_ds

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
            [mm.name in m().name for mm in clsfier.model.model.metrics])]
        np_metrics_in[clsn] = [mm(**a) for mm, a in zip(
            np_metrics_in[clsn], 
            [[{}, {'class_assign_fn':clsfier.model.np_assign_class}][int(
                tf.keras.metrics.CategoricalAccuracy().name == m().name)] for m in np_metrics_in[clsn]])]
        np_metrics_out[clsn] = [m for m in np_all_metrics if any(
            [mm.name in m().name for mm in clsfier.model.model.metrics])]
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

new_mods = ['modelMghk2C_B4_M25_R0_QS_AR_PF_FL-V3',
            'modelLSTMS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelLSTM2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRUS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRU2C_B4_M25_R0_QS_AR_PF_FL',
            'modelNBeats2C_B4_M25_R0_QS_AR_PF_FL']
old_mods = ['modelMghk2C_B4_M25_R0_QS_AR_PF_FL',
            'modelMghk2C_B4_M25_R0_QS_AR_PF_FL-V3',
            'modelLSTMS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelLSTM2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRUS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRU2C_B4_M25_R0_QS_AR_PF_FL']
new_dirs = ['IRIS_predspectra_intermediate_new',
            'IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_tf2',
            'HPC/Yggdrasil/IRIS_predspectra_intermediate_tf2']
old_dirs = ['IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_new',
            'IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_tf2',
            'IRIS_predspectra_intermediate_tf2',
            'ML/IRIS_predspectra_intermediate_tf2']
namecolor_legends = [('ib-mts',('b',1)),
                     ('lstm',('c',.5)),
                     ('ib-lstm',('g',.5)),
                     ('gru',('m',.5)),
                     ('ib-gru',('r',.5)),
                     ('nbeats',('y',.5))]
updates = [False] + [True]*(len(new_mods)-1)
n_mod = len(new_mods)
axes = None
res = {}
for i_mod, (new_mod, old_mod, new_dir, old_dir, namecolor_legend, update) in enumerate(zip(
        new_mods, old_mods, new_dirs, old_dirs, namecolor_legends, updates)):
    self.results_dir = os.path.normpath(
            self.results_dir).replace(
                os.path.normpath(old_dir), 
                os.path.normpath(new_dir)).replace(
                    os.path.normpath(old_mod), 
                    os.path.normpath(new_mod))
    assert os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))), "Could not find first set of results {}".format(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)))
    print('imported data from {}'.format(self.results_dir))
    
    results = np.load(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), allow_pickle = True)
    try:
        mean_abs_err[ds] = results['mean_abs_err']
        mean_rel_err[ds] = results['mean_rel_err']
        abs_length[ds] = results['abs_length']
        rel_length[ds] = results['rel_length']
        psnrs_df = results['psnrs_df'].all()['df']
        ssims_df = results['ssims_df'].all()['df']
        kcenters_df = results['kcenters_df'].all()
        for clsn in list(self.classifier.keys()):
            mts_results[clsn][ds] = results['mts_results'].all()[clsn]
            glob_mts_results[clsn][ds] = results['glob_mts_results'].all()[clsn]
        
        if self.add_classifier:
            for clsn in list(self.classifier.keys()):
                class_count_in[clsn][ds] = results['class_count_in'].all()[clsn]
                class_results_in[clsn][ds] = results['class_results_in'].all()[clsn]
                class_count_out[clsn][ds] = results['class_count_out'].all()[clsn]
                class_results_out[clsn][ds] = results['class_results_out'].all()[clsn]
                class_acc[clsn][ds] = results['class_acc'].all()[clsn]
                class_IOchange[clsn][ds] = results['class_IOchange'].all()[clsn]
                class_TIchange[clsn][ds] = results['class_TIchange'].all()[clsn]
                class_TOchange[clsn][ds] = results['class_TOchange'].all()[clsn]
                confusion_classes[clsn] = results['confusion_classes'].all()[clsn]
        if self.add_centercount:
            for clsn in list(self.classifier.keys()):
                pio_centers[clsn][ds] = results['pio_centers'].all()[clsn]
                globpio_centers[clsn][ds] = results['globpio_centers'].all()[clsn]
    except:
        if 'end' in results.keys():
            end_raw = results['end']
        else:
            end_raw = False
        assert end_raw, "the test savings were not terminated"
        last_raw_saving = results['last_raw_saving']
        print("loading previous raw results: last saved index is {}".format(last_raw_saving))
        mean_abs_err[ds] = results['mean_abs_err']
        mean_rel_err[ds] = results['mean_rel_err']
        abs_length[ds] = results['abs_length']
        rel_length[ds] = results['rel_length']
        psnrs_df = results['psnrs_df'].all()['df']
        ssims_df = results['ssims_df'].all()['df']
        kcenters_df = results['kcenters_df'].all()
        
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
        else:
            pio_centers['noclassifier'][ds] = [None]
            globpio_centers['noclassifier'][ds] = [None]
    
    results.close()
    
    i_img = list(makedir.keys())[nclsfier]
    
    mts = glob_mts_results[i_img][ds]
    meta=None
    glob=None
    glob_meta=None
    stride = int(self.label_length*(self.mask_ratio))
    
    w_box = 15 # width for one box of result
    h_box = 15 # height for one box of result
    test_case = False
    print("Model: {}".format(new_mod))
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
    
    if out_img:
        axes = plot_longcv(means, means_ssim, means_kcenter, stride,
                update=update, lc=namecolor_legend, axes = axes, 
                last=i_mod==n_mod-1)
    res[new_mod] = {'PSNR':[means[ds][0], np.mean(means[ds][1:])],
                    'SSIM':[means_ssim[ds][0], np.mean(means_ssim[ds][1:])],
                    **{nnn:[means_kcenter[ds][innn+1]] for innn,nnn in enumerate(['{}-NN'.format(k) for k in range(1,7)])}}

if out_img:
    self.savefig_autodpi(fname,
        bbox_inches=None)
        # bbox_inches='tight')
np.savez('psnr_ssim_long', res = res)
plt.close()

plt.rcParams.update({'font.size': int(previous_font_size)})