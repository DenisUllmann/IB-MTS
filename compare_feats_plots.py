# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:29:34 2022

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
from natsort import natsorted
import seaborn as sns

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras_tqdm import TQDMCallback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import NullFormatter
from skimage.measure import compare_ssim as ssim

from libs.pconv_model import PConvUnet
try:
    from libs.lstm_model import LSTM
    from libs.gru_model import GRU
except:
    print("Could not import libraries of LSTM and GRU")

from dataset.data_process import kinter, forplot_assignement_accuracy, kcentroids_equal, to_kcentroid_seq, chunkdata_for_longpredict, retrieve_traintimeseq, create_labelines_timeseq_dataset, convertdata_for_training, no_cosmic, rescale_data_by_seqs
from featuring.brandon_features import feature_transform, Mg_settings
from sklearn.metrics import confusion_matrix#, ConfusionMatrixDisplay

class_parms = None
from featuring.mts_metrics import NPMtsMetrics
from featuring.class_n2_metrics import tss_hss
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
plt.rcParams.update({'font.size': 500})
plt.rcParams.update({'font.family': 'Cambria'})
manual_mode = True
ds = 'TE'
label = 'QS'
nclsfier = 2
fname = 'compare_feats_relerr.pdf'

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

flags = tf.app.flags
flags.DEFINE_string("model_type", "PCUNet", "name of the model to user ['PCUNet'], ['LSTM'], ..")
flags.DEFINE_boolean("with_centerloss", False, "whether to add a term in the total loss optimizing the proximity to the centers")
flags.DEFINE_boolean("debug", False, "True to use debug mode (1 epoch and 1st item of generator for test)")
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_integer("batch_size",4, "The size of batch images [4]")
flags.DEFINE_boolean("batch_norm", True, "True for the model with batch_normalzation")
flags.DEFINE_float("learning_rate_BN", 0.0002, "Learning rate of for adam with BN (phase 1) [0.0002]")
flags.DEFINE_float("learning_rate_FINE", 0.00005, "Learning rate of for adam without BN (phase 2 - Fine tuning) [0.00005]")
flags.DEFINE_string("dataset", "iris_level_2C", "The name of dataset [iris_level_2C, UCSD, mnist]")
flags.DEFINE_string("dataset_address", 'C:/Users/Denis/Documents/IRIS/iris_level_2C/', "The path of dataset")
flags.DEFINE_boolean("given_tvt", True, "Whether the data is already separated in 'train' 'valid' 'test' (these should appear in the file names)")
flags.DEFINE_float("train_ratio", 0.7, "ratio of dataset to use for training [0.7]")
flags.DEFINE_float("test_ratio", 0.25, "ratio of dataset to use for testing [0.25]")
flags.DEFINE_integer("label_length", 240, "The length of spectra. [240]")
flags.DEFINE_float("mask_ratio", 0.25, "ending ratio of the timesequences to be masked in time / max ratio is random_ratio in True")
flags.DEFINE_boolean("random_ratio", False, "True for random ending ratio of the timesequences to be masked in time (with max value = mask_ratio)")
flags.DEFINE_string("labels", '_'.join(['QS','AR','PF','FL']), "label for training ['QS','AR','PF','FL']")
flags.DEFINE_string("nolabel", None, "allow to sample from unlabeled data and label it eg. 'nolabel'")
flags.DEFINE_string("test_labels", '_'.join(['QS','AR','PF','FL']), "label for testing ['QS','AR','PF','FL']")
flags.DEFINE_string("name", 'model%s%s_B%i_M%i_R%i_%s-V3'%([flags.FLAGS.model_type,"Mghk"][int(flags.FLAGS.model_type=="PCUNet")], flags.FLAGS.dataset[-2:], flags.FLAGS.batch_size, int(100*flags.FLAGS.mask_ratio), int(flags.FLAGS.random_ratio), flags.FLAGS.labels), "The name of the model")
flags.DEFINE_string("checkpoint_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/%s/checkpoint"%(flags.FLAGS.dataset,flags.FLAGS.name), "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logs_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/%s/log"%(flags.FLAGS.dataset,flags.FLAGS.name), "Directory name to save the log [log]")
flags.DEFINE_string("results_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/%s/results"%(flags.FLAGS.dataset,flags.FLAGS.name), "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train1", False, "True for training phase 1 (with BN) [False]")
flags.DEFINE_boolean("train2", False, "True for training phase 2 (without BN) : Fine-tuning [False]")
flags.DEFINE_boolean("preload_train", True, "True for loading a pre-trained model before training, False for testing [False]")
flags.DEFINE_boolean("testload_FINE", True, "True for loading a trained model with FINE procedure, False for loading a non FINE model [True]")
flags.DEFINE_boolean("test", True, "True for testing directly at the end of training")
flags.DEFINE_string("test_ds", '_'.join(['TEL']), "chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
flags.DEFINE_boolean("with_features", True, "whether features should be investigated")
flags.DEFINE_boolean("add_classifier", True, "True to add classification stats (it will use the params from main_classify.py).")
flags.DEFINE_string("classes", '_'.join(['QS','AR','PF','FL']), "May be overriden by 'classes_and_inclusions', labels of classification ['QS','AR','PF','FL'] OR ['QS','AR-PF-FL']..")
flags.DEFINE_string("class_inclusions", '_'.join(['']), "inclusions for classification '_'.join(['QS<AR']) OR [QS<AR, QS<PF, QS<FL] OR ['']")
flags.DEFINE_string("noclass", None, "None or name for eventual events not sampling from 'classes' labels (will be assumed to output 0 values for the classifier)")
flags.DEFINE_boolean("add_centercount", True, "True to add centers stats (it will use the params from main_classify.py).")
flags.DEFINE_boolean("predict", False, "True for predicting number_predict from each chosen dataset predict_ds")
flags.DEFINE_string("predict_ds", '_'.join(['TR', 'VAL', 'TE', 'TEL']), "chosen datasets for predictions ['TR', 'VAL', 'TE', 'TEL']")
flags.DEFINE_integer("number_predict", 4, "The maximum number of predictions to do")
flags.DEFINE_boolean("show_res", True, "True for showing results at the end")
flags.DEFINE_boolean("cosmic_to_mean", False, "True for putting cosmic rays to the mean value") # V2
flags.DEFINE_float("cosmic_t", 2000, "Threshold in DN/s for cosmic rays [2000]")
flags.DEFINE_boolean("show_dist_polar", False, "Whether to show distribution in a polar way or not")
flags.DEFINE_string("fig_form", 'pdf', "Format for saved figures in ['png', 'ps', 'pdf', 'svg']")
flags.DEFINE_boolean("backg_color", False, "Whether to colorize backgrounds or not")
flags.DEFINE_boolean("frame_res", False, "To frame marginal results in figures")
# flags.DEFINE_string("feat_legends", '__'.join(feat_legends), "list of the features legends")

FLAGS = flags.FLAGS

self = SP_PCUNet(FLAGS, 
        classes_and_inclusions_addnoclass=classes_and_inclusions_addnoclass, 
        feat_legends=feat_legends)

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
        confusion_classes[clsn] = {}
if self.add_centercount:
    if self.add_classifier:
        pio_centers = {clsn:{} for clsn in list(self.classifier.keys())}
    else:
        pio_centers = {'noclassifier':{}}

test_ds = self.test_ds

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
        # print('clsn', clsn)
        # print('confusion_classes', confusion_classes[clsn])
else:
    makedir = {'noclassifier':'noclassifier'}

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

new_mods = ['modelMghk2C_B4_M25_R0_QS_AR_PF_FL-V3',
            'modelLSTMS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelLSTM2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRUS2C_B4_M25_R0_QS_AR_PF_FL',
            'modelGRU2C_B4_M25_R0_QS_AR_PF_FL',
            'modelNBeats2C_B4_M25_R0_QS_AR_PF_FL']
old_mods = ['modelMghk2C_B4_M25_R0_QS_AR_PF_FL-V3',
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

for new_mod, old_mod, new_dir, old_dir, namecolor_legend, update in zip(
        new_mods, old_mods, new_dirs, old_dirs, namecolor_legends, updates):
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
        psnrs = results['psnrs']
        errors = results['errors']
        errors5 = results['errors5']
        kcenter = results['kcenter'].all()
        for clsn in list(self.classifier.keys()):
            mts_results[clsn][ds] = results['mts_results'].all()[clsn]
        
        if self.add_classifier:
            for clsn in list(self.classifier.keys()):
                class_count_in[clsn][ds] = results['class_count_in'].all()[clsn]
                class_results_in[clsn][ds] = results['class_results_in'].all()[clsn]
                class_count_out[clsn][ds] = results['class_count_out'].all()[clsn]
                class_results_out[clsn][ds] = results['class_results_out'].all()[clsn]
                class_IOchange[clsn][ds] = results['class_IOchange'].all()[clsn]
                class_TIchange[clsn][ds] = results['class_TIchange'].all()[clsn]
                class_TOchange[clsn][ds] = results['class_TOchange'].all()[clsn]
                confusion_classes[clsn] = results['confusion_classes'].all()[clsn]
        if self.add_centercount:
            for clsn in list(self.classifier.keys()):
                pio_centers[clsn][ds] = results['pio_centers'].all()[clsn]
    except:
        last_raw_saving = results['last_raw_saving']
        if 'end' in results.keys():
            end_raw = results['end']
        else:
            end_raw = False
        assert end_raw, "the test savings were not terminated"
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
        else:
            pio_centers['noclassifier'][ds] = [None]
    
    results.close()
    
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
    
    self.ploterr_features(mean_rel_err[ds], self.feat_legends,
                          os.path.join(
                              'testing','Err{}_feats_{}.pdf'.format('REL', ds)),
                          update=update,
                          lc = namecolor_legend,
                          portrait = True, fontsize=66)

fmt_sci2 = lambda value: 'e'.join(["{:.2e}".format(value).split('e')[0], 
                                  str(int("{:.2e}".format(value).split('e')[1]))])
fmt_min_sci1 = lambda value: 'e'.join([
    str(np.floor(10*float(fmt_sci2(value).split('e')[0]))/10),
    fmt_sci2(value).split('e')[1]])
fmt_max_sci1 = lambda value: 'e'.join([
    str(np.ceil(10*float(fmt_sci2(value).split('e')[0]))/10),
    fmt_sci2(value).split('e')[1]])
fmt_mid_sci1 = lambda value: 'e'.join([
    str(int(10*float(fmt_sci2(value).split('e')[0]))/10),
    fmt_sci2(value).split('e')[1]])
fmt_trynotsci = lambda value: ['.'.join(
    [str(float(value)).split('.')[0], 
     str(int(str(float(value)).split('.')[1][::-1]))[::-1]]), 
    str(int(float(value)))][
        int(int(str(float(value)).split('.')[1])==0)] if int(
            "{:.1e}".format(float(value)).split('e')[1])>-5 else value
fmt_min = lambda value: fmt_min_sci1(
    value) if len(fmt_trynotsci(fmt_min_sci1(value)))>6 else fmt_trynotsci(
        fmt_min_sci1(value))
fmt_max = lambda value: fmt_max_sci1(
    value) if len(fmt_trynotsci(fmt_max_sci1(value)))>6 else fmt_trynotsci(
        fmt_max_sci1(value))
fmt_mid = lambda value: fmt_max_sci1(
    value) if len(fmt_trynotsci(fmt_mid_sci1(value)))>6 else fmt_trynotsci(
        fmt_mid_sci1(value))
for axi in plt.gcf().axes:
    if len(axi.lines)>0:
        ymin, ymax = axi.get_ylim()
        yminst = fmt_min(ymin)
        ymaxst = fmt_max(ymax)
        ymin = float(yminst)
        ymax = float(ymaxst)
        axi.set_ylim(ymin = ymin - 0.02*(ymax-ymin), 
                     ymax=ymax + 0.02*(ymax-ymin))
        axi.set_yticks([None], labels=[None], minor=False)
        #axi.set_yticks([None], labels=[None], minor=True)
        axi.set_yticks([ymin, (ymax+ymin)/2, ymax], 
                       labels=[yminst, fmt_mid((ymax+ymin)/2), ymaxst],
                       minor=False)
        axi.grid(visible=True, which='major', axis='y')

self.savefig_autodpi(fname,
    bbox_inches=None)
    # bbox_inches='tight')
plt.close()