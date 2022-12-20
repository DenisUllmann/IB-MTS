# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:57:56 2022

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
plt.rcParams.update({'font.size': 44})
plt.rcParams.update({'font.family': 'Cambria'})
manual_mode = True
ds = 'TE'
label = 'QS'
nclsfier = 2

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
flags.DEFINE_boolean("debug", False, "True to use debug mode (1 epoch and 1st item of generator for test)")
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_integer("batch_size",4, "The size of batch images [4]")
flags.DEFINE_boolean("batch_norm", True, "True for the model with batch_normalzation")
flags.DEFINE_float("learning_rate_BN", 0.0002, "Learning rate of for adam with BN (phase 1) [0.0002]")
flags.DEFINE_float("learning_rate_FINE", 0.00005, "Learning rate of for adam without BN (phase 2 - Fine tuning) [0.00005]")
flags.DEFINE_string("dataset", "iris_level_2C", "The name of dataset [iris_level_2C, UCSD, mnist]")
flags.DEFINE_string("dataset_address", 'C:/Users/Denis/Documents/IRIS/iris_level_2C/', "The path of dataset")
flags.DEFINE_float("train_ratio", 0.7, "ratio of dataset to use for training [0.7]")
flags.DEFINE_float("test_ratio", 0.25, "ratio of dataset to use for testing [0.25]")
flags.DEFINE_integer("label_length", 240, "The length of spectra. [240]")
flags.DEFINE_float("mask_ratio", 0.25, "ending ratio of the timesequences to be masked in time / max ratio is random_ratio in True")
flags.DEFINE_boolean("random_ratio", False, "True for random ending ratio of the timesequences to be masked in time (with max value = mask_ratio)")
flags.DEFINE_string("labels", '_'.join(['QS','AR','PF','FL']), "label for training ['QS','AR','PF','FL']")
flags.DEFINE_string("nolabel", None, "allow to sample from unlabeled data and label it eg. 'nolabel'")
flags.DEFINE_string("test_labels", '_'.join(['QS','AR','PF','FL']), "label for testing ['QS','AR','PF','FL']")
flags.DEFINE_string("name", 'model%s%s_B%i_M%i_R%i_%s-V3'%([flags.FLAGS.model_type,"Mghk"][int(flags.FLAGS.model_type=="PCUNet")], flags.FLAGS.dataset[-2:], flags.FLAGS.batch_size, int(100*flags.FLAGS.mask_ratio), int(flags.FLAGS.random_ratio), flags.FLAGS.labels), "The name of the model")
flags.DEFINE_string("checkpoint_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/checkpoint"%flags.FLAGS.name, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logs_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/log"%flags.FLAGS.name, "Directory name to save the log [log]")
flags.DEFINE_string("results_dir", "C:/Users/Denis/ML/IRIS_predspectra_intermediate_new/%s/results"%flags.FLAGS.name, "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train1", False, "True for training phase 1 (with BN) [False]")
flags.DEFINE_boolean("train2", False, "True for training phase 2 (without BN) : Fine-tuning [False]")
flags.DEFINE_boolean("preload_train", True, "True for loading a pre-trained model before training, False for testing [False]")
flags.DEFINE_boolean("testload_FINE", True, "True for loading a trained model with FINE procedure, False for loading a non FINE model [True]")
flags.DEFINE_boolean("test", True, "True for testing directly at the end of training")
flags.DEFINE_string("test_ds", '_'.join(['TEL']), "chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
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

# assert os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))), "Could not find first set of results"
# print('imported data from {}'.format(self.results_dir))

# results = np.load(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), allow_pickle = True)

split_res = [[
    [fol, 'modelLSTM2C_B4_M25_R0_QS_AR_PF_FL'][
    int(fol=='modelMghk2C_B4_M25_R0_QS_AR_PF_FL-V3')
    ],
    'IRIS_predspectra_intermediate_tf2'][
    int(fol=='IRIS_predspectra_intermediate_new')
    ] for fol in os.path.normpath(self.results_dir).split(os.path.sep)]
self.results_dir = os.path.join(
    *[fol for foll in [[split_res[0], os.path.sep],split_res[1:]] for fol in foll])
assert os.path.isfile(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds))
                      ), "Could not find first set of results"
print('imported data from {}'.format(self.results_dir))

results = np.load(os.path.join(self.results_dir, 'test_RAW_{}.npz'.format(ds)), 
                  allow_pickle = True)

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

i_img = list(makedir.keys())[nclsfier]
lbl = 'TE'
pio_results = pio_centers[i_img][ds]
save_name=os.path.join(
    'testing', makedir[i_img],
    'Data-{}_detailedcentercount.pdf'.format(ds))

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
save_name_samptimedist = save_name[:-2]+[save_name[-2]+'_AvgSample_label_{}_polar-{}'.format(
    label, self.show_dist_polar)]+save_name[-1:]
fname = self.create_onefig_testmeasures(
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

print('save fig at {}'.format(fname))
self.savefig_autodpi(fname,
    bbox_inches=None)
    # bbox_inches='tight')
plt.close()