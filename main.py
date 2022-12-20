# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:24:05 2019

@author: CUI
"""

import os
import tensorflow as tf
import numpy as np

from models import SP_PCUNet

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
        self.dataset = "pb_2C"# The name of dataset [iris_level_2C, UCSD, mnist]")
        self.dataset_address = 'C:/Users/Denis/ML/MTS_Data/PEMS-BAY/data_2c'# The path of dataset")
        self.given_tvt = True# Whether the data is already separated in 'train' 'valid' 'test' (these should appear in the file names)
        self.train_ratio = 0.7# ratio of dataset to use for training [0.7]")
        self.test_ratio = 0.25# ratio of dataset to use for testing [0.25]")
        self.label_length = 325# The length of spectra. [240 (for Mghk), 137 (for al), 370 (for ld)))]")
        if self.model_type == "NBeats":
            self.n_blocks = int(65714097/50400)
        self.mask_ratio = 0.25# ending ratio of the timesequences to be masked in time / max ratio is random_ratio in True")
        self.random_ratio = False# True for random ending ratio of the timesequences to be masked in time (with max value = mask_ratio)")
        self.labels = '_'.join(['PB'])# label for training ['QS','AR','PF','FL'] (these should appear in the filename[:2])")
        self.nolabel = None# allow to sample from unlabeled data and label it eg. 'nolabel'")
        self.test_labels = '_'.join(['PB'])# label for testing ['QS','AR','PF','FL']")
        self.name = 'model%s%s_B%i_M%i_R%i_%s'%([self.model_type+"%s"%['','star'][int(self.with_centerloss)],"Mghk%s"%['','star'][int(self.with_centerloss)]][int(self.model_type=="PCUNet")], self.dataset[-2:], self.batch_size, int(100*self.mask_ratio), int(self.random_ratio), self.labels)# The name of the model")
        self.checkpoint_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/checkpoint"%(self.dataset,self.name)# Directory name to save the checkpoints [checkpoint]")
        self.logs_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/log"%(self.dataset,self.name)# Directory name to save the log [log]")
        self.results_dir = "C:/Users/Denis/ML/IRIS_predspectra_intermediate_tf2/%s/%s/results"%(self.dataset,self.name)# Directory name to save the image samples [samples]")
        self.train1 = True# True for training phase 1 (with BN) [False]")
        self.train2 = False# True for training phase 2 (without BN) : Fine-tuning [False]")
        self.preload_train = False# True for loading a pre-trained model before training, False for testing [False]")
        self.testload_FINE = False# True for loading a trained model with FINE procedure, False for loading a non FINE model [True]")
        self.test = True# True for testing directly at the end of training")
        self.test_ds = '_'.join(['TE','TEL'])# chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
        self.with_features = False# whether features should be investigated
        self.add_classifier = False# True to add classification stats (it will use the params from main_classify.py).")
        self.classes = '_'.join(['PB'])# May be overriden by 'classes_and_inclusions', labels of classification ['QS','AR','PF','FL'] OR ['QS','AR-PF-FL']..")
        self.class_inclusions = '_'.join([''])# inclusions for classification '_'.join(['QS<AR']) OR [QS<AR, QS<PF, QS<FL] OR ['']")
        self.noclass = None# None or name for eventual events not sampling from 'classes' labels (will be assumed to output 0 values for the classifier)")
        self.add_centercount = False# True to add centers stats (it will use the params from main_classify.py).")
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
# FLAGS.feat_legends = FLAGS.feat_legends.split('__')

def main():
    """
    The main function for training steps
    """
    
    assert FLAGS.train_ratio + FLAGS.test_ratio <= 1, "The sum of train and test ratios cannot be larger than 1, here: %f and %f"%(FLAGS.train_ratio, FLAGS.test_ratio)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    
    spectral_predictor = SP_PCUNet(FLAGS, 
        classes_and_inclusions_addnoclass=classes_and_inclusions_addnoclass, 
        feat_legends=feat_legends)
    
    if output_npz_features:
        spectral_predictor.features_feedback()
    
    if FLAGS.train1 or FLAGS.train2:        
        # Gettings samples to show results at the end of each epochs
        show_samples = next(spectral_predictor.generators['show'][0])
        print('show_samples', tuple(tuple(si.shape for si in s) for s in show_samples))
        print('show_samples', [len(e) for e in show_samples])
        spectral_predictor.train(show_samples)
    
    if FLAGS.predict:
        spectral_predictor.predicts()
    
    if FLAGS.test:
        spectral_predictor.tests()

if __name__ == '__main__':
    main()