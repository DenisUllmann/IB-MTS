# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:24:05 2019

@author: CUI
"""

import os
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
from absl import logging

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
# for IRIS data of labeled data
classes_and_inclusions_addnoclass = list(zip(
    ['_'.join(clss) for clss in [
        ['QS','AR','PF','FL']]],
    ['_'.join(incs) for incs in [
        ['']]],
    [
      None]))

FLAGS = flags.FLAGS

flags.DEFINE_boolean("manual_mode", False, "Use manual mode is you don't want to load dataand disable part of the code in models.py")
flags.DEFINE_boolean("change_traindata", True, "whether to enable to save/overwrite data_longformat.npz")
flags.DEFINE_string("model_type", "IBMTS", "name of the model to user ['IBMTS'], ['LSTM'], ['LSTMS'], ['GRU'], ['GRUS'], ['NBeats']")
flags.DEFINE_boolean("debug", False, "True to use debug mode (1 epoch and 1st item of generator for test)")
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [4]")
flags.DEFINE_float("learning_rate_BN", 0.0002, "Learning rate of for adam with BN (phase 1) [0.0002]")
flags.DEFINE_string("dataset", "iris", "The name of dataset [iris, al, pb]")
flags.DEFINE_string("root_address", os.path.dirname(os.path.realpath(__file__)), "The path for the root folder of the project")
flags.DEFINE_string("dataset_address", os.path.join(FLAGS.root_address,'iris_data'), "The path of dataset")
flags.DEFINE_boolean("given_tvt", False, "Whether the data is already separated in 'train' 'valid' 'test' (these should appear in the file names)")
flags.DEFINE_float("train_ratio", 0.7, "ratio of dataset to use for training [0.7]")
flags.DEFINE_float("test_ratio", 0.25, "ratio of dataset to use for testing [0.25]")
flags.DEFINE_integer("label_length", 325, "The length of spectra. [240 (for Mghk), 137 (for al), 325 (for pb)))]")
if FLAGS.model_type == "NBeats":
    flags.DEFINE_integer("n_blocks", 2,"number of blocks for NBeats")
flags.DEFINE_string("labels", '_'.join(['QS','AR','FL']), "label for training ['QS','AR','FL'] for IRIS, ['AL'] for AL (these should appear in the filename[:2])")
flags.DEFINE_string("test_labels", '_'.join(['QS','AR','FL']), "label for testing ['QS','AR','FL']")
flags.DEFINE_string("name", 'model%s_M%i_%s'%(FLAGS.model_type+"%s"%['','star'][int(FLAGS.with_centerloss)], int(100*FLAGS.mask_ratio), FLAGS.labels), "The name of the model")
flags.DEFINE_string("checkpoint_dir", os.path.join(FLAGS.root_address,FLAGS.dataset,FLAGS.name,"checkpoint"), "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logs_dir",  os.path.join(FLAGS.root_address,FLAGS.dataset,FLAGS.name,"log"), "Directory name to save the log [log]")
flags.DEFINE_string("results_dir",  os.path.join(FLAGS.root_address,FLAGS.dataset,FLAGS.name,"results"), "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training phase 1 (with BN) [False]")
flags.DEFINE_boolean("preload_train", False, "True for loading a pre-trained model before training, False for testing [False]")
flags.DEFINE_boolean("test", True, "True for testing directly at the end of training")
flags.DEFINE_string("test_ds", '_'.join(['TE','TEL']), "chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
flags.DEFINE_boolean("with_features", True, "whether features should be investigated")
flags.DEFINE_boolean("add_classifier", False, "True to add classification stats (it will use the params from main_classify.py).")
flags.DEFINE_string("classes", '_'.join(['PB']), "May be overriden by 'classes_and_inclusions', labels of classification ['QS','AR','PF','FL'] OR ['QS','AR-PF-FL']..")
flags.DEFINE_string("class_inclusions", '_'.join(['']), "inclusions for classification '_'.join(['QS<AR']) OR [QS<AR, QS<PF, QS<FL] OR ['']")
flags.DEFINE_boolean("add_centercount", False, "True to add centers stats (it will use the params from main_classify.py).")
flags.DEFINE_boolean("predict", False, "True for predicting number_predict from each chosen dataset predict_ds")
flags.DEFINE_string("predict_ds", '_'.join(['TR', 'VAL', 'TE', 'TEL']), "chosen datasets for predictions ['TR', 'VAL', 'TE', 'TEL']")
flags.DEFINE_integer("number_predict", 4, "The maximum number of predictions to do")
flags.DEFINE_boolean("show_res", True, "True for showing results at the end")
flags.DEFINE_boolean("show_dist_polar", False, "Whether to show distribution in a polar way or not")
flags.DEFINE_string("fig_form", 'pdf', "Format for saved figures in ['png', 'ps', 'pdf', 'svg']")
flags.DEFINE_boolean("backg_color", False, "Whether to colorize backgrounds or not")
flags.DEFINE_boolean("frame_res", False, "To frame marginal results in figures")

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
        feat_legends=feat_legends, manual_mode=FLAGS.manual_mode, 
        change_traindata=FLAGS.change_traindata)
    
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
  app.run(main)
