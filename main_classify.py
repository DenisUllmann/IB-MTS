# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:04:35 2022

@author: Denis
"""

import os
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from models_classify import SP_Conv_Dense

feat_legends = ['intensity',
                'triplet intensity',
                'line center',
                'line width',
                'line asymmetry',
                'total_continium',
                'triplet emission',
                'k/h ratio integrated',
                'kh ratio max',
                'k hight',
                'peak ratios',
                'peak separation']

# SETTINGS
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch = 15 # "Epoch to train [25]")
flags.DEFINE_integer("batch_size = 4 # "The size of batch images [4]")
flags.DEFINE_boolean("batch_norm = True # "True for the model with batch_normalzation")
flags.DEFINE_float("learning_rate_BN = 0.0002 # "Learning rate of for adam with BN (phase 1) [0.0002]")
flags.DEFINE_float("learning_rate_FINE = 0.00005 # "Learning rate of for adam without BN (phase 2 - Fine tuning) [0.00005]")
flags.DEFINE_string("dataset = "iris_level_2C" # "The name of dataset [iris_level_2C, UCSD, mnist]")
flags.DEFINE_string("dataset_address = os.path.join("iris_level_2C") # "The path of dataset")
flags.DEFINE_float("train_ratio = 0.7 # "ratio of dataset to use for training [0.7]")
flags.DEFINE_float("test_ratio = 0.25 # "ratio of dataset to use for testing [0.25]")
flags.DEFINE_boolean("fulldt_nopart = True # "to not partition data and use it fully to train")
flags.DEFINE_integer("label_length = 240 # "The length of spectra. [240]")
flags.DEFINE_float("mask_ratio = 0.25 # "ending ratio of the timesequences to be masked in time / max ratio is random_ratio in True")
flags.DEFINE_boolean("random_ratio = False # "True for random ending ratio of the timesequences to be masked in time (with max value = mask_ratio)")
flags.DEFINE_float("labels = '_'.join(['QS','AR','FL']) # "label for training ['QS','AR','FL']")
flags.DEFINE_string("nolabel = 'nolabel' # "allow to sample from unlabeled data and label it eg. 'nolabel'")
flags.DEFINE_string("classes = '_'.join(['QS']) # "labels of classification ['QS','AR','FL'] OR ['QS','AR-FL']..")
flags.DEFINE_string("noclass = None # "None or name for eventual events not sampling from 'classes' labels (will be assumed to output 0 values for the classifier)")
flags.DEFINE_string("class_inclusions = '_'.join(['']) # "inclusions for classification '_'.join(['QS<AR']) OR [QS<AR, QS<FL] OR ['']")
flags.DEFINE_string("test_labels = '_'.join(['QS','AR','FL']) # "label for testing ['QS','AR','FL']")
flags.DEFINE_string("name = 'Classifier%s_B%i_M%i_R%i_%s--%s--partitionDt%s'%( 
            self.dataset[-2:],  self.batch_size, int(100* self.mask_ratio), 
            int( self.random_ratio),  self.classes,  
            self.class_inclusions.replace('<','-'), not(self.fulldt_nopart)) # "The name of the model")
flags.DEFINE_string("checkpoint_dir = os.path.join(str( self.name),"checkpoint") # "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logs_dir = os.path.join(str( self.name),"log") # "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir = os.path.join(str( self.name),"samples") # "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train1 = True # "True for training phase 1 (with BN) [False]")
flags.DEFINE_boolean("train2 = True # "True for training phase 2 (without BN) : Fine-tuning [False]")
flags.DEFINE_boolean("preload_train = False # "True for loading a pre-trained model before training, False for testing [False]")
flags.DEFINE_boolean("preload_data = False # "True for loading pre-processed data [False]")
flags.DEFINE_boolean("testload_FINE = True # "True for loading a trained model with FINE procedure, False for loading a non FINE model [True]")
flags.DEFINE_boolean("test = False # "True for testing directly at the end of training")
flags.DEFINE_string("test_ds = '_'.join(['TE','TEL']) # "chosen datasets for tests ['TR', 'VA', 'TE', 'TEL']")
flags.DEFINE_boolean("predict = False # "True for predicting number_predict from each chosen dataset predict_ds")
flags.DEFINE_string("predict_ds = '_'.join(['TR', 'VAL', 'TE', 'TEL']) # "chosen datasets for predictions ['TR', 'VAL', 'TE', 'TEL']")
flags.DEFINE_integer("number_predict = 20 # "The maximum number of predictions to do")
flags.DEFINE_boolean("show_res = True # "True for showing results at the end")
flags.DEFINE_boolean("cosmic_to_mean = False # "True for putting cosmic rays to the mean value") # V2
flags.DEFINE_integer("cosmic_t = 2000 # "Threshold in DN/s for cosmic rays [2000]")
flags.DEFINE_string("feat_legends = '__'.join(feat_legends) # "list of the features legends")}

def main(_):
    """
    The main function for training steps
    """
    FLAGS = Settings()
    FLAGS.feat_legends = FLAGS.feat_legends.split('__')
    #FLAGS = update_settings_fromclass(FLAGS, [clas], FLAGS.class_inclusions)
    assert FLAGS.train_ratio + FLAGS.test_ratio <= 1, "The sum of train and test ratios cannot be larger than 1, here: %f and %f"%(FLAGS.train_ratio, FLAGS.test_ratio)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    spectral_classifier = SP_Conv_Dense(FLAGS)

    # spectral_classifier.features_feedback()

    if FLAGS.train1 or FLAGS.train2:        
        # print('show_samples', [e.shape for e in show_samples])
        spectral_classifier.train()
        # spectral_classifier.train()

    if FLAGS.predict:
        spectral_classifier.predicts()

    if FLAGS.test:
        spectral_classifier.tests()

def update_settings_fromclass(FLAGS, classes, class_inclusions):
    FLAGS.classes = classes
    FLAGS.class_inclusions = class_inclusions
    FLAGS.name = 'Classifier%s_B%i_M%i_R%i_%s--%s--partitionDt%s'%( 
        FLAGS.dataset[-2:],  FLAGS.batch_size, int(100* FLAGS.mask_ratio), 
        int( FLAGS.random_ratio),  FLAGS.classes,  
        FLAGS.class_inclusions.replace('<','-'), not(FLAGS.fulldt_nopart)) # "The name of the model")
    ckpt_split = os.path.split(FLAGS.checkpoint_dir)
    FLAGS.checkpoint_dir = os.path.join(*(ckpt_split[:ckpt_split.index('checkpoint')-1]), str( FLAGS.name),"checkpoint") # "Directory name to save the checkpoints [checkpoint]")
    logs_split = os.path.split(FLAGS.logs_dir)
    FLAGS.logs_dir = os.path.join(*(ckpt_split[:logs_split.index('log')-1]), str( FLAGS.name),"log") # "Directory name to save the log [log]")
    sample_split = os.path.split(FLAGS.sample_dir)
    FLAGS.sample_dir = os.path.join(*(ckpt_split[:sample_split.index('samples')-1]), str( FLAGS.name),"samples") # "Directory name to save the image samples [samples]")
    return FLAGS

if __name__ == '__main__':
    tf.app.run()
