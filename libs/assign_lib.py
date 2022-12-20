# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:27:01 2022

@author: Denis
"""
import tensorflow as tf

class BatchBool(tf.keras.layers.Layer):
  def __init__(self):
    super(BatchBool, self).__init__()
  
  def call(self, inputs):
    # inputs are [y_true, mask]
    return tf.boolean_mask(inputs[:,:,:,0], tf.cast(1-inputs[:,:,:,1], 'bool'))

class BatchAssignOneh(tf.keras.layers.Layer):
  def __init__(self, assign_center, centers):
    super(BatchAssignOneh, self).__init__()
    self.batch_oneh_proba_fn = lambda y: tf.map_fn(
        lambda x: tf.map_fn(
            lambda xx: tf.one_hot(assign_center(centers,xx),len(centers)),
            x),
        y
    )
  
  def call(self, inputs):
    # inputs are [y_true, mask]
    y_true, mask = inputs
    mask = tf.cast(mask, y_true.dtype)
    masktime = tf.cond(tf.rank(mask)==4, 
                   lambda: mask[:,:,:,0], 
                   lambda:mask)
    masktime = tf.cond(tf.rank(masktime)==3, 
                   lambda: masktime[:,:,0], 
                   lambda:masktime)
    tf.Assert(tf.rank(masktime)==2, [mask])
    # return self.batch_oneh_proba_fn(tf.keras.layers.Reshape(
    #       [tf.cast(tf.reduce_sum(1-masktime[0]), tf.int32),
    #        y_true.shape[2], y_true.shape[3]])(
    #         tf.map_fn(
    #             lambda x: tf.boolean_mask(x[:,:,:,0], tf.cast(1-x[:,:,:,1], 'bool')),
    #             tf.stack([y_true, mask],axis=-1))))
    in_bmasked = tf.keras.layers.Lambda(lambda xx: tf.map_fn(
        lambda x: BatchBool().predict(x), xx)).predict(
        tf.stack([y_true, mask],axis=-1))
    return self.batch_oneh_proba_fn(tf.keras.layers.Reshape(
          [tf.cast(tf.reduce_sum(1-masktime[0]), tf.int32),
           y_true.shape[2], y_true.shape[3]])(
            in_bmasked))

class BatchAssignProb(tf.keras.layers.Layer):
  def __init__(self, assign_proba_center, centers):
    super(BatchAssignProb, self).__init__()
    self.batch_proba_fn = lambda y: tf.map_fn(
        lambda x: tf.map_fn(
            lambda xx: assign_proba_center(centers,xx),
            x),
        y
    )
  
  def call(self, inputs):
    # inputs are [y_true, mask]
    y_pred, mask = inputs
    mask = tf.cast(mask, y_pred.dtype)
    masktime = tf.cond(tf.rank(mask)==4, 
                   lambda: mask[:,:,:,0], 
                   lambda:mask)
    masktime = tf.cond(tf.rank(masktime)==3, 
                   lambda: masktime[:,:,0], 
                   lambda:masktime)
    tf.Assert(tf.rank(masktime)==2, [mask])
    # return self.batch_proba_fn(tf.keras.layers.Reshape(
    #       [tf.cast(tf.reduce_sum(1-masktime[0]), tf.int32),
    #        y_pred.shape[2], y_pred.shape[3]])(
    #         tf.map_fn(
    #             lambda x: tf.boolean_mask(x[:,:,:,0], tf.cast(1-x[:,:,:,1], 'bool')),
    #             tf.stack([y_pred, mask],axis=-1))))
    in_bmasked = tf.keras.layers.Lambda(lambda xx: tf.map_fn(
        lambda x: BatchBool()(x), xx))(
        tf.stack([y_pred, mask],axis=-1))
    return self.batch_proba_fn(tf.keras.layers.Reshape(
          [tf.cast(tf.reduce_sum(1-masktime[0]), tf.int32),
           y_pred.shape[2], y_pred.shape[3]])(
            in_bmasked))