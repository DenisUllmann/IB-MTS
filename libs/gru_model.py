# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:56:49 2022

@author: Denis
"""
import os
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation, Lambda, ZeroPadding2D, Cropping2D, Concatenate
from tensorflow.keras.applications import VGG16
from libs.pconv_layer import PConv2D
from libs.smart_load import smart_load_hdf5 as smart_load

class GRU(tf.keras.Model):
  def __init__(self, img_rows=512, img_cols=512, c_dim=3,
               with_centerloss=False, inference_only=False, 
               mask_ratio = 0.25, vgg_weights="imagenet", 
               net_name='default',*args, **kwargs):
    super(GRU, self).__init__(*args, **kwargs)
    self.with_centerloss = with_centerloss
    self.mask_ratio = mask_ratio
    self.img_rows = img_rows
    self.img_cols = img_cols
    self.c_dim = c_dim
    self.img_overlap = 30
    self.inference_only = inference_only
    self.net_name = net_name
    self.vgg_device = False

    # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
    self.vgg_layers = [3, 6, 10]
    # Scaling for VGG input
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]


    # Instantiate the vgg network
    if self.vgg_device:
        with tf.device(self.vgg_device):
            self.vgg = self.build_vgg(vgg_weights)
    else:
        self.vgg = self.build_vgg(vgg_weights)

    # Assertions
    if self.img_rows < 256:
        print('[WARNING] : Height is < 256 pixels, images will be zero padded')
    if self.img_cols < 256:
        print('[WARNING] : Width is < 256 pixels, images will be zero padded')

    # Set current epoch
    self.current_epoch = 0

    self.build_pconv_unet()
    self.built=True
    #self.compile()
  
  def compile(self, lr=0.0002):
    super(GRU, self).compile(metrics = [self.PSNR])
    self.optimizer = Adam(learning_rate=lr)
    self.loss = self.loss_total()

  def build_vgg(self, weights="imagenet"):
      """
      Load pre-trained VGG16 from keras applications
      Extract features to be used in loss function from last conv layer, see architecture at:
      https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
      """        
          
      # Input image to extract features from
      img = Input(shape=(self.img_rows, self.img_cols, self.c_dim))
      if self.c_dim == 1:
          img_T = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(img)

      # Mean center and rescale by variance as in PyTorch
      processed = Lambda(lambda x: (x-self.mean) / self.std)(img_T)
      
      # If inference only, just return empty model        
      # if self.inference_only:
      #     model = Model(inputs=img, outputs=[img for _ in range(len(self.vgg_layers))])
      #     model.trainable = False
      #     model.compile(loss='mse', optimizer='adam')
      #     return model
              
      # Get the vgg network from Keras applications
      if weights in ['imagenet', None]:
          vgg = VGG16(weights=weights, include_top=False)
      else:
          vgg = VGG16(weights=None, include_top=False)
          vgg.load_weights(weights, by_name=True)

      # Output the first three pooling layers
      #vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]      
      vgg = Model(vgg.inputs, [vgg.layers[i].output for i in self.vgg_layers]) 
      
      # Create model and compile
      model = Model(inputs=img, outputs=vgg(processed))
      model.trainable = False
      model.compile(loss='mse', optimizer='adam')

      return model
  
  def build_pconv_unet(self, train_bn=True):      

    inputs_img = Input((self.img_rows, self.img_cols, self.c_dim), name='inputs_img')
    inputs_mask = Input((self.img_rows, self.img_cols, self.c_dim), name='inputs_mask')
    inputs_pos_info = Input((1,), name='inputs_position_info')
    
    inputs_img_P = tf.keras.layers.Reshape([self.img_rows, self.img_cols])(inputs_img)
    inputs_mask_P = tf.keras.layers.Reshape([self.img_rows, self.img_cols])(inputs_mask)
    
    inputs_boolm = tf.keras.layers.Lambda(
        lambda x: x[:,:self.img_rows-int(self.img_rows*self.mask_ratio)])(inputs_img_P)
    encoder_l1 = tf.keras.layers.GRU(100,return_sequences = True, return_state=True)
    encoder_outputs1 = encoder_l1(inputs_boolm)
    encoder_states1 = encoder_outputs1[1]
    encoder_l2 = tf.keras.layers.GRU(100, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(int(self.img_rows*self.mask_ratio))(encoder_outputs2[0])
    #
    decoder_l1 = tf.keras.layers.GRU(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    decoder_l2 = tf.keras.layers.GRU(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.img_cols))(decoder_l2)
    #
    outputs = tf.concat([inputs_boolm, decoder_outputs2], axis=1)
    
    # if self.c_dim == 1:
    #     outputs = Lambda(lambda x: tf.reduce_mean(x, axis = -1, keepdims = True))(stacked_outputs)
    # else:
    #     outputs = stacked_outputs
    outputs = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=-1), [1,1,1,self.c_dim]))(outputs)
    # Setup the model inputs / outputs
    self.model = Model(inputs=[inputs_img, inputs_mask, inputs_pos_info], outputs=outputs)
    
  @tf.function
  def call(self, inputs, training=None, mask=None):
      # Generate the missing data
      return self.model(inputs, training=training)
  
  @tf.function
  def train_step(self, data):
    # when fit_generator > fit is used from models.py
    # yields: [masked, mask, bm, mask_tiling], [ori, position]
    masked, mask, info = data[0]
    real = data[1]
    
    # TRAIN
    with tf.GradientTape() as tape:
        # Generate missing data
        generated = self.model(
            [masked, mask, info], training=True)
        
        # Loss per sample from the PConv generation
        generated = tf.cast(generated, real.dtype)
        loss = self.loss(mask, real, generated)
        
        loss = tf.reduce_sum(loss)
        
        # divide by global batch size of samples used to train pcu[lu]
        loss /= tf.cast(tf.shape(masked)[0], loss.dtype)
    
    grads = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    # total dict
    return {"loss": loss}

  def loss_total(self):
      """
      Creates a loss function which sums all the loss components 
      and multiplies by their weights. See paper eq. 7.
      """
      def loss(mask, y_true, y_pred):
          # y_true = tf.ensure_shape(y_true_m, (None,)*3+(self.c_dim,2,))
          # y_true, mask = tf.unstack(tf.cast(y_true_m, y_pred.dtype), axis=-1)
          # Compute predicted image with non-hole pixels set to ground truth
          # tf.print("y_pred", y_pred.shape)
          mask = tf.cast(mask, y_pred.dtype)
          y_true = tf.cast(y_true, y_pred.dtype)
          y_comp = mask * y_true + (1-mask) * y_pred

          # Compute the vgg features. 
          if self.vgg_device:
              with tf.device(self.vgg_device):
                  vgg_out = self.vgg(y_pred)
                  vgg_gt = self.vgg(y_true)
                  vgg_comp = self.vgg(y_comp)
          else:
              vgg_out = self.vgg(y_pred)
              vgg_gt = self.vgg(y_true)
              vgg_comp = self.vgg(y_comp)
          
          # Compute loss components
          l1 = self.loss_valid(mask, y_true, y_pred)
          l2 = self.loss_hole(mask, y_true, y_pred)
          l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
          l4 = self.loss_style(vgg_out, vgg_gt)
          l5 = self.loss_style(vgg_comp, vgg_gt)
          l6 = self.loss_tv(mask, y_comp)
          if self.with_centerloss:
              l7 =self.center_loss(mask, y_true, y_pred, np.random.randn(3,240,1))
              return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6 + l7/500
          
          # Return loss function
          return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6
          # return [l1, l2, l3, l4, l5, l6, l7]

      return loss

  def loss_hole(self, mask, y_true, y_pred):
      """Pixel L1 loss within the hole / mask"""
      return self.l1(1-mask, (1-mask) * y_true, (1-mask) * y_pred)

  def loss_valid(self, mask, y_true, y_pred):
      """Pixel L1 loss outside the hole / mask"""
      return self.l1(mask, mask * y_true, mask * y_pred)

  def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
      """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
      loss = 0
      for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
          loss += self.l1(tf.ones_like(o), o, g) + self.l1(tf.ones_like(o), c, g)
      return loss
      
  def loss_style(self, output, vgg_gt):
      """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
      loss = 0
      for o, g in zip(output, vgg_gt):
          gmo = self.gram_matrix(o)
          loss += self.l1(tf.ones_like(gmo), gmo, self.gram_matrix(g))
      return loss

  def loss_tv(self, mask, y_comp):
      """Total variation loss, used for smoothing the hole region, see. eq. 6"""

      # Create dilated hole region using a 3x3 kernel of all 1s.
      kernel = tf.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
      dilated_mask = tf.nn.conv2d(
          1-mask,
          kernel,
          strides=1,
          padding="SAME",
          data_format='NHWC'
      )

      # Cast values to be [0., 1.], and compute dilated hole region of y_comp
      dilated_mask = tf.cast(tf.greater(dilated_mask, 0), 'float32')
      P = dilated_mask * y_comp

      # Calculate total variation loss
      p11 = P[:,1:,:,:]
      a = self.l1(tf.ones_like(p11), p11, P[:,:-1,:,:])
      p12 = P[:,:,1:,:]
      b = self.l1(tf.ones_like(p12), p12, P[:,:,:-1,:])        
      return a+b
  
  def assign_proba_center(self, tf_centers, center):
    dist = tf.reduce_sum(tf.abs(tf.tile(tf.expand_dims(center,0),[len(tf_centers)]+[1]*len(center.shape))-tf_centers),[1,2])
    return tf.exp(-dist)#/(tf.reduce_sum(tf.exp(-dist))+non_zero)

  def assign_center(self, tf_centers, center):
    return tf.argmin(tf.reduce_sum(tf.abs(tf.tile(tf.expand_dims(center,0),[len(tf_centers)]+[1]*len(center.shape))-tf_centers),[1,2]))

  def center_loss(self, mask, y_true, y_pred, centers, non_zero=1e-7):
    batch_oneh_proba_fn = tf.keras.layers.Lambda(lambda y: tf.map_fn(
        lambda x: tf.map_fn(
            lambda xx: tf.one_hot(self.assign_center(centers,xx),len(centers)),
            x),
        y
    ))
    batch_proba_fn = tf.keras.layers.Lambda(lambda y: tf.map_fn(
        lambda x: tf.map_fn(
            lambda xx: self.assign_proba_center(centers,xx),
            x),
        y
    ))
    mask = tf.cast(mask, y_pred.dtype)
    masktime = tf.cond(tf.rank(mask)==4, 
                    lambda: mask[:,:,:,0], 
                    lambda:mask)
    masktime = tf.cond(tf.rank(masktime)==3, 
                    lambda: masktime[:,:,0], 
                    lambda:masktime)
    tf.Assert(tf.rank(masktime)==2, [mask])
    # batch_assign_true = BatchAssignOneh(self.assign_center, centers)(
    #     [y_true, mask])
    batch_assign_true = batch_oneh_proba_fn(
        y_true[:,-int(self.mask_ratio*self.img_rows):])
    # tf.print(y_true[:,-60:])
    # tf.print(tf.reshape(
    #     tf.map_fn(
    #         lambda x: tf.boolean_mask(x[:,:,:,0], tf.cast(1-x[:,:,:,1], 'bool')),
    #         tf.stack([y_true, mask],axis=-1)),
    #     tf.concat([[tf.cast(tf.reduce_sum(1-masktime), tf.int32)],
    #       [y_true.shape[2], y_true.shape[3]]],axis=0)))
    # batch_assign_true = batch_oneh_proba_fn(tf.reshape(
    #     tf.map_fn(
    #         lambda x: tf.boolean_mask(x[:,:,:,0], tf.cast(1-x[:,:,:,1], 'bool')),
    #         tf.stack([y_true, mask],axis=-1)),
    #     tf.concat([[tf.cast(tf.reduce_sum(1-masktime), tf.int32)],
    #       [y_true.shape[2], y_true.shape[3]]],axis=0)))
    # batch_assign_pred = BatchAssignProb(self.assign_proba_center, centers)(
    #     [y_pred, mask])
    batch_assign_pred = batch_oneh_proba_fn(
        y_pred[:,-int(self.mask_ratio*self.img_rows):])
    # batch_assign_pred = batch_proba_fn(tf.reshape(
    #     tf.map_fn(
    #         lambda x: tf.boolean_mask(x[:,:,:,0], tf.cast(1-x[:,:,:,1], 'bool')),
    #         tf.stack([y_pred, mask],axis=-1)),
    #     tf.concat([[tf.cast(tf.reduce_sum(1-masktime), tf.int32)],
    #       [y_pred.shape[2], y_pred.shape[3]]], axis=0)))
    #return tf.keras.losses.CategoricalCrossentropy()(batch_assign_true,batch_assign_pred)
    return tf.reduce_sum(-batch_assign_true*tf.math.log(batch_assign_pred+non_zero), axis=[-2,-1])

  def fit_generator(self, generator, *args, **kwargs):
      """Fit the U-Net to a (images, targets) generator

      Args:
          generator (generator): generator supplying input image & mask, as well as targets.
          *args: arguments to be passed to fit_generator
          **kwargs: keyword arguments to be passed to fit_generator
      """
      self.fit(
          generator,
          *args, **kwargs
      )
      
  def summary(self):
      """Get summary of the UNet model"""
      print(self.model.summary())

  def load(self, filepath, train_bn=True, lr=0.0002):

      # Create UNet-like model
      self.build_pconv_unet()
      self.built = True
      self.compile(lr) 

      # Load weights into model
      epoch = int(os.path.basename(filepath).split('-')[-2].split('.')[-1])
      assert epoch > 0, "Could not parse weight file. Should include the epoch"
      print("load epoch #{}".format(epoch))
      self.current_epoch = epoch
      
      smart_load(filepath, self)
      # for layer in self.layers:
      #       layer.finalize_state()
      # self.load_weights(filepath) 

  @staticmethod
  def PSNR(mask, y_true, y_pred):
      """
      PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
      The equation is:
      PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
      
      Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
      two values (4.75) as MAX_I        
      """        
      #return 20 * tf.math.log(4.75) / tf.math.log(10.0) - 10.0 * tf.math.log(tf.reduce_mean(tf.square(y_pred - y_true))) / tf.math.log(10.0) 
      return - 10.0 * tf.math.log(tf.reduce_mean(tf.square(y_pred - y_true))) / tf.math.log(10.0)

  @staticmethod
  def current_timestamp():
      return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

  @staticmethod
  def l1(mask, y_true, y_pred):
      """Calculate the L1 loss used in all loss calculations"""
      mask = tf.cast(mask, y_pred.dtype)
      mask = tf.cond(tf.rank(mask)==4, 
                     lambda: mask[:,:,0], 
                     lambda:mask)
      
      mask = tf.cond(tf.rank(mask)==3, 
                     lambda: mask[:,:,0], 
                     lambda:mask)
      tf.Assert(tf.rank(mask)==2, [mask])
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)(
              tf.keras.layers.Flatten()(y_true),
              tf.keras.layers.Flatten()(y_pred)) * tf.reduce_sum(mask, axis=-1) / tf.reduce_sum(tf.ones_like(mask), axis=-1)
      # if len(y_true.shape) == 4:
      #     return tf.reduce_mean(tf.abs(y_pred - y_true), axis=[1,2,3])
      # elif len(y_true.shape) == 3:
      #     return tf.reduce_mean(tf.abs(y_pred - y_true), axis=[1,2])
      # else:
      #     raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

  @staticmethod
  def gram_matrix(x, norm_by_channels=False):
      """Calculate gram matrix used in style loss"""
      
      # Assertions on input
      assert len(x.shape) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor' 
      
      # Permute channels and get resulting shape
      x = tf.transpose(x, [0, 3, 1, 2])
      
      # Reshape x and do batch dot product
      x = tf.keras.layers.Reshape(x.shape[1:2]+[x.shape[2]*x.shape[3]])(x)
      # x = tf.reshape(x, x.shape[:2]+[x.shape[2]*x.shape[3]])
      
      # Normalize with channels, height and width
      return tf.keras.layers.Dot(axes=2)(
              [x]*2) /  tf.cast(tf.reduce_prod(x.shape[1:]), x.dtype)

  # Prediction functions
  ######################
  def predict(self, sample, **kwargs):
      """Run prediction using this model"""
      return self.model.predict([
          *sample,
          np.ones([sample[0].shape[0], 1], dtype=sample[0].dtype)], 
          **kwargs)