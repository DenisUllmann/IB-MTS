import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, ReLU, ZeroPadding2D, Flatten
# from keras.utils.multi_gpu_utils import multi_gpu_model

# all in tf.keras

from libs.pconv_layer_tf import PConv2D

class CategoricalLookup(object):
    """
    lookup maps key (may be str) to values of a class_dict (may be lists)
    lookdown performs the inverse op (not always valid)
    """
    def __init__(self, class_dict):
        self.class_dict = class_dict
        self.keys_tensor = list(self.class_dict.keys())
        self.idx_tensor = tf.constant(np.arange(len(self.keys_tensor)))
        self.vals_tensor = tf.stack([self.class_dict[k] for k in self.keys_tensor],)
        self.keys_tensor = tf.constant(self.keys_tensor)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self.keys_tensor, self.idx_tensor),
            default_value=-1)
    
    def lookup(self, keys):
        return tf.gather(self.vals_tensor,
                  self.table.lookup(keys))
      
    def lookdown(self, vals):
        return tf.gather(self.keys_tensor,tf.argmin(tf.keras.losses.cosine_similarity(
            tf.transpose(tf.tile(tf.expand_dims(vals,-1), [1,1,len(list(self.class_dict.keys()))]),[0,2,1]), 
            self.vals_tensor), axis=1))

class CategoricalNPLookdown(object):
    """
    numpy lookdown maps values of a class_dict (may be lists) to keys (may be str)
    """
    def __init__(self, class_dict):
        self.class_dict = class_dict
        self.keys_tensor = list(self.class_dict.keys())
        self.vals_tensor = np.vstack([self.class_dict[k] for k in self.keys_tensor],)
        self.keys_tensor = np.asarray(self.keys_tensor)
    
    def aprox_zeros(self, a):
        a = np.asarray(a)
        a[a==0]=1e-10
        return a
    
    def cosine_similarity(self,a,b):
        a = self.aprox_zeros(a)
        b = self.aprox_zeros(b)
        return 1-np.sum(np.multiply(a,b), axis=-1)/np.multiply(
                np.linalg.norm(a, axis=-1),
                np.linalg.norm(b, axis=-1))
    
    def lookdown(self, vals):
        if self.vals_tensor.shape[1]==1:
            # when dim==1, cosine_similarity is always == 0
            return self.keys_tensor[np.argmin(np.sum(np.abs(
                np.transpose(np.tile(np.expand_dims(vals,-1), 
                                     [1,1,len(self.keys_tensor)]),[0,2,1])- 
                np.transpose(np.tile(np.expand_dims(self.vals_tensor,-1), 
                                     [1,1,len(vals)]),[2,0,1])), axis=-1),
                axis=1)]
        else:
            return self.keys_tensor[np.argmin(self.cosine_similarity(
                np.transpose(np.tile(np.expand_dims(vals,-1), 
                                     [1,1,len(self.keys_tensor)]),[0,2,1]), 
                np.transpose(np.tile(np.expand_dims(self.vals_tensor,-1), 
                                     [1,1,len(vals)]),[2,0,1])),
                axis=1)]

class CatLossFromDict(tf.keras.losses.Loss):
    def __init__(self, class_dict, name='categorical_crossentropy_fromdict', **kwargs):
        super(CatLossFromDict, self).__init__(**kwargs)
        self.class_dict = class_dict
    
    def call(self, y_true, y_pred):
        catmap = CategoricalLookup(self.class_dict)
        return tf.keras.losses.CategoricalCrossentropy()(
            catmap.lookup(y_true), 
            y_pred)

class BinLossFromDict(tf.keras.losses.Loss):
    def __init__(self, class_dict, name='binary_crossentropy_fromdict', **kwargs):
        super(BinLossFromDict, self).__init__(**kwargs)
        self.class_dict = class_dict
    
    def call(self, y_true, y_pred):
        catmap = CategoricalLookup(self.class_dict)
        return tf.keras.losses.BinaryCrossentropy()(
            catmap.lookup(y_true), 
            y_pred)

class CatCroFromDict(tf.keras.metrics.Metric):
    
    def __init__(self, class_dict, name='categorical_crossentropy_fromdict', **kwargs):
        super(CatCroFromDict, self).__init__(name=name, **kwargs)
        self.cat_cro = self.add_weight(name='cata', initializer='zeros')
        self.class_dict = class_dict
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = CatLossFromDict(self.class_dict)(y_true, y_pred)
        self.cat_cro.assign_add(tf.reduce_sum(values))
    
    def result(self):
        return self.cat_cro
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_cro.assign(0.0)

class BinCroFromDict(tf.keras.metrics.Metric):
    
    def __init__(self, class_dict, name='binary_crossentropy_fromdict', **kwargs):
        super(BinCroFromDict, self).__init__(name=name, **kwargs)
        self.bin_cro = self.add_weight(name='bina', initializer='zeros')
        self.class_dict = class_dict
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = BinLossFromDict(self.class_dict)(y_true, y_pred)
        self.bin_cro.assign_add(tf.reduce_sum(values))
    
    def result(self):
        return self.bin_cro
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bin_cro.assign(0.0)

class CatAccFromDict(tf.keras.metrics.Metric):
    
    def __init__(self, class_dict, name='categorical_accuracy_fromdict', **kwargs):
        super(CatAccFromDict, self).__init__(name=name, **kwargs)
        self.class_dict = class_dict
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(
            tf.reduce_prod(tf.cast(tf.equal(
                CategoricalLookup(self.class_dict)(y_true),
                tf.cast(y_pred, tf.int32)), tf.int32), axis=0)))
        self.count.assign_add(tf.shape(y_true)[0])
    
    def result(self):
        return self.total/self.count
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.0)
        self.count.assign(0.0)

class BinAccFromDict(tf.keras.metrics.Metric):
    
    def __init__(self, class_dict, name='categorical_accuracy_fromdict', **kwargs):
        super(BinAccFromDict, self).__init__(name=name, **kwargs)
        self.class_dict = class_dict
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(
            tf.cast(tf.equal(
                CategoricalLookup(self.class_dict)(y_true),
                tf.cast(y_pred, tf.int32)), tf.int32)))
        self.count.assign_add(tf.shape(y_true)[0])
    
    def result(self):
        return self.total/self.count
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.0)
        self.count.assign(0.0)

###
### Numpy versions of the previous metrics (difficulty to get the result() and count values of tf.keras metrics in TF1.13)
###

class NP_CategoricalCrossentropy(object):
    
    def __init__(self, axis=-1, epsilon=1e-7,
                 name=tf.keras.metrics.CategoricalCrossentropy().name):
        self.name = name
        self.axis = axis
        self.total = 0.0
        self.count = 0
        self._epsilon = epsilon
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred /= np.sum(y_pred, axis=self.axis, keepdims=True)
        y_pred = np.clip(y_pred, self._epsilon, 1.-self._epsilon)
        values = -np.sum(y_true * np.log(y_pred), axis=self.axis)
        self.total += np.sum(values)
        self.count += np.size(values)
    
    def result(self):
        return np.divide(self.total,self.count)
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0.0
        self.count = 0

class NP_BinaryCrossentropy(object):
    
    
    def __init__(self, axis=-1, epsilon=1e-7,
                 name=tf.keras.metrics.BinaryCrossentropy().name):
        self.name = name
        self.axis = axis
        self.total = 0.0
        self.count = 0
        self._epsilon = epsilon
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = np.clip(y_pred, self._epsilon, 1.-self._epsilon)
        values = - y_true * np.log(y_pred + self._epsilon)
        values -= (1-y_true) * np.log(1 - y_pred + self._epsilon)
        self.total += np.sum(values)
        self.count += np.size(values)
    
    def result(self):
        return np.divide(self.total,self.count)
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0.0
        self.count = 0

class NP_CategoricalAccuracy(object):
    # Compare the categorical representations (may not be one hot representations)
    def __init__(self, 
                 class_assign_fn = None,
                 name=tf.keras.metrics.CategoricalAccuracy().name):
        self.class_assign_fn = class_assign_fn if (
            class_assign_fn is not None) else lambda x: np.argmax(
                np.round(x), axis=-1)
        self.name = name
        self.total = 0.0
        self.count = 0
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        class_trues = self.class_assign_fn(y_true)
        self.total += np.sum(class_trues==self.class_assign_fn(y_pred))
        self.count += np.size(class_trues)
    
    def result(self):
        return np.divide(self.total, self.count)
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0.0
        self.count = 0

class NP_BinaryAccuracy(object):
    
    def __init__(self, name=tf.keras.metrics.BinaryAccuracy().name):
        self.name = name
        self.total = 0.0
        self.count = 0
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = np.round(y_true)==np.round(y_pred)
        self.total += np.sum(values)
        self.count += np.size(values)
    
    def result(self):
        return np.divide(self.total,self.count)
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total = 0.0
        self.count = 0

class PConvDense(object):

    def __init__(self, img_rows=512, img_cols=512, c_dim=3, inference_only=False, 
                 classes=['QS', 'AR', 'PF', 'FL'], 
                 class_dict={'QS': [1,0,0,0], 'AR': [0,1,0,0], 
                             'PF': [0,0,1,0], 'FL': [0,0,0,1]}, 
                 noclass = 'noclass', dtype='float32',
                 net_name='default', gpus=1, vgg_device=None):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None
        
        Args:
            img_rows (int): image height.
            img_cols (int): image width.
            inference_only (bool): initialize BN layers for inference.
            classes (list): list of classes to predict.
            class_dict (dict): dictionary of the outputs to give for each input labeled data
            net_name (str): Name of this network (used in logging).
            gpus (int): How many GPUs to use for training.
        """
        
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.c_dim = c_dim
        self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.classes = classes
        # print('classes_classifier_init', self.classes)
        self.class_dict = class_dict
        self.noclass = noclass
        
        self.lkd_dict_map = {k:tf.constant([int(ii==i) for ii in range(len(self.classes))], dtype=dtype) for i,k in enumerate(self.classes)}
        if self.noclass is not None:
            self.lkd_dict_map = {
                **self.lkd_dict_map,
                **{self.noclass:tf.constant([0]*len(self.classes), dtype=dtype)}}
        
        # {'class': catlabel_np.array[]}
        self.lkdNP_dict_map = {k:np.asarray([int(ii==i) for ii in range(len(self.classes))], dtype=dtype) for i,k in enumerate(self.classes)}
        if self.noclass is not None:
            self.lkdNP_dict_map = {
                **self.lkdNP_dict_map,
                **{self.noclass:np.asarray([0]*len(self.classes), dtype=dtype)}}
        

        # Assertions
        if self.img_rows < 256:
            print('[WARNING] : Height is < 256 pixels, images will be zero padded')
        if self.img_cols < 256:
            print('[WARNING] : Width is < 256 pixels, images will be zero padded')

        # Set current epoch
        self.current_epoch = 0
                
        # Create PConv-Dense-like model
        if self.gpus <= 1:
            self.model = self.build_pconv_dense()
            self.compile_pconv_dense(self.model)            
        else:
            assert False, "do not have the old keras multi_gpu_model option"
            # with tf.device("/cpu:0"):
            #     self.model = self.build_pconv_unet()
            # self.model = multi_gpu_model(self.model, gpus=self.gpus)
            # self.compile_pconv_dense(self.model)
    
    def assign_class(self, pred_class, dtype='float32'):
        """
        Maps the output of the classifier (sigmoid vector) to a class among self.classes by smallest cosine similarity (biggest index) to the one hot outputs
        """
        # print('lk_dict',{k:tf.constant([int(ii==i) for ii in range(len(self.classes))], dtype=dtype) for i,k in enumerate(self.classes)})
        return CategoricalLookup(self.lkd_dict_map).lookdown(pred_class)
    
    def np_assign_class(self, pred_class, dtype='float32'):
        """
        Like self.assign_class but for np.array
        """
        return CategoricalNPLookdown(self.lkdNP_dict_map).lookdown(pred_class)
    
    def build_pconv_dense(self, train_bn=True):      

        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, self.c_dim), name='inputs_img', dtype=tf.float32)
        inputs_mask = Input((self.img_rows, self.img_cols, self.c_dim), name='inputs_mask', dtype=tf.float32)
        inputs_pos_info = Input((1,), name='inputs_position_info')
        if np.log2(self.img_rows)-int(np.log2(self.img_rows))!=0 or np.log2(self.img_cols)-int(np.log2(self.img_cols))!=0: # dimensions are not a power of 2
            r_pow2 = max(8,np.ceil(np.log2(self.img_rows)))
            pad_r1 = (2 ** r_pow2 - self.img_rows) // 2
            pad_r2 = (2 ** r_pow2 - self.img_rows) - ((2 ** r_pow2 - self.img_rows) // 2)
            c_pow2 = max(8,np.ceil(np.log2(self.img_cols)))
            pad_c1 = (2 ** c_pow2 - self.img_cols) // 2
            pad_c2 = (2 ** c_pow2 - self.img_cols) - ((2 ** c_pow2 - self.img_cols) // 2)
            
            inputs_img_P = ZeroPadding2D(((pad_r1, pad_r2),(pad_c1, pad_c2)))(inputs_img)
            inputs_mask_P = ZeroPadding2D(((pad_r1, pad_r2),(pad_c1, pad_c2)))(inputs_mask)
        else:
            inputs_img_P = inputs_img
            inputs_mask_P = inputs_mask
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same', dtype=inputs_img.dtype)([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        # tf.print('tf_input', inputs_img_P.shape)
        x, m = encoder_layer(inputs_img_P, inputs_mask_P, 64, 7, bn=False)
        x, m = encoder_layer(x, m, 128, 5)
        x, m = encoder_layer(x, m, 256, 5)
        x, m = encoder_layer(x, m, 512, 3)
        x, m = encoder_layer(x, m, 512, 3)
        x, m = encoder_layer(x, m, 512, 3)
        x, m = encoder_layer(x, m, 512, 3)
        x, m = encoder_layer(x, m, 512, 3)
        
        # DENSE
        def dense_layer(img_in, units, bn=True):
            dense = Dense(units)(img_in)
            if bn:
                dense = BatchNormalization()(dense)
            dense = ReLU()(dense)
            return dense
        
        x = Flatten()(x)
        for u in [int(256/2**k) for k in range(9)]:
            if u >= 2*len(self.classes):
                x = dense_layer(x, u)
        
        outputs = Dense(len(self.classes),
                        activation = 'sigmoid', 
                        name='outputs_class')(x)
        
        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask, inputs_pos_info],
                      outputs=outputs)

        return model
    
    def compile_pconv_dense(self, model, lr=0.0002):
        if len(self.classes)>1:
            model.compile(
                optimizer = Adam(lr=lr),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalCrossentropy(),
                         tf.keras.metrics.CategoricalAccuracy()]
            )
        else:
            model.compile(
                optimizer = Adam(lr=lr),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryCrossentropy(),
                         tf.keras.metrics.BinaryAccuracy()]
            )
        print("model.model metrics", model.metrics)

    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model = self.build_pconv_dense(train_bn)
        self.compile_pconv_dense(self.model, lr) 

        # Load weights into model
        epoch = int(os.path.basename(filepath).split('-')[-2].split('.')[-1])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)
    
    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        # return self.model.predict([*sample,tf.ones(sample[0].shape[0], 1)], **kwargs)
        # print('dtype_predict0', sample[0].dtype)
        # print('dtype_predict1', sample[1].dtype)
        return self.model.predict([*sample,np.ones([sample[0].shape[0], 1], dtype=sample[0].dtype)], **kwargs)
