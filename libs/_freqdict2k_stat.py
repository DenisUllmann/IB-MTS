# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:46:46 2022

@author: Denis
"""
import tensorflow as tf
import numpy as np
import itertools

def dadd(dct, k, v):
    """
    adds values v (dict or scalar) to dict dct at key k, or creates the pair k:v
    """
    if isinstance(v, dict):
        for kk in v:
            dct[k] = add_or_createkey(dct[k], kk, v[kk])
    else:
        dct[k] += v
    return dct

def dcre(dct, k, v):
    """
    creates the pair k:v
    """
    return {**dct, k:v}

def add_or_createkey(dct, k, v):
    """
    adds values v (dict or scalar) to dict dct at key k, or creates the pair k:v
    """
    # dct is a dict of dicts or scalars
    # v is the dict or scalar to add at the key k
    if k in dct:
        dct = dadd(dct, k, v)
    else:
        dct = dcre(dct, k, v)
    return dct

def update_count(dt, classes, res, lab):
    """
    dt: dictionnary of the counts {per label {per class: {center: count}}}
    res: results of center assignments
    classes: class prediction
    lab: true label
    center: key of the center (a deterministic assignement)
    
    """
    for l in set(lab.numpy()):
        for k in set(classes.numpy()):
            dt = add_or_createkey(
                dt, l, 
                {k:{c:v for c,v in tf.transpose(tf.stack(
                    list(tf.unique_with_counts(
                        tf.gather_nd(res, tf.where(tf.logical_and(
                            tf.convert_to_tensor(lab)==l,
                            tf.convert_to_tensor(classes)==k))))[0::2]))).numpy()}})
    return dt

class Frequency2_lc_Dict(object):
    """
    NOT FINISHED
    Creates a '4D' dict with 4 levels of keys:
        labels: first meta info on data (true label)
        center: 2nd meta info on data (eg. kmeans on data)
    
    update_state add the count to the corresponding variables:
        lab, res: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, centers, name='centers_count'):
        self.name = name
        self.labels = labels
        self.centers = centers
        self.total = {l:{c: tf.Variable(0.0, name='total_%s%s'%(l,c)) for c in centers} for l in labels}
        self.count = tf.Variable(0.0, name='count')
    
    def update_state(self, lab, cla, res, sample_weight=None):
        # Update count and total
        self.count.assign_add(len(res))
        for l in self.labels:
            for k in self.classes:
                for c,v in tf.transpose(tf.stack(
                        list(tf.unique_with_counts(
                            tf.gather_nd(res, tf.where(tf.logical_and(
                                tf.convert_to_tensor(lab)==l,
                                tf.convert_to_tensor(cla)==k))))[0::2]))).numpy():
                    self.total[l][k][c].assign_add(v)
    
    def result(self):
        # Outputs dict of freq
        return {l:{k:{c: self.total[l][k][c]/self.count for c in self.centers} for k in self.classes} for l in self.labels}
    
    def reset_state(self):
        # To reset the state.
        for l in self.labels:
            for k in self.classes:
                for c in self.centers:
                    self.total[l][k][c].assign(0.0)
                    self.count.assign(0.0)

class NPFrequency2_lc_Dict(object):
    """
    Numpy version
    Creates a '2D' dict with 2 levels of keys:
        labels: first meta info on data (true label)
        centers: 2nd meta info on joint data (eg. kmeans on data)
    The dict is made of np.float32 with corresponding keys
    
    update_state add the count to the corresponding variables:
        lab, res: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, centers, name='centers_count'):
        self.name = name
        self.labels = labels
        self.centers = centers
        self.total = {l:{c: 0 for c in centers} for l in labels}
        self.count = 0
    
    # TODO amend the tf version
    def update_state(self, lab, res, sample_weight=None):
        # Update count and total based on cla1 and cla2 given same lab and res for both
        assert len(res)==len(lab), "length of the two joint sequences should be equal for joint stats"
        self.count += len(res)
        for (l,c),v in zip(
                *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                    list(zip(lab,res)),
                    return_counts=True, axis=0)]):
            self.total[l][c] += int(float(v))
    
    def count_assert(self):
        assert self.count==sum(sum(self.total[l][c] for c in self.centers) for l in self.labels), "counting error"
    
    def result(self):
        # Outputs dict of freq (joint l,c)
        self.count_assert()
        return {l:{c: self.total[l][c]/self.count for c in self.centers} for l in self.labels}
        
    def finalize_results(self):
        # returns 0 and not np.nan for 0*np.log2(0)
        xlog2x = lambda x:x*np.log2([x,1][int(x==0)])
        # Outputs dict of freq (l)
        self.joint =  self.result()
        self.margin_1 = {l: sum(self.joint[l][cc] for cc in self.centers) for l in self.labels}
    
        # Outputs dict of freq (c)
        self.margin_2 = {c: sum(self.joint[ll][c] for ll in self.labels) for c in self.centers}
    
        # Outputs the entropy of c
        # returns H( c)
        self.entropy_2 = -sum(self.margin_2[c]*np.log2(self.margin_2[c]) for c in self.centers)
        
        zero_cond = lambda x,y: [np.divide(x,y),0][int(y==0)]
        # Outputs dict of freq (joint c | l)
        self.result_cond_1 = {l:{c: zero_cond(self.joint[l][c],self.margin_1[l]) for c in self.centers} for l in self.labels}
    
        # Outputs dict of freq (c | l)
        self.margin_2_cond_1 = self.result_cond_1
    
        # Outputs the entropy of c|l
        # returns dict {class k: H( c|l)}
        self.entropy_2_cond_1 = {l:-sum(xlog2x(self.margin_2_cond_1[l][c]) for c in self.centers) for l in self.labels}  
    
    def info_result_2(self):
        # Returns Information Theory details on classes2 and centers
        # Returns dict{
            # 'entropy': float
        h2 = self.entropy_2
        return {'entropy': h2}
    
    def info_result_2_cond_1(self):
        # Returns Information Theory details on classes2 and centers, conditioned by classes1
        # Returns dict{
            # 'entropy': dict{}
        h2 = self.entropy_2_cond_1
        return {'entropy': h2}
    
    def reset_state(self):
        # To reset the state.
        for l in self.labels:
            for c in self.centers:
                self.total[l][c] = 0.0
                self.count = 0.0
