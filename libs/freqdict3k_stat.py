# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:16:51 2022

@author: Denis
"""
import tensorflow as tf
import numpy as np

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

class Frequency3kDict(object):
    """
    Creates a '3D' dict with 3 levels of keys:
        labels: first meta info on data (true label)
        classes: second meta info on data (classification)
        centers: 3rd meta info on data (eg. kmeans on data)
    The dict is made of tf.Variables with corresponding keys
    
    update_state add the count to the corresponding variables:
        lab, cla, res: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, classes, centers, name='centers_count'):
        self.name = name
        self.labels = labels
        self.classes = classes
        self.centers = centers
        self.total = {l:{k:{c: tf.Variable(0.0, name='total_%s%s%s'%(l,k,c)) for c in centers} for k in classes} for l in labels}
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

class NPFrequency3kDict(object):
    """
    Numpy version
    Creates a '3D' dict with 3 levels of keys:
        labels: first meta info on data (true label)
        classes: second meta info on data (classification)
        centers: 3rd meta info on data (eg. kmeans on data)
    The dict is made of np.float32 with corresponding keys
    
    update_state add the count to the corresponding variables:
        lab, cla, res: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, classes, centers, name='centers_count'):
        self.name = name
        self.labels = labels
        self.classes = classes
        self.centers = centers
        self.total = {l:{k:{c: 0.0 for c in centers} for k in classes} for l in labels}
        self.count = 0.0
    
    # TODO amend the tf version
    def update_state(self, lab, cla, res, sample_weight=None):
        # cla and lab must be str
        # Update count and total
        assert len(res)==len(cla)==len(lab), "length of the two joint sequences should be equal for joint stats"
        self.count += len(res)
        for (l,k,c),v in zip(
                *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                    list(zip(lab,cla,res)),
                    return_counts=True, axis=0)]):
            self.total[l][k][c] += int(float(v))
    
    def count_assert(self):
        assert self.count==sum(sum(sum(self.total[l][k][c] for c in self.centers) for k in self.classes) for l in self.labels), "counting error"
    
    def result(self):
        # Outputs dict of freq (joint l,k,c1,c2)
        self.count_assert()
        return {l:{k:{c: self.total[l][k][c]/self.count for c in self.centers} for k in self.classes} for l in self.labels}
    
    def margin_3(self):
        # Outputs dict of freq (c)
        res =  self.result()
        return {c: sum(sum(res[l][k][c] for k in self.classes) for l in self.labels) for c in self.centers}
    
    def entropy_3(self):
        # Outputs the entropy of c
        # returns H( c)
        p_c = self.margin_3()
        return sum(p_c[c]*np.log2(p_c[c]) for c in self.centers)
    
    def result_cond_1(self):
        # Outputs dict of freq (joint k,c | l)
        self.count_assert()
        return {l:{k:{c: self.total[l][k][c]/sum(sum(self.total[l][kk][cc] for cc in self.centers) for kk in self.classes) for c in self.centers} for k in self.classes} for l in self.labels}
    
    def entropy_3_cond_1(self):
        # Outputs the entropy of c|k
        # returns dict {class k: H( c|k)}
        p_c_cond_k = self.result_cond_1()
        return {k:sum(p_c_cond_k[k][c]*np.log2(p_c_cond_k[k][c]) for c in self.centers) for k in self.classes}
    
    def result_cond_2(self):
        # Outputs dict of freq (joint l,c | k)
        self.count_assert()
        return {l:{k:{c: self.total[l][k][c]/sum(sum(self.total[ll][k][cc] for cc in self.centers) for ll in self.labels) for c in self.centers} for k in self.classes} for l in self.labels}
    
    def entropy_3_cond_2(self):
        # Outputs the entropy of c|l
        # returns dict {label k: H( c|l)}
        p_c_cond_l = self.result_cond_2()
        return {l:sum(p_c_cond_l[l][c]*np.log2(p_c_cond_l[l][c]) for c in self.centers) for l in self.labels}
    
    def result_cond_1_2(self):
        # Outputs dict of freq (joint c | l,k)
        self.count_assert()
        return {l:{k:{c: self.total[l][k][c]/sum(self.total[l][k][cc] for cc in self.centers) for c in self.centers} for k in self.classes} for l in self.labels}
    
    def entropy_3_cond_1_2(self):
        # Outputs the entropy of c|k,l
        # returns dict {class k:{label l: H( c|k,l)}}
        p_c_cond_kl = self.result_cond_1_2()
        return {k:{l:sum(p_c_cond_kl[k][l][c]*np.log2(p_c_cond_kl[k][l][c]) for c in self.centers) for l in self.labels} for k in self.classes}

    def info_result_3_cond_1_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{'entropy': dict{dict{}}}
        return {'entropy': self.entropy_3_cond_1_2()}
    
    def info_result_3_cond_1(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{'entropy': dict{}}
        return {'entropy': self.entropy_3_cond_1()}
    
    def info_result_3_cond_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{'entropy': dict{}}
        return {'entropy': self.entropy_3_cond_2()}
    
    def info_result_3(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{'entropy': float}
        return {'entropy': self.entropy_3()}
    
    def reset_state(self):
        # To reset the state.
        for l in self.labels:
            for k in self.classes:
                for c in self.centers:
                    self.total[l][k][c] = 0.0
                    self.count = 0.0

