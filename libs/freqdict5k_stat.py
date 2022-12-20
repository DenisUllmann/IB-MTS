# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:53:30 2022

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

class Frequency5kDict(object):
    """
    NOT FINISHED
    Creates a '5D' dict with 5 levels of keys:
        labels: first meta info on data (true label)
        classes: second meta info on data (classification)
        center: 3rd meta info on data (eg. kmeans on data)
        center: 4rd meta info on data (eg. kmeans on data)
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

class NPFrequency5_lkkcc_Dict(object):
    """
    Numpy version
    Creates a '5D' dict with 5 levels of keys:
        labels: first meta info on data (true label)
        classes: second 'input' meta info on data (classification)
        classes: 3rd 'output' meta info on data (classification)
        centers: 4th 'input' meta-1 info on joint data (eg. kmeans on data)
        centers: 5th 'output' meta-2 info on joint data (eg. kmeans on data)
    The dict is made of np.float32 with corresponding keys
    
    update_state add the count to the corresponding variables:
        lab, cla1, cla2, res1, res2: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, classes1, classes2, centers1, centers2, name='centers_count'):
        self.name = name
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.centers1 = centers1
        self.centers2 = centers2
        self.total = {l:{k1:{k2:{c1:{c2: 0 for c2 in centers2} for c1 in centers1} for k2 in classes2} for k1 in classes1} for l in labels}
        self.count = 0.
    
    # TODO amend the tf version
    def update_state(self, lab, cla, res, sample_weight=None):
        # Update count and total based on res1 and res2 given same lab and cla for both
        (cla1, cla2) = cla
        (res1, res2) = res
        assert len(res1)==len(res2)==len(cla1)==len(cla2)==len(lab), "length of the two joint sequences should be equal for joint stats"
        self.count += len(res1)
        # print('input_unique',list(zip(lab,cla1,cla2,res1,res2)))
        for (l,k1,k2,c1,c2),v in zip(
                *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                    list(zip(lab,cla1,cla2,res1,res2)),
                    return_counts=True, axis=0)]):
            self.total[l][k1][k2][c1][c2] += int(float(v))
        
    def count_assert(self):
        assert self.count==sum(sum(sum(sum(sum(self.total[l][k1][k2][c1][c2] for c2 in self.centers2) for c1 in self.centers1) for k2 in self.classes2) for k1 in self.classes1) for l in self.labels), "counting error"
    
    def result(self):
        # Outputs dict of freq (joint l,k1,k2,c1,c2)
        self.count_assert()
        return {l:{k1:{k2:{c1:{c2: self.total[l][k1][k2][c1][c2]/self.count for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
    def finalize_results(self):
        # returns 0 and not np.nan for 0*np.log2(0)
        xlog2x = lambda x:x*np.log2([x,1][int(x==0)])
        xlog2x_yz = lambda x,y,z:[x*np.log2(np.divide(x,y*z)),0][x==0]
        
        # Outputs dict of freq (l,k1,k2)
        self.joint = self.result()
        self.margin_1_2_3_4 = {l:{k1:{k2:{c1: sum(self.joint[l][k1][k2][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        self.margin_1_2_3_5 = {l:{k1:{k2:{c2: sum(self.joint[l][k1][k2][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        self.margin_1_2_4_5 = {l:{k1:{c1:{c2: sum(self.joint[l][k1][k22][c1][c2] for k22 in self.classes2) for c2 in self.centers2} for c1 in self.centers1} for k1 in self.classes1} for l in self.labels}
        self.margin_1_3_4_5 = {l:{k2:{c1:{c2: sum(self.joint[l][k11][k2][c1][c2] for k11 in self.classes1) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for l in self.labels}
        
        self.margin_1_2_3 = {l:{k1:{k2: sum(self.margin_1_2_3_4[l][k1][k2][c11] for c11 in self.centers1) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        
        # Outputs dict of freq (l,k1)
        self.margin_1_2 = {l:{k1: sum(self.margin_1_2_3[l][k1][k22] for k22 in self.classes2) for k1 in self.classes1} for l in self.labels}
        
        # Outputs dict of freq (l,k2)
        self.margin_1_3 = {l:{k2: sum(self.margin_1_2_3[l][k11][k2] for k11 in self.classes1) for k2 in self.classes2} for l in self.labels}
        
        # Outputs dict of freq (k1)
        self.margin_1 = {l: sum(self.margin_1_2[l][k11] for k11 in self.classes1) for l in self.labels}
        
        # Outputs dict of freq (k1,k2)
        self.margin_2_3 = {k1:{k2: sum(self.margin_1_2_3[ll][k1][k2] for ll in self.labels) for k2 in self.classes2} for k1 in self.classes1}
    
        # Outputs dict of freq (k1)
        self.margin_2 = {k1: sum(self.margin_2_3[k1][k22] for k22 in self.classes2) for k1 in self.classes1}
    
        # Outputs dict of freq (k2)
        self.margin_3 = {k2: sum(self.margin_2_3[k1][k2] for k1 in self.classes1) for k2 in self.classes2}
    
        # Outputs dict of freq (l,c1,c2)
        res =  self.margin_1_2_4_5
        self.margin_1_4_5 = {l:{c1:{c2: sum(res[l][k11][c1][c2] for k11 in self.classes1) for c2 in self.centers2} for c1 in self.centers1} for l in self.labels}
        
        # Outputs dict of freq (c1,c2)
        res =  self.margin_1_4_5
        self.margin_4_5 = {c1:{c2: sum(res[ll][c1][c2] for ll in self.labels) for c2 in self.centers2} for c1 in self.centers1}
    
        # Outputs dict of freq (c1)
        res =  self.margin_4_5
        self.margin_4 = {c1: sum(res[c1][c2] for c2 in self.centers2) for c1 in self.centers1}
    
        # Outputs dict of freq (c2)
        res =  self.margin_4_5
        self.margin_5 = {c2: sum(res[c1][c2] for c1 in self.centers1) for c2 in self.centers2}
    
        # Outputs the mutual information between the keys l1 and k2 of the dict
        # returns I( k1 ; k2)
        self.mutual_info_2_3 = sum(sum(xlog2x_yz(
            self.margin_2_3[k1][k2],
            self.margin_2[k1],
            self.margin_3[k2]) for k1 in self.classes1) for k2 in self.classes2)
    
        # Outputs the entropy of k1
        # returns H( k1)
        self.entropy_2 = -sum(xlog2x(self.margin_2[k1]) for k1 in self.classes1)
    
        # Outputs the entropy of 22
        # returns H( k2)
        self.entropy_3 = -sum(xlog2x(self.margin_3[k2]) for k2 in self.classes2)
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns I( c1 ; c2)
        self.mutual_info_4_5 = sum(sum(xlog2x_yz(
            self.margin_4_5[c1][c2],
            self.margin_4[c1],
            self.margin_5[c2]) for c1 in self.centers1) for c2 in self.centers2)
    
        # Outputs the entropy of c1
        # returns H( c1)
        self.entropy_4 = -sum(xlog2x(self.margin_4[c1]) for c1 in self.centers1)
    
        # Outputs the entropy of c2
        # returns H( c2)
        self.entropy_5 = -sum(xlog2x(self.margin_5[c2]) for c2 in self.centers2)
        
        zero_cond = lambda x,y: [np.divide(x,y),0][int(y==0)]
        # Outputs dict of freq (joint k1,k2,c1,c2 | l)
        self.result_cond_1 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_1[l]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        
        # Outputs dict of freq (k1,k2 | l)
        self.margin_2_3_cond_1 = {l:{k1:{k2: zero_cond(self.margin_1_2_3[l][k1][k2],self.margin_1[l]) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (k1 | l)
        self.margin_2_cond_1 = {l:{k1: zero_cond(self.margin_1_2[l][k1],self.margin_1[l]) for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (k2 | l)
        self.margin_3_cond_1 = {l:{k2: zero_cond(self.margin_1_3[l][k2],self.margin_1[l]) for k2 in self.classes2} for l in self.labels}
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class l: I( k1|l ; k2|l)}}
        self.mutual_info_2_3_cond_1 = {l:sum(sum(xlog2x_yz(
            self.margin_2_3_cond_1[l][k1][k2],
            self.margin_2_cond_1[l][k1],
            self.margin_3_cond_1[l][k2]) for k1 in self.classes1) for k2 in self.classes2) for l in self.labels}
    
        # Outputs dict of freq (c1,c2 | l)
        self.margin_4_5_cond_1 = {l:{c1:{c2: zero_cond(self.margin_1_4_5[l][c1][c2],self.margin_1[l]) for c2 in self.centers2} for c1 in self.centers1} for l in self.labels}
    
        # Outputs dict of freq (c1 | l)
        self.margin_4_cond_1 = {l:{c1: sum(self.margin_4_5_cond_1[l][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for l in self.labels}
    
        # Outputs dict of freq (c2 | l)
        self.margin_5_cond_1 = {l:{c2: sum(self.margin_4_5_cond_1[l][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for l in self.labels}
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class l: I( c1|l ; c2|l)}}
        self.mutual_info_4_5_cond_1 = {l:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_1[l][c1][c2],
            self.margin_4_cond_1[l][c1],
            self.margin_5_cond_1[l][c2]) for c1 in self.centers1) for c2 in self.centers2) for l in self.labels}
    
        # Outputs the entropy of k1|l
        # returns dict {class k: H( k1|l)}
        self.entropy_2_cond_1 = {l:-sum(xlog2x(self.margin_2_cond_1[l][k11]) for k11 in self.classes1) for l in self.labels}
    
        # Outputs the entropy of k2|l
        # returns dict {class k: H( k2|l)}
        self.entropy_3_cond_1 = {l:-sum(xlog2x(self.margin_3_cond_1[l][k22]) for k22 in self.classes2) for l in self.labels}
    
        # Outputs the entropy of c1|l
        # returns dict {class k: H( c1|l)}
        self.entropy_4_cond_1 = {l:-sum(xlog2x(self.margin_4_cond_1[l][c1]) for c1 in self.centers1) for l in self.labels}
    
        # Outputs the entropy of c2|l
        # returns dict {class k: H( c2|l)}
        self.entropy_5_cond_1 = {l:-sum(xlog2x(self.margin_5_cond_1[l][c2]) for c2 in self.centers2) for l in self.labels}
    
        # Outputs dict of freq (joint l,k2,c1,c2 | k1)
        self.result_cond_2 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_2[k1]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}

        # Outputs dict of freq (c1,c2 | k1)
        self.margin_4_5_cond_2 = {k1:{c1:{c2: sum(sum(self.result_cond_2[ll][k1][k22][c1][c2] for k22 in self.classes2) for ll in self.labels) for c2 in self.centers2} for c1 in self.centers1} for k1 in self.classes1}
    
        # Outputs dict of freq (c1 | k1)
        self.margin_4_cond_2 = {k1:{c1: sum(self.margin_4_5_cond_2[k1][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k1 in self.classes1}
    
        # Outputs dict of freq (c2 | k1)
        self.margin_5_cond_2 = {k1:{c2: sum(self.margin_4_5_cond_2[k1][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k1 in self.classes1}

        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class k1: I( c1|k1 ; c2|k1)}
        self.mutual_info_4_5_cond_2 = {k1:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_2[k1][c1][c2],
            self.margin_4_cond_2[k1][c1],
            self.margin_5_cond_2[k1][c2]) for c1 in self.centers1) for c2 in self.centers2) for k1 in self.classes1}
    
        # Outputs the entropy of c1|k1
        # returns dict {class k1: H( c1|k1)}
        self.entropy_4_cond_2 = {k1:-sum(xlog2x(self.margin_4_cond_2[k1][c1]) for c1 in self.centers1) for k1 in self.classes1}
    
        # Outputs the entropy of c2|k1
        # returns dict {class k: H( c2|k1)}
        self.entropy_5_cond_2 = {k1:-sum(xlog2x(self.margin_5_cond_2[k1][c2]) for c2 in self.centers2) for k1 in self.classes1}    
    
        # Outputs dict of freq (joint l,k1,c1,c2 | k2)
        self.result_cond_3 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_3[k2]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}

        # Outputs dict of freq (c1,c2 | k2)
        self.margin_4_5_cond_3 = {k2:{c1:{c2: sum(sum(self.result_cond_3[ll][k11][k2][c1][c2] for k11 in self.classes1) for ll in self.labels) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2}
    
        # Outputs dict of freq (c1 | k2)
        self.margin_4_cond_3 = {k2:{c1: sum(self.margin_4_5_cond_3[k2][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k2 in self.classes2}
    
        # Outputs dict of freq (c2 | k2)
        self.margin_5_cond_3 = {k2:{c2: sum(self.margin_4_5_cond_3[k2][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k2 in self.classes2}

        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class k2: I( c1|k2 ; c2|k2)}
        self.mutual_info_4_5_cond_3 = {k2:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_3[k2][c11][c22],
            self.margin_4_cond_3[k2][c11],
            self.margin_5_cond_3[k2][c22]) for c11 in self.centers1) for c22 in self.centers2) for k2 in self.classes2}
    
        # Outputs the entropy of c1|k2
        # returns dict {class k2: H( c1|k2)}
        self.entropy_4_cond_3 = {k2:-sum(xlog2x(self.margin_4_cond_3[k2][c11]) for c11 in self.centers1) for k2 in self.classes2}
    
        # Outputs the entropy of c2|k2
        # returns dict {class k2: H( c2|k2)}
        self.entropy_5_cond_3 = {k2:-sum(xlog2x(self.margin_5_cond_3[k2][c22]) for c22 in self.centers2) for k2 in self.classes2}    
    
        # Outputs dict of freq (joint k2,c1,c2 | l,k1)
        self.result_cond_1_2 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_1_2[l][k1]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (k2 | l,k1)
        self.margin_3_cond_1_2 = {l:{k1:{k2: sum(sum(self.result_cond_1_2[l][k1][k2][c11][c22] for c22 in self.centers2) for c11 in self.centers1) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}    
    
        # Outputs dict of freq (c1,c2 | l,k1)
        self.margin_4_5_cond_1_2 = {l:{k1:{c1:{c2: sum(self.result_cond_1_2[l][k1][k22][c1][c2] for k22 in self.classes2) for c2 in self.centers2} for c1 in self.centers1} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (c1 | l,k1)
        self.margin_4_cond_1_2 = {l:{k1:{c1: sum(self.margin_4_5_cond_1_2[l][k1][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (c2 | l,k1)
        self.margin_5_cond_1_2 = {l:{k1:{c2: sum(self.margin_4_5_cond_1_2[l][k1][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {label l: {class k:I( c1|l,k1 ; c2|l,k1)}}
        self.mutual_info_4_5_cond_1_2 = {l:{k1:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_1_2[l][k1][c1][c2],
            self.margin_4_cond_1_2[l][k1][c1],
            self.margin_5_cond_1_2[l][k1][c2]) for c1 in self.centers1) for c2 in self.centers2) for k1 in self.classes1} for l in self.labels}
    
        # Outputs the entropy of c1|l,k1
        # returns dict {label l:{class k1: H( c1|l,k1)}}
        self.entropy_4_cond_1_2 = {l:{k1:-sum(xlog2x(self.margin_4_cond_1_2[l][k1][c1]) for c1 in self.centers1) for k1 in self.classes1} for l in self.labels}
    
        # Outputs the entropy of c2|k1,l
        # returns dict {label l:{class k1: H( c2|k1,l)}}
        self.entropy_5_cond_1_2 = {l:{k1:-sum(xlog2x(self.margin_5_cond_1_2[l][k1][c2]) for c2 in self.centers2) for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (joint k1,c1,c2 | l,k2)
        self.result_cond_1_3 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_1_3[l][k2]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (k1 | l,k2)
        self.margin_2_cond_1_3 = {l:{k1:{k2: sum(sum(self.result_cond_1_3[l][k1][k2][c11][c22] for c22 in self.centers2) for c11 in self.centers1) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}    
    
        # Outputs dict of freq (c1,c2 | l,k2)
        self.margin_4_5_cond_1_3 = {l:{k2:{c1:{c2: sum(self.result_cond_1_3[l][k11][k2][c1][c2] for k11 in self.classes1) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for l in self.labels}
    
        # Outputs dict of freq (c1 | l,k2)
        self.margin_4_cond_1_3 = {l:{k2:{c1: sum(self.margin_4_5_cond_1_3[l][k2][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k2 in self.classes2} for l in self.labels}
    
        # Outputs dict of freq (c2 | l,k2)
        self.margin_5_cond_1_3 = {l:{k2:{c2: sum(self.margin_4_5_cond_1_3[l][k2][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k2 in self.classes2} for l in self.labels}
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {label l: {class k:I( c1|l,k2 ; c2|l,k2)}}
        self.mutual_info_4_5_cond_1_3 = {l:{k2:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_1_3[l][k2][c1][c2],
            self.margin_4_cond_1_3[l][k2][c1],
            self.margin_5_cond_1_3[l][k2][c2]) for c1 in self.centers1) for c2 in self.centers2) for k2 in self.classes2} for l in self.labels}
    
        # Outputs the entropy of c1|k2,l
        # returns dict {label l:{class k2: H( c1|k2,l)}}
        self.entropy_4_cond_1_3 = {l:{k2:-sum(xlog2x(self.margin_4_cond_1_3[l][k2][c1]) for c1 in self.centers1) for k2 in self.classes2} for l in self.labels}
    
        # Outputs the entropy of c2|k2,l
        # returns dict {label l:{class k2: H( c2|k2,l)}}
        self.entropy_5_cond_1_3 = {l:{k2:-sum(xlog2x(self.margin_5_cond_1_3[l][k2][c2]) for c2 in self.centers2) for k2 in self.classes2} for l in self.labels}
        
        # Outputs dict of freq (joint c1,c2 | l,k1,k2)
        self.result_cond_1_2_3 = {l:{k1:{k2:{c1:{c2: zero_cond(self.joint[l][k1][k2][c1][c2],self.margin_1_2_3[l][k1][k2]) for c2 in self.centers2} for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (c1,c2 | l,k1,k2)
        self.margin_4_5_cond_1_2_3 = self.result_cond_1_2_3
    
        # Outputs dict of freq (c1 | l,k1,k2)
        self.margin_4_cond_1_2_3 = {l:{k1:{k2:{c1: sum(self.margin_4_5_cond_1_2_3[l][k1][k2][c1][c22] for c22 in self.centers2) for c1 in self.centers1} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs dict of freq (c2 | l,k1,k2)
        self.margin_5_cond_1_2_3 = {l:{k1:{k2:{c2: sum(self.margin_4_5_cond_1_2_3[l][k1][k2][c11][c2] for c11 in self.centers1) for c2 in self.centers2} for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {label l: {class k1:{class k2:I( c1|l,k1,k2 ; c2|l,k1,k2)}}}
        self.mutual_info_4_5_cond_1_2_3 = {l:{k1:{k2:sum(sum(xlog2x_yz(
            self.margin_4_5_cond_1_2_3[l][k1][k2][c1][c2],
            self.margin_4_cond_1_2_3[l][k1][k2][c1],
            self.margin_5_cond_1_2_3[l][k1][k2][c2]) for c1 in self.centers1) for c2 in self.centers2) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs the entropy of c1|k1,k22,l
        # returns dict {label l:{class k1: {class k2: H( c1|k1,k2,l)}}
        self.entropy_4_cond_1_2_3 = {l:{k1:{k2:-sum(xlog2x(self.margin_4_cond_1_2_3[l][k1][k2][c1]) for c1 in self.centers1) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
        # Outputs the entropy of c2|k1,k2,l
        # returns dict {label l:{class k1: {class k2: H( c2|k1,k2,l)}}
        self.entropy_5_cond_1_2_3 = {l:{k1:{k2:-sum(xlog2x(self.margin_5_cond_1_2_3[l][k1][k2][c2]) for c2 in self.centers2) for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
    
    def info_result_2_3(self):
        # Returns Information Theory details on classes1 and classes2
        # Returns dict{
            # 'mutual_info' float
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are floats
            # 'entropies': (float,float)
        mi = self.mutual_info_2_3
        h2 = self.entropy_2
        h3 = self.entropy_3
        return {'mutual_info': mi, 
                'mi_proportions': (mi/h2, mi/h3),
                'entropies': (h2,h3)}
    
    def info_result_4_5(self):
        # Returns Information Theory details on centers1 and centers2
        # Returns dict{
            # 'mutual_info' float
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are floats
            # 'entropies': (float,float)
        mi = self.mutual_info_4_5
        h4 = self.entropy_4
        h5 = self.entropy_5
        return {'mutual_info': mi, 
                'mi_proportions': (mi/h4, mi/h5),
                'entropies': (h4,h5)} 
    
    def info_result_2_3_cond_1(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropies': (dict{},dict{})
        mi = self.mutual_info_2_3_cond_1
        h4 = self.entropy_2_cond_1
        h5 = self.entropy_3_cond_1
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l: mi[l]/h4[l] for l in self.labels},
                    {l: mi[l]/h5[l] for l in self.labels}),
                'entropies': (h4,h5)}  
    
    def info_result_4_5_cond_1(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropies': (dict{},dict{})
        mi = self.mutual_info_4_5_cond_1
        h4 = self.entropy_4_cond_1
        h5 = self.entropy_5_cond_1
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l: mi[l]/h4[l] for l in self.labels},
                    {l: mi[l]/h5[l] for l in self.labels}),
                'entropies': (h4,h5)}
    
    def info_result_4_5_cond_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropies': (dict{},dict{})
        mi = self.mutual_info_4_5_cond_2
        h4 = self.entropy_4_cond_2
        h5 = self.entropy_5_cond_2
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {k1: mi[k1]/h4[k1] for k1 in self.classes1},
                    {k1: mi[k1]/h5[k1] for k1 in self.classes1}),
                'entropies': (h4,h5)}
    
    def info_result_4_5_cond_3(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropies': (dict{},dict{})
        mi = self.mutual_info_4_5_cond_3
        h4 = self.entropy_4_cond_3
        h5 = self.entropy_5_cond_3
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {k2: mi[k2]/h4[k2] for k2 in self.classes2},
                    {k2: mi[k2]/h5[k2] for k2 in self.classes2}),
                'entropies': (h4,h5)}

    def info_result_4_5_cond_1_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{
            # 'mutual_info' dict{dict{}}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{dict{}}
            # 'entropies': (dict{dict{}},dict{dict{}})
        mi = self.mutual_info_4_5_cond_1_2
        h4 = self.entropy_4_cond_1_2
        h5 = self.entropy_5_cond_1_2
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l:{k1: mi[l][k1]/h4[l][k1] for k1 in self.classes1} for l in self.labels},
                    {l:{k1: mi[l][k1]/h5[l][k1] for k1 in self.classes1} for l in self.labels}),
                'entropies': (h4,h5)}

    def info_result_4_5_cond_1_3(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{
            # 'mutual_info' dict{dict{}}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{dict{}}
            # 'entropies': (dict{dict{}},dict{dict{}})
        mi = self.mutual_info_4_5_cond_1_3
        h4 = self.entropy_4_cond_1_3
        h5 = self.entropy_5_cond_1_3
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l:{k2: mi[l][k2]/h4[l][k2] for k2 in self.classes2} for l in self.labels},
                    {l:{k2: mi[l][k2]/h5[l][k2] for k2 in self.classes2} for l in self.labels}),
                'entropies': (h4,h5)}
    
    def info_result_4_5_cond_1_2_3(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{
            # 'mutual_info' dict{dict{}}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{dict{}}
            # 'entropies': (dict{dict{}},dict{dict{}})
        mi = self.mutual_info_4_5_cond_1_2_3
        h4 = self.entropy_4_cond_1_2_3
        h5 = self.entropy_5_cond_1_2_3
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l:{k1:{k2: mi[l][k1][k2]/h4[l][k1][k2] for k2 in self.classes2} for k1 in self.classes1} for l in self.labels},
                    {l:{k1:{k2: mi[l][k1][k2]/h5[l][k1][k2] for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}),
                'entropies': (h4,h5)}    
    
    def reset_state(self):
        # To reset the state.
        for l in self.labels:
            for k1 in self.classes1:
                for k2 in self.classes2:
                    for c1 in self.centers1:
                        for c2 in self.centers2:
                            self.total[l][k1][k2][c1][c2] = 0.0
                            self.count = 0.0
