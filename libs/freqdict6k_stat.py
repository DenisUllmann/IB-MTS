# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:30:30 2022

@author: Denis
"""
### NOT FINISHED ###
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

class Frequency6kDict(object):
    """
    NOT FINISHED
    Creates a '6D' dict with 6 levels of keys:
        labels: first meta info on data (true label)
        classes: second meta info on data (input classification)
        classes: 3rd meta info on data (output classification)
        center: 4th meta info on data (eg. prior kmeans on data)
        center: 5thrd meta info on data (eg. input output kmeans on data)
        center: 6th meta info on data (eg. kmeans on data)
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

class NPFrequency4kDict(object):
    """
    Numpy version
    Creates a '6D' dict with 6 levels of keys:
        labels: first meta info on data (true label)
        classes: second meta info on data (input classification)
        classes: 3rd meta info on data (output classification)
        center: 4th meta info on data (eg. prior kmeans on data)
        center: 5th meta info on data (eg. input output kmeans on data)
        center: 6th meta info on data (eg. kmeans on data)
    The dict is made of np.float32 with corresponding keys
    
    update_state add the count to the corresponding variables:
        lab, cla1, cla2, res0, res1, res2: correspond to the previously described parameters
        they have the same shape, the values should correspond to the ones 
      given in __init__, and directly or indirectly (with some more processing
      like ) to some input/output of the model.
    
    result give the dict of joint frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, labels, classes1, classes2, 
                 centers0, centers1, centers2, name='centers_count'):
        self.name = name
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.centers0 = centers0
        self.centers1 = centers1
        self.centers2 = centers2
        self.total_prediction = {l:{k1:{k2:{c0:{c1:{c2: 0 for c2 in centers2} for c1 in centers1} for c0 in centers0} for k2 in classes2} for k1 in classes1} for l in labels}
        self.count_prior = 0
        self.count_prediction = 0
    
    # TODO amend the tf version
    def update_state(self, lab, cla1, cla2, res0, res1, res2, 
                     sample_weight=None):
        # Theoretical - not used for the  prediction of Mg data
        # Update count and total based on res1 and res2 given same lab and cla for both
        assert len(res1)==len(res2)==len(cla1)==len(cla2)==len(lab), "length of the two joint sequences should be equal for joint stats"
        self.count_prior += len(res0)
        self.count_prediction += len(res1)
        for (l,k1,k2,c0,c1,c2),v in zip(
                *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                    list(zip(lab,cla1, cla2, res0,res1,res2)),
                    return_counts=True, axis=0)]):
            self.total[l][k1][k2][c0][c1][c2] += int(float(v))
    
    def jointupdate_state(self, lab, cla, res, 
                          sample_weight=None):
        # Update count and total based on the joint data lab, cla, res (tuples of len 2)
        (cla1, cla2) = cla
        (res0, res1, res2) = res
        assert len(res1)==len(res2)==len(cla1)==len(cla2)==len(lab), "length of the two joint sequences should be equal for joint stats"
        self.count_prior += len(res0)
        self.count_prediction += len(res1)
        for l in self.labels:
            for (c1,c2,k1,k2),v in zip(
                    *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                        np.take(list(zip(*res)), np.where(np.logical_and(
                            np.logical_and(np.asarray(lab1, dtype=type(l))==l,
                                           np.asarray(lab2, dtype=type(l))==l),
                            np.logical_and(np.asarray(cla1, dtype=type(k))==k,
                                           np.asarray(cla2, dtype=type(k))==k))), axis=0)[0],
                        return_counts=True, axis=0)]):
                self.total[l][k][c1][c2] += int(float(v))
        for l in self.labels:
            for k in self.classes:
                for (c1,c2),v in zip(
                        *[[np.cast['str'](ee) for ee in e] for e in np.unique(
                            np.take(list(zip(*res)), np.where(np.logical_and(
                                np.logical_and(np.asarray(lab1, dtype=type(l))==l,
                                               np.asarray(lab2, dtype=type(l))==l),
                                np.logical_and(np.asarray(cla1, dtype=type(k))==k,
                                               np.asarray(cla2, dtype=type(k))==k))), axis=0)[0],
                            return_counts=True, axis=0)]):
                    self.total[l][k][c1][c2] += int(float(v))
    
    def count_assert(self):
        assert self.count==sum(sum(sum(sum(self.total[l][k][c1][c2] for c2 in self.centers2) for c1 in self.centers1) for k in self.classes) for l in self.labels), "counting error"
    
    def result(self):
        # Outputs dict of freq (joint l,k,c1,c2)
        self.count_assert()
        return {l:{k:{c1:{c2: self.total[l][k][c1][c2]/self.count for c2 in self.centers2} for c1 in self.centers1} for k in self.classes} for l in self.labels}
    
    def margin_3_4(self):
        # Outputs dict of freq (c1,c2)
        res =  self.result()
        return {c1:{c2: sum(sum(res[l][k][c1][c2] for l in self.labels) for k in self.classes) for c2 in self.centers2} for c1 in self.centers1}
    
    def margin_3(self):
        # Outputs dict of freq (c1)
        res =  self.result()
        return {c1: sum(sum(sum(res[l][k][c1][c2] for c2 in self.centers2) for k in self.classes) for l in self.labels) for c1 in self.centers1}
    
    def margin_4(self):
        # Outputs dict of freq (c2)
        res =  self.result()
        return {c2: sum(sum(sum(res[l][k][c1][c2] for c1 in self.centers1) for k in self.classes) for l in self.labels) for c2 in self.centers2}
    
    def mutual_info_3_4(self):
        # Outputs the mutual information between the last 2 keys of the dict
        # returns I( c1 ; c2)
        p_c1c2 = self.margin_3_4()
        return sum(sum(p_c1c2[c1][c2]*np.log2(
                p_c1c2[c1][c2]/self.margin_3()[c1]/self.margin_4()[c2]) for c1 in self.centers1) for c2 in self.centers2)
    
    def entropy_3(self):
        # Outputs the entropy of c1
        # returns H( c1)
        p_c = self.margin_3()
        return sum(p_c[c1]*np.log2(p_c[c1]) for c1 in self.centers1)
    
    def entropy_4(self):
        # Outputs the entropy of c2
        # returns H( c2)
        p_c = self.margin_4()
        return sum(p_c[c2]*np.log2(p_c[c2]) for c2 in self.centers2)
    
    def result_cond_1(self):
        # Outputs dict of freq (joint k,c1,c2 | l)
        self.count_assert()
        return {l:{k:{c1:{c2: self.total[l][k][c1][c2]/sum(sum(sum(self.total[l][kk][c11][c22] for c22 in self.centers2) for c11 in self.centers1) for kk in self.classes) for c2 in self.centers2} for c1 in self.centers1} for k in self.classes} for l in self.labels}
    
    def margin_3_4_cond_1(self):
        # Outputs dict of freq (c1,c2 | l)
        res =  self.result_cond_1()
        return {l:{c1:{c2: sum(res[l][k][c1][c2] for k in self.classes) for c2 in self.centers2} for c1 in self.centers1} for l in self.labels}
    
    def margin_3_cond_1(self):
        # Outputs dict of freq (c1 | l)
        res =  self.margin_3_4_cond_1()
        return {l:{c1: sum(res[l][c1][c2] for c2 in self.centers2) for c1 in self.centers1} for l in self.labels}
    
    def margin_4_cond_1(self):
        # Outputs dict of freq (c2 | l)
        res =  self.margin_3_4_cond_1()
        return {l:{c2: sum(res[l][c1][c2] for c1 in self.centers1) for c2 in self.centers2} for l in self.labels}
    
    def mutual_info_3_4_cond_1(self):
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class k: I( c1|k ; c2|k)}}
        p_c1c2_cond_k = self.result_cond_1()
        return {k:sum(sum(p_c1c2_cond_k[k][c1][c2]*np.log2(
                p_c1c2_cond_k[k][c1][c2]/self.margin_3_cond_1()[k][c1]/self.margin_4_cond_1()[k][c2]) for c1 in self.centers1) for c2 in self.centers2) for k in self.classes}
    
    def entropy_3_cond_1(self):
        # Outputs the entropy of c1|k
        # returns dict {class k: H( c1|k)}
        p_c_cond_k = self.margin_3_cond_1()
        return {k:sum(p_c_cond_k[k][c1]*np.log2(p_c_cond_k[k][c1]) for c1 in self.centers1) for k in self.classes}
    
    def entropy_4_cond_1(self):
        # Outputs the entropy of c2|k
        # returns dict {class k: H( c2|k)}
        p_c_cond_k = self.margin_4_cond_1()
        return {k:sum(p_c_cond_k[k][c1]*np.log2(p_c_cond_k[k][c1]) for c1 in self.centers1) for k in self.classes}
    
    def result_cond_2(self):
        # Outputs dict of freq (joint l,c1,c2 | k)
        self.count_assert()
        return {l:{k:{c1:{c2: self.total[l][k][c1][c2]/sum(sum(sum(self.total[ll][k][c11][c22] for c22 in self.centers2) for c11 in self.centers1) for ll in self.labels) for c2 in self.centers2} for c1 in self.centers1} for k in self.classes} for l in self.labels}

    def margin_3_4_cond_2(self):
        # Outputs dict of freq (c1,c2 | k)
        res =  self.result_cond_2()
        return {k:{c1:{c2: sum(res[l][k][c1][c2] for l in self.labels) for c2 in self.centers2} for c1 in self.centers1} for k in self.classes}
    
    def margin_3_cond_2(self):
        # Outputs dict of freq (c1 | k)
        res =  self.margin_3_4_cond_2()
        return {k:{c1: sum(res[k][c1][c2] for c2 in self.centers2) for c1 in self.centers1} for k in self.classes}
    
    def margin_4_cond_2(self):
        # Outputs dict of freq (c2 | k)
        res =  self.margin_3_4_cond_2()
        return {k:{c2: sum(res[k][c1][c2] for c1 in self.centers1) for c2 in self.centers2} for k in self.classes}

    def mutual_info_3_4_cond_2(self):
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {label l: I( c1|l ; c2|l)}
        p_c1c2_cond_l = self.result_cond_2()
        return {l:sum(sum(p_c1c2_cond_l[l][c1][c2]*np.log2(
                p_c1c2_cond_l[l][c1][c2]/self.margin_3_cond_2()[l][c1]/self.margin_4_cond_2()[l][c2]) for c1 in self.centers1) for c2 in self.centers2) for l in self.labels}
    
    def entropy_3_cond_2(self):
        # Outputs the entropy of c1|l
        # returns dict {label k: H( c1|l)}
        p_c_cond_l = self.margin_3_cond_2()
        return {l:sum(p_c_cond_l[l][c1]*np.log2(p_c_cond_l[l][c1]) for c1 in self.centers1) for l in self.labels}
    
    def entropy_4_cond_2(self):
        # Outputs the entropy of c2|l
        # returns dict {label k: H( c2|l)}
        p_c_cond_l = self.margin_4_cond_2()
        return {l:sum(p_c_cond_l[l][c2]*np.log2(p_c_cond_l[l][c2]) for c2 in self.centers2) for l in self.labels}
    
    def result_cond_1_2(self):
        # Outputs dict of freq (joint c1,c2 | l,k)
        self.count_assert()
        return {l:{k:{c1:{c2: self.total[l][k][c1][c2]/sum(sum(self.total[l][k][c11][c22] for c22 in self.centers2) for c11 in self.centers1) for c2 in self.centers2} for c1 in self.centers1} for k in self.classes} for l in self.labels}
    
    def margin_3_cond_1_2(self):
        # Outputs dict of freq (c1 | l,k)
        res =  self.result_cond_1_2()
        return {l:{k:{c1: sum(res[l][k][c1][c2] for c2 in self.centers2) for c1 in self.centers1} for k in self.classes} for l in self.labels}
    
    def margin_4_cond_1_2(self):
        # Outputs dict of freq (c2 | l,k)
        res =  self.result_cond_1_2()
        return {l:{k:{c2: sum(res[l][k][c1][c2] for c1 in self.centers1) for c2 in self.centers2} for k in self.classes} for l in self.labels}
    
    def mutual_info_3_4_cond_1_2(self):
        # Outputs the mutual information between the last 2 keys of the dict
        # returns dict {class k:{label l: I( c1|k,l ; c2|k,l)}}
        p_c1c2_cond_kl = self.result_cond_1_2()
        return {k:{l:sum(sum(p_c1c2_cond_kl[k][l][c1][c2]*np.log2(
                p_c1c2_cond_kl[k][l][c1][c2]/self.margin_3_cond_1_2()[k][l][c1]/self.margin_4_cond_1_2()[k][l][c2]) for c1 in self.centers1) for c2 in self.centers2) for l in self.labels} for k in self.classes}
    
    def entropy_3_cond_1_2(self):
        # Outputs the entropy of c1|k,l
        # returns dict {class k:{label l: H( c1|k,l)}}
        p_c_cond_kl = self.margin_3_cond_1_2()
        return {k:{l:sum(p_c_cond_kl[k][l][c1]*np.log2(p_c_cond_kl[k][l][c1]) for c1 in self.centers1) for l in self.labels} for k in self.classes}
    
    def entropy_4_cond_1_2(self):
        # Outputs the entropy of c2|k,l
        # returns dict {class k:{label l: H( c2|k,l)}}
        p_c_cond_kl = self.margin_4_cond_1_2()
        return {k:{l:sum(p_c_cond_kl[k][l][c1]*np.log2(p_c_cond_kl[k][l][c1]) for c1 in self.centers1) for l in self.labels} for k in self.classes}

    def info_result_3_4_cond_1_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes and labels
        # Returns dict{
            # 'mutual_info' dict{dict{}}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{dict{}}
            # 'entropy1': dict{dict{}}
            # 'entropy2: dict{dict{}}
        mi = self.mutual_info_3_4_cond_1_2()
        h3 = self.entropy_3_cond_1_2()
        h4 = self.entropy_4_cond_1_2()
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {k:{l: mi[k][l]/h3[k][l] for l in self.labels} for k in self.classes},
                    {k:{l: mi[k][l]/h4[k][l] for l in self.labels} for k in self.classes}),
                'entropy1': h3,
                'entropy2': h4}
    
    def info_result_3_4_cond_1(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by classes
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropy1': dict{}
            # 'entropy2: dict{}
        mi = self.mutual_info_3_4_cond_1()
        h3 = self.entropy_3_cond_1()
        h4 = self.entropy_4_cond_1()
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {k: mi[k]/h3[k] for k in self.classes},
                    {k: mi[k]/h4[k] for k in self.classes}),
                'entropy1': h3,
                'entropy2': h4}
    
    def info_result_3_4_cond_2(self):
        # Returns Information Theory details on centers1 and centers2, conditioned by labels
        # Returns dict{
            # 'mutual_info' dict{}
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are dict{}
            # 'entropy1': dict{}
            # 'entropy2: dict{}
        mi = self.mutual_info_3_4_cond_2()
        h3 = self.entropy_3_cond_2()
        h4 = self.entropy_4_cond_2()
        return {'mutual_info': mi, 
                'mi_proportions': (
                    {l: mi[l]/h3[l] for l in self.labels},
                    {l: mi[l]/h4[l] for l in self.labels}),
                'entropy1': h3,
                'entropy2': h4}
    
    def info_result_3_4(self):
        # Returns Information Theory details on centers1 and centers2
        # Returns dict{
            # 'mutual_info' float
            # 'mi_proportions': (mi/entropy1, mi/entropy2) are floats
            # 'entropy1': float
            # 'entropy2: float
        mi = self.mutual_info_3_4()
        h3 = self.entropy_3()
        h4 = self.entropy_4()
        return {'mutual_info': mi, 
                'mi_proportions': (mi/h3, mi/h4),
                'entropy1': h3,
                'entropy2': h4}
    
    def reset_state(self):
        # To reset the state.
        for l in self.labels:
            for k in self.classes:
                for c1 in self.centers1:
                    for c2 in self.centers2:
                        self.total[l][k][c1][c2] = 0.0
                        self.count = 0.0
