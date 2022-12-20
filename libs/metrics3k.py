# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:19:50 2022

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

class NPMtsMetric3kDict(object):
    """
    Numpy version
    Creates a '3D' dict with 3 levels of keys:
        labels: first meta info on data (true label)
        classes: 2nd meta info on joint data (eg. class on input data)
        classes: 3rd meta info on joint data (eg. class on output data)
    The dict is made of a dict {'info_measure':np.float32} with corresponding keys
    'info_measure' can be KL(c0||c1), KL(c0||c2), I(c1||c2), H(c0/c1/c2)
    
    Also creates a dict of batches 'last_info' with the above 'info_measure's
    
    update_state add the count to the corresponding variables
    
    result give the dict of frequencies
    
    reset_state for reset all counts to 0.
    """
    def __init__(self, metric_cls, labels, classes1, classes2, 
                 name = 'info_count'):
        self.name = name
        self.metric_cls = metric_cls
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.reset_state()
    
    def update_state(self, keys, res, sample_weight=None):
        # creates info data 'last_info' for each sample of the batch 
        # and update the dicts of info 'count' and 'total'
        # keys is (label, clas1, clas2)
        # res is (res1, res2)
        # all are batched data (tuples of batched info)
        assert all(all(len(res[j])==len(keys[i]) for j in range(2)) for i in range(3)), "all batches should have the same length"
        assert all(res[0][i].shape==res[1][i].shape for i in range(len(res[0]))), "results should share the same shape to compare them"
        
        for lab in self.labels:
            for cla1 in self.classes1:
                for cla2 in self.classes2:
                    resi = tuple(resk[
                        np.logical_and(
                            np.asarray(keys[0], type(lab))==lab,
                            np.logical_and(
                                np.asarray(keys[1], type(cla1))==cla1,
                                np.asarray(keys[2], type(cla2))==cla2))
                        ] for resk in res)
                    self.global_metric[lab][cla1][cla2].update_state(resi)
                    self.update_or_create_state(
                        self.time_metrics[lab][cla1][cla2], 
                        self.update_time_step,
                        resi,
                        resi[0].shape[1])
    
    def update_time_step(self, list_metric, res):
        assert res[0].shape==res[1].shape, "the 2 results should have the same shapes to be compared"
        for time in range(res[0].shape[1]):
            list_metric[time].update_state(
                tuple(resk[:,time] for resk in res))
    
    def update_or_create_state(self, list_metric, update_fn, res, times):
        # res is 2-tuple( [batch,time,...] , ..)
        # list_metric is a list, eventually empty, of metrics for each time step
        list_metric.extend(
            [self.initialized_metric() for i_ext in range(
                times-len(list_metric))])
        update_fn(list_metric, res)
    
    def initialized_metric(self):
        metric = self.metric_cls()
        metric.reset_state()
        return metric
    
    def result(self):
        return {lab:{cla1:{cla2:(
            self.global_metric[lab][cla1][cla2].count,
            self.global_metric[lab][cla1][cla2].result(),
            [metric.result() for metric in self.time_metrics[lab][cla1][cla2]]
            ) for cla2 in self.classes2
                           } for cla1 in self.classes1
                     } for lab in self.labels
                }
    
    def result_by_1_2_3(self):
        return self.result()
        
    def add_time_count(self, list_metric, res):
        for time in range(len(res)):
            list_metric[time].count += res[time].count
            list_metric[time].total += res[time].total
    
    def result_by_1_2(self):
        # {lab:{cla1:(global_metric,list[time_metric])}}
        global_metric = {
            lab:{
                cla1:self.initialized_metric() for cla1 in self.classes1
                } for lab in self.labels
            }
        
        time_metrics = {
            lab:{
                cla1:[] for cla1 in self.classes1
                } for lab in self.labels
            }
        
        for lab in self.labels:
            for cla1 in self.classes1:
                for cla2 in self.classes2:
                    global_metric[lab][cla1].count += self.global_metric[lab][cla1][cla2].count
                    global_metric[lab][cla1].total += self.global_metric[lab][cla1][cla2].total
                    self.update_or_create_state(
                        time_metrics[lab][cla1], 
                        self.add_time_count,
                        self.time_metrics[lab][cla1][cla2],
                        len(self.time_metrics[lab][cla1][cla2]))
        
        return {
            lab:{
                cla1:(
                    global_metric[lab][cla1].count,
                    global_metric[lab][cla1].result(),
                    [metric.result() for metric in time_metrics[lab][cla1]]
                    ) for cla1 in self.classes1
                } for lab in self.labels
            }
    
    def result_by_1_3(self):
        # {lab:{cla2:(global_metric,list[time_metric])}}
        global_metric = {
            lab:{
                cla2:self.initialized_metric() for cla2 in self.classes2
                } for lab in self.labels
            }
        
        time_metrics = {
            lab:{
                cla2:[] for cla2 in self.classes2
                } for lab in self.labels
            }
        
        for lab in self.labels:
            for cla2 in self.classes2:
                for cla1 in self.classes1:
                    global_metric[lab][cla2].count += self.global_metric[lab][cla1][cla2].count
                    global_metric[lab][cla2].total += self.global_metric[lab][cla1][cla2].total
                    self.update_or_create_state(
                        time_metrics[lab][cla2], 
                        self.add_time_count,
                        self.time_metrics[lab][cla1][cla2],
                        len(self.time_metrics[lab][cla1][cla2]))
        
        return {
            lab:{
                cla2:(
                    global_metric[lab][cla2].count,
                    global_metric[lab][cla2].result(),
                    [metric.result() for metric in time_metrics[lab][cla2]]
                    ) for cla2 in self.classes2
                } for lab in self.labels
            }
    
    def result_by_1(self):
        # {lab:(global_metric,list[time_metric])}
        global_metric = {
            lab: self.initialized_metric() for lab in self.labels
            }
        
        time_metrics = {
            lab: [] for lab in self.labels
            }
        
        for lab in self.labels:
            for cla1 in self.classes1:
                for cla2 in self.classes2:
                    global_metric[lab].count += self.global_metric[lab][cla1][cla2].count
                    global_metric[lab].total += self.global_metric[lab][cla1][cla2].total
                    self.update_or_create_state(
                        time_metrics[lab], 
                        self.add_time_count,
                        self.time_metrics[lab][cla1][cla2],
                        len(self.time_metrics[lab][cla1][cla2]))
        
        return {
            lab:(
                global_metric[lab].count,
                global_metric[lab].result(),
                [metric.result() for metric in time_metrics[lab]]
                ) for lab in self.labels
            }
    
    def result_by_no(self):
        # (global_metric,list[time_metric])
        global_metric = self.initialized_metric()
        
        time_metrics = []
        
        for lab in self.labels:
            for cla1 in self.classes1:
                for cla2 in self.classes2:
                    global_metric.count += self.global_metric[lab][cla1][cla2].count
                    global_metric.total += self.global_metric[lab][cla1][cla2].total
                    self.update_or_create_state(
                        time_metrics, 
                        self.add_time_count,
                        self.time_metrics[lab][cla1][cla2],
                        len(self.time_metrics[lab][cla1][cla2]))
        
        return (
            global_metric.count,
            global_metric.result(),
            [metric.result() for metric in time_metrics]
            )
    
    def return_save(self):
        return {
            'global': {
                'count': {lab: {cla1: {cla2: 
        self.global_metric[lab][cla1][cla2].count for cla2 in self.classes2
                                       } for cla1 in self.classes1
                                } for lab in self.labels},
                'total': {lab: {cla1: {cla2: 
        self.global_metric[lab][cla1][cla2].total for cla2 in self.classes2
                                       } for cla1 in self.classes1
                                } for lab in self.labels}},
            'time': {
                'count': {lab: {cla1: {cla2: 
        [metric.count for metric in self.time_metrics[lab][cla1][cla2]] for cla2 in self.classes2
                                       } for cla1 in self.classes1
                                } for lab in self.labels
                           },
                'total': {lab: {cla1: {cla2: 
        [metric.total for metric in self.time_metrics[lab][cla1][cla2]] for cla2 in self.classes2
                                       } for cla1 in self.classes1
                                } for lab in self.labels
                           }
            }}
    
    def from_saved(self, saved):
        for lab in self.labels:
            for cla1 in self.classes1:
                for cla2 in self.classes2:
                    self.global_metric[
                        lab][cla1][cla2].count = saved['global']['count'][
                            lab][cla1][cla2]
                    self.global_metric[
                        lab][cla1][cla2].total = saved['global']['total'][
                            lab][cla1][cla2]
                    self.time_metrics[lab][cla1][cla2].extend(
                        [self.initialized_metric() for i_ext in range(
                            len(
                                saved['time']['count'][lab][cla1][cla2])-len(
                                    self.time_metrics[lab][cla1][cla2]))])
                    for tis, tvs in enumerate(saved['time']['count'][lab][cla1][cla2]):
                        self.time_metrics[lab][cla1][cla2][tis].count = tvs
                    for tis, tvs in enumerate(saved['time']['total'][lab][cla1][cla2]):
                        self.time_metrics[lab][cla1][cla2][tis].total = tvs
    
    def reset_state(self):
        # To reset the state.
        self.global_metric = {
            lab:{
                cla1:{
                    cla2:self.initialized_metric() for cla2 in self.classes2
                    } for cla1 in self.classes1
                } for lab in self.labels
            }
        
        self.time_metrics = {
            lab:{
                cla1:{
                    cla2:[] for cla2 in self.classes2
                    } for cla1 in self.classes1
                } for lab in self.labels
            }