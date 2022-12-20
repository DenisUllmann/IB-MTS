# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:45:44 2022

@author: Denis
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from libs.metrics3k import NPMtsMetric3kDict

class NPmetric(object):
    def __init__(self, metric_fn, result_fn=lambda x:x, zero_divide=1e-7):
        self.count = 0.
        self.total = 0
        self.zero_divide = zero_divide
        self.metric_fn = metric_fn
        self.result_fn = result_fn
    
    def update_state(self, res_1_2):
        assert res_1_2[0].shape == res_1_2[1].shape, "the inputs for the metric function should have the same shape"
        self.total += self.metric_fn(*res_1_2)
        self.count += res_1_2[0].size
    
    def reset_state(self):
        self.count = 0.
        self.total = 0
    
    def result(self):
        return self.result_fn(
            np.divide(self.total,
                      np.clip(self.count, self.zero_divide,None)))

class NPMeanAbsoluteError(NPmetric):
    def __init__(self, dtype=np.float32, zero_divide=1e-7):
        super(NPMeanAbsoluteError, self).__init__(
            lambda x,y: np.sum(np.abs(
                np.asarray(x, dtype)-np.asarray(y, dtype))),
            zero_divide=zero_divide)

class NPRootMeanSquaredError(NPmetric):
    def __init__(self, dtype=np.float32, zero_divide=1e-7):
        super(NPRootMeanSquaredError, self).__init__(
            lambda x,y: np.sum((
                np.asarray(x, dtype)-np.asarray(y, dtype))**2),
            np.sqrt,
            zero_divide=zero_divide)

class NPMeanAbsolutePercentageError(NPmetric):
    def __init__(self, dtype=np.float32, zero_divide=1e-7, tolerance=0.01):
        super(NPMeanAbsolutePercentageError, self).__init__(
            lambda x,y: np.sum(100*np.abs(np.divide(
                (np.asarray(x, dtype)-np.asarray(y, dtype))*np.cast[x.dtype](np.asarray(x)-np.asarray(y)>tolerance),
                np.clip(x,zero_divide,None)))*np.cast[x.dtype](x!=0)),
            np.sqrt,
            zero_divide=zero_divide)

class NPMtsMetrics(object):
    """
    Numpy version
    mts = multi time series
    Creates a '3D' NPFrequency3kDict dict with 3 levels of keys:
        labels (str): first meta info on data (true label)
        classes (str): second meta info on data (in classification)
        classes (str): 3rd meta info on data (eg. out classification)
    self.fit_batch(lab, cla1, cla2, img_batch1, img_batch2) updates the Frequency3kDict stats with the 
    batched data given in the 'lab, clas and img_batches':
        lab : batch of labels corresponding to 'labels'
        cla1 : batch of predicted classes corresponding to input 'classes'
        cla2 : batch of predicted classes corresponding to output 'classes'
        img_batch1, img_batch2 : batch of input and output images used for mts metrics'
    self.result() is to output the mean results
    """
    def __init__(self, labels, classes1, classes2, name = 'mts_metrics'):
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.metrics = {
            'mae': NPMeanAbsoluteError,
            'rmse': NPRootMeanSquaredError,
            'mape': NPMeanAbsolutePercentageError}
        self.metrics_dict = {
            metric_n: NPMtsMetric3kDict(
                metric_cls, labels, 
                classes1, classes2, 
                name = name) for metric_n, metric_cls in self.metrics.items()}
    
    def update(self, keys, res):
        # keys is (label, clas1, clas2)
        # res is (res1, res2)
        # all are batched data
        for metric_cnt in self.metrics_dict.values():
            metric_cnt.update_state(keys, res)
    
    def result(self):
        # {metric:{lab:{cla1:{cla2:(global_count, global_metric,list[time_metric])}}}}
        return {metric_n: metric_cnt.result(
            ) for metric_n, metric_cnt in self.metrics_dict.items()}
    
    def result_by_1_2_3(self):
        # {metric:{lab:{cla1:{cla2:(global_count, global_metric,list[time_metric])}}}}
        return self.result()
    
    def result_by_1_2(self):
        # {metric:{lab:{cla1:(global_count, global_metric,list[time_metric])}}}
        return {metric_n: metric_cnt.result_by_1_2(
            ) for metric_n, metric_cnt in self.metrics_dict.items()}
    
    def result_by_1_3(self):
        # {metric:{lab:{cla2:(global_count, global_metric,list[time_metric])}}}
        return {metric_n: metric_cnt.result_by_1_3(
            ) for metric_n, metric_cnt in self.metrics_dict.items()}
    
    def result_by_1(self):
        # {metric:{lab:(global_count, global_metric,list[time_metric])}}
        return {metric_n: metric_cnt.result_by_1(
            ) for metric_n, metric_cnt in self.metrics_dict.items()}
    
    def result_by_no(self):
        # {metric:(global_count, global_metric,list[time_metric])}
        return {metric_n: metric_cnt.result_by_no(
            ) for metric_n, metric_cnt in self.metrics_dict.items()}
        
    def reset(self):
        for metric_cnt in self.metrics_dict.values():
            metric_cnt.reset_state()