# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:12:47 2022

@author: Denis
"""

import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '..')
from dataset.data_process import to_kcentroid_seq
from libs.freqdict3k_stat import Frequency3kDict, NPFrequency3kDict
from libs.freqdict2k_stat import NPFrequency2_lc_Dict
from libs.freqdict4k_stat import Frequency4kDict
from libs.freqdict5k_stat import NPFrequency5_lkkcc_Dict
from libs.info3k import NPInfo3_lkk_Dict

class CenterStat(object):
    """
    NOT FINISHED
    Creates a '3D' Frequency3kDict dict with 3 levels of keys:
        labels (str): first meta info on data (true label)
        classes (str): second meta info on data (classification)
        centers (str): 3rd meta info on data (eg. kmeans on data)
    self.fit_batch(lab, cla, img_batch) updates the Frequency3kDict stats with the 
    batched data given in the 'lab, cla and img_batch':
        lab : batch of labels corresponding to 'labels'
        cla : batch of predicted classes corresponding to 'classes'
        img_batch : batch of images used for stats on 'centers'
    self.result() is to output the freq results
    """
    def __init__(self, labels, classes, centers, name = 'centers_count'):
        self.labels = labels
        self.classes = classes
        self.centers = centers
        self.freqdict = Frequency3kDict(labels, classes, centers, name = name)
    
    def update(self, lab, cla, res):
        self.freqdict.update_state(lab, cla, res)
    
    def result(self):
        return self.freqdict.result()
    
    def reset(self):
        self.freqdict.reset_state()
            
    def fit_batch(self, lab, cla, img_batch):
        # batch dims: [batch, time, lambda, 1]
        # TODO lab, cla may be tensor
        lab = np.hstack([[l]*img_batch[i].shape[0] for i,l in enumerate(lab)])
        cla = np.hstack([[c]*img_batch[i].shape[0] for i,c in enumerate(cla)])
        res = np.hstack(
            [to_kcentroid_seq(
                tf.squeeze(img_batch[i]), k=1)[1][:,0] for i in range(img_batch.shape[0])])
        self.update(lab, cla, res)

class JointCenterStat(object):
    """
    NOT FINISHED
    Creates a '4D' Frequency4kDict dict with 4 levels of keys:
        labels (str): first meta info on data (true label)
        classes (str): second meta info on data (classification)
        centers (str): 3rd meta info on data (eg. kmeans on data)
        centers (str): 2nd level for the joint statistics on data
    self.fit_batch(lab, cla, img_batch1, img_batch2) updates the Frequency4kDict stats with the 
    batched data given in the 'lab, cla and img_batch':
        lab : batch of labels corresponding to 'labels'
        cla : batch of predicted classes corresponding to 'classes'
        img_batch : batch of images used for stats on 'centers'
    self.result() is to output the freq results
    """
    def __init__(self, labels, classes, centers, name = 'centers_count'):
        self.labels = labels
        self.classes = classes
        self.centers = centers
        self.freqdict = Frequency4kDict(labels, classes, centers, centers, name = name)
    
    def update(self, lab, cla, res):
        self.freqdict.update_state(lab, cla, res)
    
    def result(self):
        return self.freqdict.result()
    
    def reset(self):
        self.freqdict.reset_state()
            
    def fit_batch(self, lab, cla, img_batch):
        # batch dims: [batch, time, lambda, 1]
        # TODO lab, cla may be tensor
        lab = np.hstack([[l]*img_batch[i].shape[0] for i,l in enumerate(lab)])
        cla = np.hstack([[c]*img_batch[i].shape[0] for i,c in enumerate(cla)])
        res = np.hstack(
            [to_kcentroid_seq(
                tf.squeeze(img_batch[i]), k=1)[1][:,0] for i in range(img_batch.shape[0])])
        self.update(lab, cla, res)

class NPCenterStat(object):
    """
    Numpy version
    Creates a '3D' NPFrequency3kDict dict with 3 levels of keys:
        labels (str): first meta info on data (true label)
        classes (str): second meta info on data (classification)
        centers (str): 3rd meta info on data (eg. kmeans on data)
    self.fit_batch(lab, cla, img_batch) updates the Frequency3kDict stats with the 
    batched data given in the 'lab, cla and img_batch':
        lab : batch of labels corresponding to 'labels'
        cla : batch of predicted classes corresponding to 'classes'
        img_batch : batch of images used for stats on 'centers'
    self.result() is to output the freq results
    """
    def __init__(self, labels, classes, centers, name = 'centers_count'):
        self.labels = labels
        self.classes = classes
        self.centers = centers
        self.freqdict = NPFrequency3kDict(labels, classes, centers, name = name)
    
    def update(self, lab, cla, res):
        self.freqdict.update_state(lab, cla, res)
    
    def result(self):
        # dict p(k,l,c)
        return self.freqdict.result()
    
    def result_cond_1_2(self):
        # tuple of dict:
        # (p(c|k,l), 
        #  {'entropy':H(c|k,l)})
        return (self.freqdict.result_cond_1_2(),
                self.freqdict.info_result_3_cond_1_2())
    
    def result_cond_1(self):
        # tuple of dict:
        # (p(c|k), 
        #  {'entropy':H(c|k)})
        return (self.freqdict.result_cond_1(),
                self.freqdict.info_result_3_cond_1())
    
    def result_cond_2(self):
        # tuple of dict:
        # (p(c|l), 
        #  {'entropy':H(c|l)})
        return (self.freqdict.result_cond_2(),
                self.freqdict.info_result_3_cond_2())
    
    def result_cond_no(self):
        # tuple of dict:
        # (p(c), 
        #  {'entropy':H(c)})
        return (self.freqdict.margin_3(),
                self.freqdict.info_result_3())
        
    def reset(self):
        self.freqdict.reset_state()
    
    def fit_batch(self, lab, cla, img_batch):
        # batch dims: [batch, time, lambda, 1]
        lab = np.hstack([[l]*img_batch[i].shape[0] for i,l in enumerate(lab)])
        cla = np.hstack([[c]*img_batch[i].shape[0] for i,c in enumerate(cla)])
        res = np.hstack(
            [to_kcentroid_seq(
                np.squeeze(img_batch[i]), k=1)[1][:,0] for i in range(img_batch.shape[0])])
        self.update(lab, cla, res)

class NPJointCenterStat(object):
    """
    Numpy version
    Creates a '6D' NPFrequency4kDict + NPFrequency5kDict dict with 4 levels of keys:
        labels (str): first meta info on data (true label)
        classes (str): second meta info on data (classification)
        classes (str): second meta info on data (classification)
        centers (str): 3rd prior meta info on data (eg. kmeans on data)
        centers (str): 2nd level for the joint input-statistics on data
        centers (str): 3rd level for the joint output-statistics on data
    self.fit_batch(lab, cla, img_batch) updates the FrequencyDicts stats with the 
    batched data given in the 'lab, cla and img_batch1, img_batch2':
        lab : batch of labels corresponding to 'labels'
        cla : tuple of batch of predicted classes (input, output) corresponding to 'classes'
        img_batch : tuple of batch of images used for marginal-(prio,input,output) stats on 'centers'
    self.result() is to output the freq results
    """
    def __init__(self, labels, classes1, classes2, centers0, centers1, centers2, name = 'centers_count'):
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.centers0 = centers0
        self.centers1 = centers1
        self.centers2 = centers2
        self.freqdict_prior = NPFrequency2_lc_Dict(labels, centers0, 
                                                  name = name+'_prior')
        self.freqdict_io = NPFrequency5_lkkcc_Dict(labels, classes1, classes2, 
                                                  centers1, centers2, name=name+'_inout')
        assert set(centers1)==set(centers2), "centers1 and 2 should be the same for KL and I info"
        self.batchinfo_pio = NPInfo3_lkk_Dict(labels, classes1, classes2, 
                                              (centers0,centers1,centers2),
                                              name=name+'_info_pio')
    
    def update(self, lab, cla, res):
        # lab is 2-tuple (lab_prior, lab_pred)
        # cla is 2-tuple (in, out)
        # res is 3-tuple (prior, in, out)
        self.freqdict_prior.update_state(lab[0], res[0])
        self.freqdict_io.update_state(lab[1], cla, (res[1], res[2]))
    
    def batch_info_update(self, lab, cla, res):
        # creates info data 'last_info' for each sample of the batch 
        # and update the dict of info 'count' and total
        # lab is batch 
        # cla is 2-tuple of batch (in, out)
        # res is 3-tuple of batch center assignements (prior, in, out)
        self.batchinfo_pio.update_state(lab, cla, res)
    
    def result(self):
        # TODO amend to 3 outputs when result is used
        # 3 dict: p(l,k1,k2,[c0/OR/c1,c2])
        return (self.freqdict_prior.result(), 
                self.freqdict_io.result(), 
                self.batchinfo_pio.result())
    
    def kl_div(self, p,q):
        return sum([pp*np.log2(np.divide(pp,qq)),0][int(pp==0 or qq==0)] for pp,qq in zip(p,q))
    
    def entropy(self, p):
        xlog2x = lambda x:x*np.log2([x,1][int(x==0)])
        return -sum(xlog2x(pp) for pp in p)
    
    def predictinfo_centers_cond_no(self, pc0, pc1):
        assert pc0.keys()==pc1.keys()
        pc0 = [pc0[c] for c in pc0.keys()]
        pc1 = [pc1[c] for c in pc1.keys()]
        kl_div = self.kl_div(pc0, pc1)
        entropies = (self.entropy(pc0), self.entropy(pc1))
        return {'kl-div':kl_div,
                'kl_proportion': np.divide(kl_div,entropies[0]),
                'entropies':entropies}
    
    def predictinfo_centers_cond_1(self, pc0, pc1):
        assert pc0.keys()==pc1.keys()
        keys = pc0.keys()
        assert all([pc0[l].keys()==pc1[l].keys() for l in keys])
        pc0 = {l: [pc0[l][c] for c in pc0[l].keys()] for l in keys}
        pc1 = {l: [pc1[l][c] for c in pc1[l].keys()] for l in keys}
        kl_div = {l: self.kl_div(pc0[l], pc1[l]) for l in keys}
        entropies = ({l: self.entropy(pc0[l]) for l in keys}, 
                     {l: self.entropy(pc1[l]) for l in keys})
        return {'kl-div':kl_div,
                'kl_proportion':{l: np.divide(kl_div[l],entropies[0][l]) for l in keys},
                'entropies':entropies}
    
    def result_cond_no(self):
        # dict:
        # {'centers': {'c0':p(c0), 
        #              'c1c2':p(c1,c2), 
        #              'c1':p(c1),
        #              'c2':p(c2)},
        #  'classes': {'k1k2':p(k1,k2), 
        #              'k1':p(k1),
        #              'k2':p(k2)},
        #  'info0':{'mutual_info':I(c1 ; c2),
        #            'mi_proportions':(mi/H(c1), mi/H(c2)),
        #            'entropies':(H(c1), H(c2))},
        #  'info1':{'kl-div':KL(c0 || c1),
        #            'kl_proportion': kl/H(c0),
        #            'entropies':(H(c0),H(c1))},
        #  'info2':{'kl-div':KL(c0 || c2),
        #            'kl_proportion': kl/H(c0),
        #            'entropies':(H(c0),H(c2))},
        #  'infoK':{'mutual_info':I(k1 ; k2),
        #            'mi_proportions':(mi/H(k1), mi/H(k2)),
        #            'entropies':(H(k1), H(k2))},
        #  'sample_info':{'info':values}}}
        self.freqdict_prior.finalize_results()
        self.freqdict_io.finalize_results()
        pc0 = self.freqdict_prior.margin_2
        pc1 = self.freqdict_io.margin_4
        pc2 = self.freqdict_io.margin_5
        return {'centers': {'c0':pc0, 
                            'c1c2':self.freqdict_io.margin_4_5, 
                            'c1':pc1,
                            'c2':pc2},
                'classes': {'k1k2':self.freqdict_io.margin_2_3,
                            'k1':self.freqdict_io.margin_2,
                            'k2':self.freqdict_io.margin_3},
                'info_c1c2': self.freqdict_io.info_result_4_5(),
                'info_c0c1': self.predictinfo_centers_cond_no(pc0, pc1),
                'info_c0c2': self.predictinfo_centers_cond_no(pc0, pc2),
                'infoK': self.freqdict_io.info_result_2_3(),
                'sample_info': self.batchinfo_pio.result_by_no()} 
    
    def result_cond_1(self):
        # tuple of dict:
        # {'centers': {'c0':p(c0|l), 
        #              'c1c2':p(c1,c2|l), 
        #              'c1':p(c1|l),
        #              'c2':p(c2|l)},
        #  'classes': {'k1k2':p(k1,k2|l), 
        #              'k1':p(k1|l),
        #              'k2':p(k2|l)},
        #  'info0':{'mutual_info':I(c1|l ; c2|l),
        #           'mi_proportions':(mi/H(c1|l), mi/H(c2|l)),
        #           'entropies':(H(c1|l), H(c2|l))},
        #  'info1':('kl-div':KL(c0|l || c1|l),
        #           'kl_proportion': kl/H(c0|l),
        #           'entropies':(H(c0|l),H(c1|l))},
        #  'info2':('kl-div':KL(c0|l || c2|l),
        #           'kl_proportion': kl/H(c0|l),
        #           'entropies':(H(c0|l),H(c2|l))},
        #  'infoK':{'mutual_info':I(k1|l ; k2|l),
        #           'mi_proportions':(mi/H(k1|l), mi/H(k2|l)),
        #           'entropies':(H(k1|l), H(k2|l))},
        #  'sample_info':{'info':values}}}
        self.freqdict_prior.finalize_results()
        self.freqdict_io.finalize_results()
        pc0_condl = self.freqdict_prior.margin_2_cond_1
        pc1_condl = self.freqdict_io.margin_4_cond_1
        pc2_condl = self.freqdict_io.margin_5_cond_1
        return {'centers': {'c0':pc0_condl, 
                            'c1c2':self.freqdict_io.margin_4_5_cond_1, 
                            'c1':pc1_condl,
                            'c2':pc2_condl},
                'classes': {'k1k2':self.freqdict_io.margin_2_3_cond_1,
                            'k1':self.freqdict_io.margin_2_cond_1,
                            'k2':self.freqdict_io.margin_3_cond_1},
                'info_c1c2': self.freqdict_io.info_result_4_5_cond_1(),
                'info_c0c1': self.predictinfo_centers_cond_1(pc0_condl, pc1_condl),
                'info_c0c2': self.predictinfo_centers_cond_1(pc0_condl, pc2_condl),
                'infoK': self.freqdict_io.info_result_2_3_cond_1(),
                'sample_info': self.batchinfo_pio.result_by_1()}
    
    def result_cond_1_2(self):
        # tuple of dict:
        # {'centers': {'c0':None, 
        #              'c1c2':p(c1,c2|l,k1), 
        #              'c1':p(c1|l,k1),
        #              'c2':p(c2|l,k1)},
        #  'classes': {'k2':p(k2|l,k1)},
        #  'info0':{'mutual_info':I(c1|l,k1 ; c2|l,k1),
        #           'mi_proportions':(mi/H(c1|l,k1), mi/H(c2|l,k1)),
        #           'entropies':(H(c1|l,k1), H(c2|l,k1))},
        #  'info1':None,
        #  'info2':None,
        #  'infoK':None,
        #  'sample_info':{'info':values}}}
        self.freqdict_io.finalize_results()
        pc1_condlk1 = self.freqdict_io.margin_4_cond_1_2
        pc2_condlk1 = self.freqdict_io.margin_5_cond_1_2
        return {'centers': {'c0':None, 
                            'c1c2':self.freqdict_io.margin_4_5_cond_1_2, 
                            'c1':pc1_condlk1,
                            'c2':pc2_condlk1},
                'classes': {'k1k2':None, # to not overload memory
                            'k1':None,
                            'k2':self.freqdict_io.margin_3_cond_1_2},
                'info_c1c2': self.freqdict_io.info_result_4_5_cond_1_2(),
                'info_c0c1': None,
                'info_c0c2': None,
                'infoK': None,
                'sample_info': self.batchinfo_pio.result_by_1_2()}  
    
    def result_cond_1_3(self):
        # tuple of dict:
        # {'centers': {'c0':None, 
        #              'c1c2':p(c1,c2|l,k2), 
        #              'c1':p(c1|l,k2),
        #              'c2':p(c2|l,k2)},
        #  'classes': {'k2':p(k2|l,k2)},
        #  'info0':{'mutual_info':I(c1|l,k2 ; c2|l,k2),
        #           'mi_proportions':(mi/H(c1|l,k2), mi/H(c2|l,k2)),
        #           'entropies':(H(c1|l,k2), H(c2|l,k2))},
        #  'info1':None,
        #  'info2':None,
        #  'infoK':None,
        #  'sample_info':{'info':values}}}
        self.freqdict_io.finalize_results()
        pc1_condlk2 = self.freqdict_io.margin_4_cond_1_3
        pc2_condlk2 = self.freqdict_io.margin_5_cond_1_3
        return {'centers': {'c0':None, 
                            'c1c2':self.freqdict_io.margin_4_5_cond_1_3, 
                            'c1':pc1_condlk2,
                            'c2':pc2_condlk2},
                'classes': {'k1k2':None, # to not overload memory
                            'k1':self.freqdict_io.margin_2_cond_1_3,
                            'k2':None},
                'info_c1c2': self.freqdict_io.info_result_4_5_cond_1_3(),
                'info_c0c1': None,
                'info_c0c2': None,
                'infoK': None,
                'sample_info': self.batchinfo_pio.result_by_1_3()}
    
    def result_cond_1_2_3(self):
        # tuple of dict:
        # {'centers': {'c0':None, 
        #              'c1c2':p(c1,c2|l,k1,k2), 
        #              'c1':p(c1|l,k1,k2),
        #              'c2':p(c2|l,k1,k2)},
        #  'classes': None,
        #  'info0':{'mutual_info':I(c1|l,k1,k2 ; c2|l,k1,k2),
        #           'mi_proportions':(mi/H(c1|l,k1,k2), mi/H(c2|l,k1,k2)),
        #           'entropies':(H(c1|l,k1,k2), H(c2|l,k1,k2))},
        #  'info1':None,
        #  'info2':None,
        #  'infoK':None,
        #  'sample_info':{'info':values}}}
        self.freqdict_io.finalize_results()
        pc1_condlk1k2 = self.freqdict_io.margin_4_cond_1_2_3
        pc2_condlk1k2 = self.freqdict_io.margin_5_cond_1_2_3
        return {'centers': {'c0':None, 
                            'c1c2':self.freqdict_io.margin_4_5_cond_1_2_3, 
                            'c1':pc1_condlk1k2,
                            'c2':pc2_condlk1k2},
                'classes': {'k1k2':None, # to not overload memory
                            'k1':None,
                            'k2':None},
                'info_c1c2': self.freqdict_io.info_result_4_5_cond_1_2_3(),
                'info_c0c1': None,
                'info_c0c2': None,
                'infoK': None,
                'sample_info': self.batchinfo_pio.result_by_1_2_3()}
    
    def reset(self):
        self.freqdict_prior.reset_state()
        self.freqdict_io.reset_state()
        self.batchinfo_pio.reset_state()
    
    def fit_batch(self, lab, cla, img_batch):
        # lab is str
        # cla is a 2-tuple for keys k1 & k2
        # img_batch is a 3-tuple for keys c0,c1,c2
        # batch dims: [batch, time, lambda, 1]
        assert len(cla[0])==len(cla[1])==len(img_batch[0])==len(img_batch[1])==len(img_batch[2]), "batch dim should be coherent for all inputs"
        assert img_batch[1].shape[0]==img_batch[2].shape[0], "the time length should be cohenrent in the data"
        res = tuple(np.vstack(
            [to_kcentroid_seq(
                np.squeeze(ib[i]), k=1)[1][:,0] for i in range(ib.shape[0])]) for ib in img_batch) # center result for prior/in/out
        self.batch_info_update(lab,cla,res)
        lab = (np.hstack([[ll]*img_batch[0][i].shape[0] for i,ll in enumerate(lab)]), #lab for prior
               np.hstack([[ll]*img_batch[1][i].shape[0] for i,ll in enumerate(lab)])) # lab for pred
        cla = tuple(np.hstack([[cc]*img_batch[1][i].shape[0] for i,cc in enumerate(c)]) for c in cla) # cla for pred_in/out
        res = tuple(np.hstack(resb) for resb in res) # center result for prior/in/out
        self.update(lab, cla, res)
        # self.ibtime_pio.update_state(lab, cla, res)
