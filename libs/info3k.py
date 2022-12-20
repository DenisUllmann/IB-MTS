# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:19:13 2022

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

class NPInfo3_lkk_Dict(object):
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
    def __init__(self, labels, classes1, classes2, centers, name='info_count'):
        self.name = name
        self.labels = labels
        self.classes1 = classes1
        self.classes2 = classes2
        self.centers0, self.centers1, self.centers2 = centers
        self.reset_state()
    
    def count_onesample(self, sample):
        return {str(u):v for u,v in zip(*np.unique(
            sample, return_counts=True, axis=0))}
    
    def count_key_zero(self, count_dict, key):
        if key in count_dict.keys():
            return count_dict[key]
        else:
            return 0
    
    def count_zero(self, count_dict, keys):
        return {u: self.count_key_zero(count_dict,u) for u in keys}
    
    def proba_bysample(self, batch, keys):
        counts = [self.count_zero(
            self.count_onesample(r0),keys) for r0 in batch]
        return [{c:np.divide(
            v, sum(count.values())) for c,v in count.items(
                )} for count in counts]
    
    def jointcount_onesample(self, sample1, sample2, keys):
        jointcount = {u1:{u2:0 for u2 in keys[1]} for u1 in keys[0]}
        for (u1,u2),v in zip(*np.unique(
                list(zip(sample1, sample2)), 
                return_counts=True, axis=0)):
            jointcount[str(u1)][str(u2)] += int(float(v))
        return jointcount
    
    def jointproba_bysample(self, batch1, batch2, keys):
        # keys = set(np.hstack([np.hstack(batch1),
        #                       np.hstack(batch2)]))
        jointcounts = [self.jointcount_onesample(
            r0,r1,keys) for r0,r1 in zip(batch1,batch2)]
        return [{c1:{c2:np.divide(v12,sum(sum(
            jointcount[c11].values())for c11 in jointcount.keys(
                ))) for c2, v12 in v1.items()} for c1,v1 in jointcount.items(
                    )} for jointcount in jointcounts]
    
    def entropy_bysample(self, batch):
        xlog2x = lambda x:x*np.log2([x,1][int(x==0)])
        return [- sum(xlog2x(pp) for pp in p.values()) for p in batch]
    
    def mi_bysample(self, batch):
        xlog2x_yz = lambda x,y,z:[x*np.log2(np.divide(x,y*z)),0][int(x==0)]
        return [sum(sum(xlog2x_yz(
            ppp,
            sum(pp.values()),
            sum(p[uu][uuu] for uu in p.keys())) for uuu,ppp in pp.items(
                )) for pp in p.values()) for p in batch]
    
    def kl_bysample(self, batch1, batch2):
        xlog2x_y = lambda x,y:[x*np.log2(np.divide(x,y)),0][int(x==0 or y==0)]
        assert all(p1.keys()==p2.keys() for p1,p2 in zip(batch1,batch2))
        return [sum(xlog2x_y(p1[k],p2[k]) for k in p1.keys()) for p1,p2 in zip(batch1,batch2)]
    
    def dict_add(self, dictval1, dictval2):
        # adds a float ot a dict to another dict:
        # for a float it adds it to each value of the dict
        # for a dict it adds up the corresponding values, in this case, the keys should correspond.
        # dict can be a dict of dict of unknown level of intrication
        return self.dict_sum([dictval1, dictval2])
    
    def dictval_getkey(self, dictval, key):
        # returns dictval[key] if possible
        try:
            return dictval[key]
        except:
            return dictval
    
    def iterative_try(self, listval, idx, fn):
        try:
            return fn(listval, idx)
        except:
            return self.iterative_try(listval, idx+1, fn)
    
    def dict_sum(self, dictvallist):
        # adds a float ot a dict to another dict:
        # for a float it adds it to each value of the dict
        # for a dict it adds up the corresponding values, in this case, the keys should correspond.
        # dict can be a dict of dict of unknown level of intrication
        try:
            return sum(dictvallist)
        except:
            return self.iterative_try(
                dictvallist, 0,
                lambda listval, idx:
                    {key: self.dict_sum(
                        [self.dictval_getkey(dv, 
                                             key) for dv in listval]
                        ) for key in dictvallist[idx].keys()})
    
    def dict_substract(self, dictval1, dictval2):
        # substracts a float ot a dict to another dict:
        # for a float it substracts it to each value of the dict
        # for a dict it adds up the corresponding values, in this case, the keys should correspond.
        # dict can be a dict of dict of unknown level of intrication
        try:
            return dictval1 - dictval2
        except:
            try:
                return {key: self.dict_substract(
                    dictval1[key], 
                    dictval2[key]) for key in dictval2.keys()}
            except:
                try:
                    return {key: self.dict_substract(
                        dictval1[key], 
                        dictval2) for key in dictval1.keys()}
                except:
                    return {key: self.dict_substract(
                        dictval1, 
                        dictval2[key]) for key in dictval2.keys()}
    
    def dict_zero_divide(self, dictval1, dictval2):
        # self.zero_divides a float ot a dict to another dict:
        # for a float it self.zero_divides it to each value of the dict
        # for a dict it divides the corresponding values, in this case, the keys should correspond.
        # dict can be a dict of dict of unknown level of intrication
        try:
            return self.zero_divide(dictval1, dictval2)
        except:
            try:
                return {key: self.dict_zero_divide(
                    dictval1[key], 
                    dictval2[key]) for key in dictval2.keys()}
            except:
                try:
                    return {key: self.dict_zero_divide(
                        dictval1[key], 
                        dictval2) for key in dictval1.keys()}
                except:
                    return {key: self.dict_zero_divide(
                        dictval1, 
                        dictval2[key]) for key in dictval2.keys()}
    
    def dict_square(self, dictval):
        # squares a float or values of a dict:
        # for a dict it squares up the corresponding values
        # dict can be a dict of dict of unknown level of intrication
        try:
            return dictval**2
        except:
            return {key: self.dict_square(
                dictval[key]) for key in dictval.keys()}
    
    def dict_sqrt(self, dictval):
        # squares a float or values of a dict:
        # for a dict it squares up the corresponding values
        # dict can be a dict of dict of unknown level of intrication
        try:
            return np.sqrt(dictval)
        except:
            return {key: self.dict_sqrt(
                dictval[key]) for key in dictval.keys()}
    
    def update_state(self, lab, cla, res, sample_weight=None):
        # creates info data 'last_info' for each sample of the batch 
        # and update the dicts of info 'count' and 'total'
        # lab is batch 
        # cla is 2-tuple of batch (in, out)
        # res is 3-tuple of batch center assignements (prior, in, out)
        assert all(all(len(cla[j])==len(res[i])==len(lab) for j in range(2)) for i in range(3)), "batch length should be equal for batched data"
        self.last = {
            'p_c0': self.proba_bysample(res[0], self.centers0),
            'p_c1': self.proba_bysample(res[1], self.centers1),
            'p_c2': self.proba_bysample(res[2], self.centers2),
            'p_c1c2': self.jointproba_bysample(res[1], res[2], 
                                               (self.centers1, self.centers2))}
        self.last = {
            **self.last,
            **{
                'KL(c0||c1)': self.kl_bysample(self.last['p_c0'], 
                                               self.last['p_c1']), 
                'KL(c0||c2)': self.kl_bysample(self.last['p_c0'], 
                                               self.last['p_c2']), 
                'I(c1||c2)': self.mi_bysample(self.last['p_c1c2']), 
                'H(c0)': self.entropy_bysample(self.last['p_c0']), 
                'H(c1)': self.entropy_bysample(self.last['p_c1']), 
                'H(c2)': self.entropy_bysample(self.last['p_c2'])}}
        assert set(self.info_keys+self.proba_keys)== set(self.last.keys()), "Info keys should match"
        # Update counts
        count_verif1 = {l:{k1:{k2: 0 for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        for (l,k1,k2), c in zip(*np.unique(
                    list(zip(lab,*cla)),
                    return_counts=True, axis=0)):
            count_verif1[l][k1][k2] += c
            self.count[l][k1][k2] += c
        # Update proba and info
        count_verif2 = {l:{k1:{k2: 0 for k2 in self.classes2} for k1 in self.classes1} for l in self.labels}
        for i in range(len(lab)):
            count_verif2[lab[i]][cla[0][i]][cla[1][i]] += 1
            for key in self.proba_keys+self.info_keys:
                self.total[key][lab[i]][cla[0][i]][cla[1][i]]['mean'] = self.dict_add(
                    self.total[key][lab[i]][cla[0][i]][cla[1][i]]['mean'],
                    self.last[key][i])
                self.total[key][lab[i]][cla[0][i]][cla[1][i]]['std'] = self.dict_add(
                    self.total[key][lab[i]][cla[0][i]][cla[1][i]]['std'],
                    self.dict_square(self.last[key][i]))
            try:
                self.control_totalmean_key('p_c0')
            except:
                print('name failed', self.name)
                print('i_lab failed', i)
                print('counts', self.count)
                self.control_totalmean_key('p_c0')
        assert all(all(all(count_verif1[l][k1][k2]==count_verif2[l][k1][k2] for k2 in self.classes2) for k1 in self.classes1) for l in self.labels), "counting error"
    
    def control_totalmean_key(self, key):
        for l in self.labels:
            for k1 in self.classes1:
                for k2 in self.classes2:
                    assert self.control_dict(self.total[key][l][k1][k2]['mean']), "bad structure label {} class1 {} class2 {}".format(l,k1,k2)
    
    def control_dict(self, test):
        return isinstance(test, dict)
    
    def zero_divide(self,x,y):
        if y == 0:
            assert x==0, "the total should be 0 when the count is 0"
            return 0
        else:
            return np.divide(x,y)
    
    # def finalize_probas(self):
    #     # Get the mean and std of the probas (prior, input, output)
    #     # And also the conditional probas
    #     self.marg_proba = [uu for u in [
    #         [name+condition for name in self.proba_keys] for condition in [
    #             '_by_no', '_by_1', '_by_1_2', '_by_1_3', '_by_1_2_3']
    #         ] for uu in u]
    #     for prob in self.proba_keys:
    #         for l in self.labels:
    #             for k1 in self.classes1:
    #                 for k2 in self.classes2:
    #                     # Get STD * N
    #                     self.total[prob][l][k1][k2][1] = 
    #                     self.total[prob][l][k1][k2][1] = np.sqrt(self.total[prob][l][k1][k2][1]*self.count[l][k1][k2] - self.total[prob][l][k1][k2][0]**2)
    
    def mean_std_frompartmean(self, temp):
        # temp is {'mean', 'std'} where 'std' id the mean of squares
        return {'mean':temp['mean'], 
                'std': self.dict_substract(
                    temp['std'],
                    self.dict_square(temp['mean']))}
    
    def result(self):
        # Outputs dict of (mean_value, std_value) for each 'key' measured form 
        # self.proba_keys and self.info_keys
        temp= {key:{l:{k1:{k2:{
            kkey:self.dict_zero_divide(
                self.total[key][l][k1][k2][kkey],
                self.count[l][k1][k2]) for kkey in self.stat_keys
            } for k2 in self.classes2} for k1 in self.classes1
            } for l in self.labels} for key in self.info_keys+self.proba_keys}
        for key in ['p_c0','p_c1','p_c1']:
            for l in self.labels:
                for k1 in self.classes1:
                    for k2 in self.classes2:
                        assert self.control_dict(temp[key][l][k1][k2]['mean']), "bad structure name {} key {} label {} class1 {} class2 {}".format(self.name, key,l,k1,k2)
        return {key:{l:{k1:{k2: (
            self.count[l][k1][k2],
            self.mean_std_frompartmean(
            temp[key][l][k1][k2]
            )) for k2 in self.classes2} for k1 in self.classes1
            } for l in self.labels} for key in self.info_keys+self.proba_keys}
    
    def result_by_1_2_3(self):
        return self.result()
        
    def result_by_1_2(self):
        # Outputs dict of proba+info measurements
        temp = {key:{l:{k1:{
            kkey: self.dict_zero_divide(
                self.dict_sum([self.total[key][l][k1][k22][kkey] for k22 in self.classes2]),
                self.dict_sum([self.count[l][k1][k22] for k22 in self.classes2])
                ) for kkey in self.stat_keys
            } for k1 in self.classes1} for l in self.labels
            } for key in self.info_keys+self.proba_keys}
        for key in ['p_c0','p_c1','p_c1']:
            for l in self.labels:
                for k1 in self.classes1:
                    assert self.control_dict(temp[key][l][k1]['mean']), "bad structure name {} key {} label {} class1 {}".format(self.name, key,l,k1)
        return {key:{l:{k1: (
            self.dict_sum([self.count[l][k1][k22] for k22 in self.classes2]),
            self.mean_std_frompartmean(
            temp[key][l][k1]
            )) for k1 in self.classes1
            } for l in self.labels} for key in self.info_keys+self.proba_keys}
    
    def result_by_1_3(self):
        # Outputs dict of proba+info measurements
        temp = {key:{l:{k2:{
            kkey: self.dict_zero_divide(
                self.dict_sum([self.total[key][l][k11][k2][kkey] for k11 in self.classes1]),
                self.dict_sum([self.count[l][k11][k2] for k11 in self.classes1])
                ) for kkey in self.stat_keys
            } for k2 in self.classes2} for l in self.labels
            } for key in self.info_keys+self.proba_keys}
        for key in ['p_c0','p_c1','p_c1']:
            for l in self.labels:
                for k2 in self.classes2:
                    assert self.control_dict(temp[key][l][k2]['mean']), "bad structure name {} key {} label {} class2 {}".format(self.name, key,l,k2)
        return {key:{l:{k2: (
            self.dict_sum([self.count[l][k11][k2] for k11 in self.classes1]),
            self.mean_std_frompartmean(
            temp[key][l][k2]
            )) for k2 in self.classes2
            } for l in self.labels} for key in self.info_keys+self.proba_keys}
    
    def result_by_1(self):
        # Outputs dict of info measurements
        temp = {key:{l:{
            kkey: self.dict_zero_divide(
                self.dict_sum([self.dict_sum([
                    self.total[key][l][k11][k22][kkey] for k11 in self.classes1]
                    ) for k22 in self.classes2]),
                self.dict_sum([self.dict_sum([
                    self.count[l][k11][k22] for k11 in self.classes1]
                    ) for k22 in self.classes2])
                ) for kkey in self.stat_keys
            } for l in self.labels} for key in self.info_keys+self.proba_keys}
        # case when the selected set of keys has never been updated 
        # (but it should have a dict structure filled with zeros)
        for key in ['p_c0','p_c1','p_c1']:
            for l in self.labels:
                assert self.control_dict(temp[key][l]['mean']), "bad structure name {} key {} label {}".format(self.name, key,l)
        return {key:{l: (
            self.dict_sum([self.dict_sum([
                self.count[l][k11][k22] for k11 in self.classes1]
                ) for k22 in self.classes2]),
            self.mean_std_frompartmean(
            temp[key][l]
            )) for l in self.labels} for key in self.info_keys+self.proba_keys}
    
    def result_by_no(self):
        # Outputs dict of info measurements
        temp = {key: {
            kkey: self.dict_zero_divide(
                self.dict_sum([self.dict_sum([self.dict_sum([
                    self.total[key][ll][k11][k22][kkey] for ll in self.labels]
                    ) for k11 in self.classes1]) for k22 in self.classes2]),
                self.dict_sum([self.dict_sum([self.dict_sum([
                    self.count[ll][k11][k22] for ll in self.labels]
                    ) for k11 in self.classes1]) for k22 in self.classes2])
                ) for kkey in self.stat_keys
            } for key in self.info_keys+self.proba_keys}
        for key in ['p_c0','p_c1','p_c1']:
            assert self.control_dict(temp[key]['mean']), "bad structure name {} key {}".format(self.name, key)
        return {key: (
            self.dict_sum([self.dict_sum([self.dict_sum([
                self.count[ll][k11][k22] for ll in self.labels]
                ) for k11 in self.classes1]) for k22 in self.classes2]),
            self.mean_std_frompartmean(
            temp[key]
            )) for key in self.info_keys+self.proba_keys}
    
    def reset_state(self):
        # To reset the state.
        self.stat_keys = ['mean', 'std']
        self.proba_keys = ['p_c0', 'p_c1', 'p_c2', 'p_c1c2']
        # self.proba_keys = [uu for u in [
        #     [name+condition for name in self.proba_keys] for condition in [
        #         '_by_no', '_by_1', '_by_1_2', '_by_1_3', '_by_1_2_3']
        #     ] for uu in u]
        self.info_keys = ['KL(c0||c1)', 'KL(c0||c2)', 'I(c1||c2)', 'H(c0)', 'H(c1)', 'H(c2)']
        self.total = {
            key:{l:{k1:{k2: {kkey:0 for kkey in self.stat_keys} for k2 in self.classes2 # (mean_proba, std_proba)
                        } for k1 in self.classes1
                    } for l in self.labels
                 } for key in self.proba_keys+self.info_keys}# if '_by_1_2_3' in prob},
        self.count = {
            l:{k1:{k2: 0 for k2 in self.classes2 # (mean_proba, std_proba)
                   } for k1 in self.classes1
               } for l in self.labels
            }
        
        self.last = {
            'p_c0': [{c0:0 for c0 in self.centers0}],
            'p_c1': [{c1:0 for c1 in self.centers1}],
            'p_c2': [{c2:0 for c2 in self.centers2}],
            'p_c1c2': [{c1:{c2:0 for c2 in self.centers2} for c1 in self.centers1}]}
        self.last = {
            **self.last,
            **{
                'KL(c0||c1)': [0], 
                'KL(c0||c2)': [0], 
                'I(c1||c2)': [0], 
                'H(c0)': [0], 
                'H(c1)': [0], 
                'H(c2)': [0]}}
        assert set(self.info_keys+self.proba_keys)== set(self.last.keys()), "Info keys should match"
        # Update proba and info
        for l in self.labels:
            for k1 in self.classes1:
                for k2 in self.classes2:
                    for key in self.proba_keys+self.info_keys:
                        self.total[key][l][k1][k2]['mean'] = self.dict_add(
                            self.total[key][l][k1][k2]['mean'],
                            self.last[key][0])
                        self.total[key][l][k1][k2]['std'] = self.dict_add(
                            self.total[key][l][k1][k2]['std'],
                            self.dict_square(self.last[key][0]))
        try:
            self.control_totalmean_key('p_c0')
        except:
            print('init failed')
            print('name', self.name)