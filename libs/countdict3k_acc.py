# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:46:43 2022

@author: Denis
"""
import numpy as np

class NPAccuracyOverTime3D(object):
    def __init__(self, labels, class_in, class_out):
        self.labels = labels
        self.class_in = class_in
        self.class_out = class_out
        self.count = {
            l:{
                k1:{
                    k2: {0:0}for k2 in self.class_out
                    } for k1 in self.class_in} for l in self.labels}
    
    def update_states(self, l, k1, k2, t):
        self.add_batch(*self.get_batches([l,k1,k2,t]))
    
    def get_batches(self, elems):
        for idx_e, elem in enumerate(elems):
            if not(any(isinstance(
                    elem, arrtype) for arrtype in [list, np.ndarray])):
                elems[idx_e] = [elem]
        len_batch = len(elems[0])
        assert all(len(elem)==len_batch for elem in elems[:-1]
                   ), "batches should have same length"
        if len(elems[-1])==1:
            elems[-1] = elems[-1]*len_batch
        return elems
    
    def add_batch(self, l, k1, k2, t):
        for keys in zip(l, k1, k2, t):
            self.add_or_create(self.count, keys)
    
    def add_or_create(self, dicti, keys):
        try:
            l, k1, k2, t = keys
            dicti[l][k1][k2][t] += 1
        except:
            dicti[l][k1][k2][t] = 1
    
    def reducesum_bykeys(self, keys):
        return self.reducesum_self(self.count, 0, keys)
    
    def reducesum_self(self, dicti, try_key, keys):
        # print('reducesum', dicti, try_key, keys)
        if isinstance(dicti, dict):
            if try_key in keys:
                return {k: self.reducesum_self(
                    v, try_key, [ke-1 for ke in keys]) for k,v in dicti.items()}
            else:
                return self.dictsum([self.reducesum_self(
                    v, try_key, [ke-1 for ke in keys]) for k,v in dicti.items()])
        else:
            return dicti
    
    def dictsum(self, dicts):
        # print('dicts', dicts)
        # dicts is a list of dicts
        nkeys = self.nkeys(dicts[0])
        assert all(self.nkeys(d)==nkeys for d in dicts), "all dicts should have same length"
        if isinstance(dicts[0], dict):
            handle_nokey = lambda dicti, key: dicti[key] if key in dicti.keys() else 0
            return {k: self.dictsum(
                [handle_nokey(d,k) for d in dicts]) for k in set(
                    [e for l in list(list(dicts[n_d].keys()) for n_d in range(
                        len(dicts))) for e in l])}
        else:
            return sum(dicts)
    
    def nkeys(self, dicti):
        n = 0
        testdict = dicti
        while True:
            if isinstance(testdict, dict):
                testdict = testdict[list(testdict.keys())[0]]
                n += 1
            else:
                self.assert_len(dicti, n)
                return n
    
    def assert_len(self, dicti, n):
        if n==0 and not(isinstance(dicti, dict)):
            return True
        else:
            assert all(self.assert_len(d, n-1) for d in dicti.values()), "wrong format for dict"
            return all(self.assert_len(d, n-1) for d in dicti.values())
    
    def results(self):
        dkeys = np.arange(4)
        keys = [kk for k in [
            [','.join(['%s'%a,'%s'%b,'%s'%c]) for a in dkeys for b in dkeys for c in dkeys if a<b<c],
            [','.join(['%s'%a,'%s'%b]) for a in dkeys for b in dkeys if a<b], 
            [','.join(['%s'%a]) for a in dkeys]] for kk in k]
        return {
            **{'all': self.count,
               'glob': self.reducesum_bykeys([])},
            **{k: self.reducesum_bykeys([int(kk) for kk in k.split(',')]) for k in keys}
            }