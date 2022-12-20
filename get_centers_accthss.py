# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:39:36 2022

@author: Denis
"""
import numpy as np
res = np.load('compare_centers.npz', allow_pickle=True)
dres = res['res'].all()
res.close()
for k in dres.keys():
    print("{}".format(k))
    for lab in dres[k].keys():
        print("{}".format(lab))
        for met in dres[k][lab].keys():
            print(["{} : {:.1f}", "{} : {:.3f}"][int('ss' in met)].format(
                met, dres[k][lab][met]))