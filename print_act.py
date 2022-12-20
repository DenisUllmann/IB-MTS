# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:37:48 2022

@author: Denis
"""
import numpy as np
res = np.load('compare_act.npz', allow_pickle=True)
dres = res['res'].all()
res.close()
mkey = 'ClassifiedInput VS ClassifiedOutput'
for k in dres.keys():
    print("{}".format(k))
    for lab in dres[k][mkey].keys():
        print(lab)
        print(("{} : {:.1f}  /"+"{} : {:.3f}  /"*(len(dres[k][mkey][lab])-1)).format(*[vv  for v in dres[k][mkey][lab].items() for vv in v]))