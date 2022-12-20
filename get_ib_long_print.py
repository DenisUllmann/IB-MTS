# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:23:26 2022

@author: Denis
"""
import numpy as np
res = np.load('compare_ib_long.npz', allow_pickle=True)
dres = res['res'].all()
res.close()
for k in dres.keys():
    print("{}".format(k))
    for dt in dres[k].keys():
        print("{}".format(dt))
        for met in dres[k][dt].keys():
            print("{} : {:.3f} /".format(met,dres[k][dt][met]))