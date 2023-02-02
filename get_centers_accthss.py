# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:39:36 2022

@author: Denis
"""
import numpy as np
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("fname", 'compare_centers.npz', "File name: path with name of the file to read")

def main():
    res = np.load(FLAGS.fname, allow_pickle=True)
    dres = res['res'].all()
    res.close()
    for k in dres.keys():
        print("{}".format(k))
        for lab in dres[k].keys():
            print("{}".format(lab))
            for met in dres[k][lab].keys():
                print(["{} : {:.1f}", "{} : {:.3f}"][int('ss' in met)].format(
                    met, dres[k][lab][met]))

if __name__ == '__main__':
    app.run(main)
