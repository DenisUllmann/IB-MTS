# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:18:20 2022

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
        for act in dres[k].keys():
            print("Activity {}".format(act))
            for met in dres[k][act].keys():
                print((["{} : {:.2f}","{} : {:.1f}"][int(met=='acc')]).format(met, dres[k][act][met]))

if __name__ == '__main__':
    app.run(main)
