# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:23:26 2022

@author: Denis
"""
import numpy as np
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("fname", 'compare_ib_long.npz', "File name: path with name of the file to read")

def main():
    res = np.load(FLAGS.fname, allow_pickle=True)
    dres = res['res'].all()
    res.close()
    for k in dres.keys():
        print("{}".format(k))
        for dt in dres[k].keys():
            print("{}".format(dt))
            for met in dres[k][dt].keys():
                print("{} : {:.3f} /".format(met,dres[k][dt][met]))

if __name__ == '__main__':
    app.run(main)
