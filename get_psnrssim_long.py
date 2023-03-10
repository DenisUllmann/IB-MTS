# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:04:34 2022

@author: Denis
"""
import numpy as np
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("fname", 'compare_cvnn_long.npz', "File name: path with name of the file to read")

def main():
    res = np.load(FLAGS.fname, allow_pickle=True)
    dres = res['res'].all()
    res.close()
    for k in dres.keys():
        print("{}".format(k))
        for met in dres[k].keys():
            print("{}".format(met))
            if 'NN' in met:
                print(["{:.1f} /"*len(dres[k][met]),"{:.3f} /"*len(dres[k][met])][
                int(met=='SSIM')].format(100*dres[k][met][0][0]))
            else:
                print(["{:.1f} /"*len(dres[k][met]),"{:.3f} /"*len(dres[k][met])][
                int(met=='SSIM')].format(*dres[k][met]))

if __name__ == '__main__':
    app.run(main)
