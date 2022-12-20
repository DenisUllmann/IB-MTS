# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:40:49 2022

@author: Denis
"""
import numpy as np

def tss_hss_all(cm, labels):
    """
    TSS (True Skill Statistics - Hansen & Kuipers 1965) metric 
    HSS (Heidke Skill Score - Heidke 1926)
    Parameters
    ----------
    cm : TYPE np.array squared array shape n x n
        DESCRIPTION. the confusion matrix of a classifier (n classes)
        first dim: true labels
        second dim: predicted labels
    
    labels : TYPE list or np.array array shape n
        DESCRIPTION. labels for cm
    Returns
    -------
    the TSS scores for each labels VS all the others

    """
    assert len(labels)==len(cm), "Given cm and labels should have compatible shapes (here {} and {})".format(cm.shape, len(labels))
    tss, hss = {}, {}
    for ilbl, label in enumerate(labels):
        cmi = np.asarray([
            [cm[ilbl, ilbl],
             np.sum(
                 [cm[ilbl,j] for j in range(len(cm)) if j!=ilbl])],
            [np.sum(
                [cm[i, ilbl] for i in range(len(cm)) if i!=ilbl]),
             np.sum(
                 [[cm[i, j] for i in range(len(cm)) if i!=ilbl
                   ] for j in range(len(cm)) if j!=ilbl]
                 )]
             ])
        assert np.isclose(np.sum(cm),np.sum(cmi)), "error in calculations"
        tss[label], hss[label] = tss_hss(cmi)
    tss['glob'], hss['glob'] = tss_hss(cm)
    return tss, hss

def tss_hss(cm):
    genu_marg = np.sum(cm, axis = 1)
    pred_marg = np.sum(cm, axis = 0)
    total = np.sum(cm)
    exp_rand = genu_marg.reshape([-1,1])@pred_marg.reshape([1,-1])/total
    ideal = np.diag(genu_marg)
    ide_rand = genu_marg.reshape([-1,1])@genu_marg.reshape([1,-1])/total
    tss = np.divide(np.trace(cm-exp_rand),np.trace(ideal-ide_rand))
    hss = np.divide(np.trace(cm-exp_rand),np.trace(ideal-exp_rand))
    return tss, hss