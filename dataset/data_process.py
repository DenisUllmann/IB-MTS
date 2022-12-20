# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:04:26 2019

@author: Denis Ullmann
"""
import numpy as np
import os
from copy import copy

from irisreader.data.mg2k_centroids import get_mg2k_centroids
from irisreader.data.mg2k_centroids import LAMBDA_MIN as centroid_lambda_min, LAMBDA_MAX as centroid_lambda_max
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

dataset_adress = 'C:/Users/Denis/Documents/IRIS/iris_level_2C/'

def to_kcentroid_seq(sample, k=1):
    # sample shape[1] is the lambda dim
    centroids = get_mg2k_centroids()
    lambda_min = 2793.8401
    lambda_max = 2806.02
    
    f = interp1d( np.linspace( centroid_lambda_min, centroid_lambda_max, centroids.shape[1] ), centroids, kind="cubic" )
    centroids_interpolated = f( np.linspace( centroid_lambda_min, centroid_lambda_max, num=200 ) )
    centroids_interpolated /= np.max( centroids_interpolated, axis=1 ).reshape(-1,1)
    
    g = interp1d( np.linspace( lambda_min, lambda_max, sample.shape[1] ), sample, kind="cubic" )
    sample_interpolated = g( np.linspace( centroid_lambda_min, centroid_lambda_max, num=200 ) )
    sample_interpolated /= np.max( sample_interpolated, axis=1 ).reshape(-1,1)
    
    nc = NearestNeighbors(n_neighbors=k).fit( centroids_interpolated, np.arange(len(centroids)) )
    k_distances, k_assigned_centroids = nc.kneighbors( sample_interpolated )
    
    return k_distances, k_assigned_centroids

def kcentroids_equal(pred, real):
    return [np.sum(np.isin(a, b))>0 for a,b in zip(pred, real)]

def forplot_assignement_accuracy(kcent_equal, bin_size=1):
    means = [np.mean(kcent_equal[i:i+bin_size]) for i in range(0, len(kcent_equal),bin_size)]
    xs = [i+len(kcent_equal[i:i+bin_size])/2 for i in range(0, len(kcent_equal),bin_size)]
    return xs, means

def forplot_kcentroids_accuracy(pred, real, k=1, bin_size=1):
    return forplot_assignement_accuracy(kcentroids_equal(pred, real, k), bin_size)

def comb(n,k):
    return np.math.factorial(n)/np.math.factorial(k)/np.math.factorial(n-k)

def kinter(k):
    """ returns the probability for two sets of k values choosen randomly in a set of 53 to have an intersection """
    return(comb(53,k) - comb(53 - k, k))/comb(53, k)

def timeseqs_clean(timeseq_list, next_seq_pos, filename):
    clean_list = []
    clean_pos = []
    seq_pos = 0
    for seq in timeseq_list:
        if np.sum(np.isnan(seq))<seq.size: # not a nan sequence
            clean_seq = seq[~np.isnan(seq).any(axis=1)] # remove times with spectra including nans
            if clean_seq.shape[0] >= clean_seq.shape[1] and np.sum(np.isnan(clean_seq))==0: # check that time dim is larger than spectral dim
                clean_list.append(clean_seq)
                clean_pos.append([seq_pos + next_seq_pos, filename, seq_pos, ~np.isnan(seq).any(axis=1)]) # positions = [[sequence position in all seq of all files, file path, sequence position in file, np.array([selected time positions = 1, 0 otherwise])], ..]
            # TODO see if it need to extend for missing data
        seq_pos += 1
    return clean_list, clean_pos, seq_pos + next_seq_pos

def find_max_data(data):    # V2
    mx = 0
    for seq in data:
        mx = max(mx, np.max(seq))
    return mx

def find_mean_nocosmic(data, t): # V2
    # returns the mean value of the supposed non cosmic ray values (smaller than t [2000 DN/s])
    return np.mean(np.concatenate([u[u<2000].flatten() for u in data]))

def no_cosmic(data, t): # V2
    data_no_c = []
    mean_no_c = find_mean_nocosmic(data, t)
    for seq in data:
        seq[seq >= t] = mean_no_c
        data_no_c.append(seq)
    return data_no_c

def rescale_data(data, mx): # V2
    return [seq / mx for seq in data]

def rescale_data_by_seqs(data): # V3
    return[seq / np.max(seq) for seq in data]

def create_labelines_timeseq_dataset(dir_path, labels, labels_shape):
    np_data = []
    positions = np.array([],dtype=np.float64)
    ini = 0
    keyfunc = np.vectorize(lambda a, s: a[s])   # V2
    # print(os.walk(dir_path))
    # for dirpath, dirnames, filenames in os.walk(dir_path):
    #     for filename in filenames:
    #         print("dir_path name", filename)
    #         print('isfile', os.path.isfile(dirpath+'/'+filename))
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if any(label in filename for filename in filenames for label in labels):
            for filename in filenames:
                if 'npz' in filename and 'spectral_files' not in filename and any(label in filename for label in labels):
                    mg_label = np.load(dirpath+'/'+filename, allow_pickle = True)   # V2
                    mg_label_data = mg_label['data']
                    # print('shape', mg_label_data.shape)
                    #mg_label_exp = keyfunc(mg_label['headers'], 'EXPTIME')  # V2
                    #mg_label_data = mg_label_data / mg_label_exp[:,:,None,None] # V2
                    assert mg_label_data.shape[3] == labels_shape,'Shape %i not expected from the label with shape %i in file %s'%(mg_label_data.shape[3],labels_shape, filename)
                    mg_label_data = list(mg_label_data.transpose(0,2,1,3).reshape(-1,mg_label_data.shape[1], mg_label_data.shape[3])) # list [np.arr(time, spectra), ..]
                    if len(np_data) == 0:
                        np_data, positions, ini = timeseqs_clean(mg_label_data, ini, filename)
                    else:
                        new_data, new_positions, ini = timeseqs_clean(mg_label_data, ini, filename)
                        np_data.extend(new_data)
                        positions.extend(new_positions)
    print('%i samples found'%len(np_data))
    return np_data, positions

def convertdata_for_training(data, positions, labels_shape, mask_ratio):
    np_data = []
    c_positions = []
    idx = 0
    for seq, position in zip(data, positions):
        assert seq.shape[1] == labels_shape, "ERROR: the sequence %i from %s has not the right length for spectra: %i for %i expected"%(position[2], position[1], seq.shape[1], labels_shape)
        assert np.sum(position[3]) == seq.shape[0], "ERROR: mismatch between the sequence %i from %s and its position description"%(position[2], position[1])
        if np.sum(position[3]) == labels_shape:
            np_data.append(seq)
            c_positions.append(position.insert(0, idx))
            idx += 1
        else:
            assert seq.shape[0] > seq.shape[1], "ERROR: the sequence %i from %s  should have a time length larger than spectra: here %i < %i"%(position[2], position[1], seq.shape[0], seq.shape[1])
            stride = int(seq.shape[1]*(1-mask_ratio))
            time_seq_ori = position[3]
            for stride_idx in range((seq.shape[0]-labels_shape+int(seq.shape[1]*(1-mask_ratio)))//stride):
                np_data.append(seq[stride * stride_idx : stride * stride_idx + labels_shape, :]) # get patches of the sequences of length label_length, stride for the patches is label_length*(1-mask_ratio)
                new_time_seq = copy(time_seq_ori)
                new_time_seq[np.where(time_seq_ori==1)[0][:stride * stride_idx]] = 0 # get the corresponding time positions in the original sequence
                new_time_seq[np.where(time_seq_ori==1)[0][stride * stride_idx + labels_shape:]] = 0
                c_positions.append([idx, position[0], position[1], position[2], new_time_seq])
                idx += 1
            if ((seq.shape[0]-labels_shape+int(seq.shape[1]*(1-mask_ratio))) % stride) > 0:
                np_data.append(seq[-labels_shape:, :])
                new_time_seq = copy(time_seq_ori)
                new_time_seq[np.where(time_seq_ori==1)[0][:-labels_shape]] = 0
                c_positions.append([idx, position[0], position[1], position[2], new_time_seq])
                idx += 1
    return np.expand_dims(np.array(np_data), axis = -1), c_positions

def timeseqs_clean_positions(timeseq_list, next_seq_pos, filename):
    clean_pos = []
    seq_pos = 0
    for seq in timeseq_list:
        if np.sum(np.isnan(seq))<seq.size: # not a nan sequence
            clean_seq = seq[~np.isnan(seq).any(axis=1)] # remove times with spectra including nans
            if clean_seq.shape[0] >= clean_seq.shape[1] and np.sum(np.isnan(clean_seq))==0: # check that time dim is larger than spectral dim
                clean_pos.append([seq_pos + next_seq_pos, filename, seq_pos, ~np.isnan(seq).any(axis=1)]) # positions = [[sequence position in all seq of all files, file path, sequence position in file, np.array([selected time positions = 1, 0 otherwise])], ..]
        seq_pos += 1
    return clean_pos, seq_pos + next_seq_pos

def dataset_positions(dir_path, labels, labels_shape):
    positions = []
    ini = 0
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if any(label in filename for filename in filenames for label in labels):
            for filename in filenames:
                if 'npz' in filename and any(label in filename for label in labels):
                    mg_label_data = np.load(dirpath+'/'+filename)['data']
                    assert mg_label_data.shape[3] == labels_shape,'Shape %i not expected from the label with shape %i in file %s'%(mg_label_data.shape[3],labels_shape, filename)
                    mg_label_data = list(mg_label_data.transpose(0,2,1,3).reshape(-1,mg_label_data.shape[1], mg_label_data.shape[3])) # list [np.arr(time, spectra), ..]
                    if len(positions) == 0:
                        positions, ini = timeseqs_clean_positions(mg_label_data, ini, filename)
                    else:
                        new_positions, ini = timeseqs_clean_positions(mg_label_data, ini, filename)
                        positions.extend(new_positions)
    return positions

def convertpos_for_training(data, positions, labels_shape, mask_ratio):
    c_positions = []
    for seq, position in zip(data, positions):
        assert seq.shape[1] == labels_shape, "ERROR: the sequence %i from %s has not the right length for spectra: %i for %i expected"%(position[2], position[1], seq.shape[1], labels_shape)
        assert np.sum(position[3]) == seq.shape[0], "ERROR: mismatch between the sequence %i from %s and its position description"%(position[2], position[1])
        if np.sum(position[3]) == labels_shape:
            c_positions.append(position)
        else:
            assert seq.shape[0] > seq.shape[1], "ERROR: the sequence %i from %s  should have a time length larger than spectra: here %i < %i"%(position[2], position[1], seq.shape[0], seq.shape[1])
            stride = int(seq.shape[1]*(1-mask_ratio))
            time_seq_ori = position[3]
            for stride_idx in range((seq.shape[0]-labels_shape+int(seq.shape[1]*(1-mask_ratio)))//stride):
                new_time_seq = copy(time_seq_ori)
                new_time_seq[np.where(time_seq_ori==1)[0][:stride * stride_idx]] = 0
                new_time_seq[np.where(time_seq_ori==1)[0][stride * stride_idx + labels_shape:]] = 0
                c_positions.append([position[0], position[1], position[2], new_time_seq])
            if ((seq.shape[0]-labels_shape+int(seq.shape[1]*(1-mask_ratio))) % stride) > 0:
                new_time_seq = copy(time_seq_ori)
                new_time_seq[np.where(time_seq_ori==1)[0][:-labels_shape]] = 0
                c_positions.append([position[0], position[1], position[2], new_time_seq])
    return c_positions

def retrieve_longtimeseq(position):
    time_select = position[3]
    filename = position[1]
    input_shape = np.load(filename)['data'].shape
    seq = list(np.zeros(input_shape).transpose(0,2,1,3).reshape(-1,input_shape[1], input_shape[3])) # list [np.arr(time, spectra), ..] 0 everywhere
    seq[position[2]] += 1 # add 1 to the right sequence
    seq = np.array(seq).reshape(input_shape[0], input_shape[2], input_shape[1], input_shape[3]).transpose(0,2,1,3)
    assert seq.shape == input_shape, "Error in retrieving timesequence shape"
    raster, y_pos = list(np.array(np.where(np.sum(np.sum(seq,axis=1),axis=2)>0)).squeeze())
    assert (isinstance(raster, int) or isinstance(raster, np.integer)) and (isinstance(y_pos, int) or isinstance(y_pos, np.integer)), "Error too many positions found for this timesequence (should be only for unique raster position and y position)"
    return filename, raster, time_select, y_pos

def retrieve_traintimeseq(position):
    time_select = position[4]
    filename = position[2]
    input_shape = np.load(dataset_adress + filename)['data'].shape
    seq = list(np.zeros(input_shape).transpose(0,2,1,3).reshape(-1,input_shape[1], input_shape[3])) # list [np.arr(time, spectra), ..] 0 everywhere
    seq[position[3]] += 1 # add 1 to the right sequence
    seq = np.array(seq).reshape(input_shape[0], input_shape[2], input_shape[1], input_shape[3]).transpose(0,2,1,3)
    assert seq.shape == input_shape, "Error in retrieving timesequence shape"
    raster, y_pos = list(np.array(np.where(np.sum(np.sum(seq,axis=1),axis=2)>0)).squeeze())
    assert (isinstance(raster, int) or isinstance(raster, np.integer)) and (isinstance(y_pos, int) or isinstance(y_pos, np.integer)), "Error too many positions found for this timesequence (should be only for unique raster position and y position)"
    return filename, raster, time_select, y_pos

def chunkdata_for_longpredict(data, positions, labels_shape, mask_ratio):
    if isinstance(positions[0], int):
        return convertdata_for_training([data], [positions], labels_shape, 1-mask_ratio)
    else:
        return convertdata_for_training(data, positions, labels_shape, 1-mask_ratio)