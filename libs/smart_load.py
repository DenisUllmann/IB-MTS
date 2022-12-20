# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:45:54 2022

@author: Denis
"""
import h5py
from keras.saving import saving_utils
from keras.saving.hdf5_format import load_attributes_from_hdf5_group
from keras.saving.hdf5_format import load_subset_weights_from_hdf5_group
from keras.saving.hdf5_format import preprocess_weights_for_loading
from keras import backend

def smart_load_hdf5(filepath, self):
    """
    The smart_loading is built from elements of keras.saving.hdf5_format
    It allows to load only part of the layers saved into the corresponding 
    part of the layers to load
    To be used, all layers should have defined names (not defined automatically)
    """
    assert saving_utils.is_hdf5_filepath(filepath), "Wrong file format, should be hdf5"
    f = h5py.File(filepath, "r")
    #hdf5_format.load_weights_from_hdf5_group(f, self)
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None
    filtered_layers = []
    for layer in self.layers:
        weights = layer.trainable_weights + layer.non_trainable_weights
        if weights:
            filtered_layers.append(layer)
    layer_names = load_attributes_from_hdf5_group(f, "layer_names")
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    s_l = []
    for inms, name in enumerate(layer_names):
        try:
            # If bugs, try with permutations of the True indexes
            inspect = [all(any([sn in flw.name for flw in fl.weights]) for sn in f[name].keys()) for fl in filtered_layers]
            for isl in s_l:
                inspect[isl[1]] = False
            ifl = inspect.index(True)
        except:
            pass
        else:
            s_l.append((inms, ifl))
    weight_value_tuples = []
    for inmi, ifli in s_l:
        g = f[layer_names[inmi]]
        layer = filtered_layers[ifli]
        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
        weight_values = load_subset_weights_from_hdf5_group(g)
        weight_values = preprocess_weights_for_loading(
            layer, weight_values, original_keras_version, original_backend
        )
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(
                f"Weight count mismatch for layer #{ifli} (named {layer.name} in "
                f"the current model, {name} in the save file). "
                f"Layer expects {len(symbolic_weights)} weight(s). Received "
                f"{len(weight_values)} saved weight(s)"
                )
        weight_value_tuples += zip(symbolic_weights, weight_values)
        
        if "top_level_model_weights" in f:
            symbolic_weights = (
                self._trainable_weights + self._non_trainable_weights
                )
            weight_values = load_subset_weights_from_hdf5_group(
                f["top_level_model_weights"]
                )
            if len(weight_values) != len(symbolic_weights):
                raise ValueError(
                    f"Weight count mismatch for top-level weights when loading "
                    f"weights from file. "
                    f"Model expects {len(symbolic_weights)} top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)"
                    )
            weight_value_tuples += zip(symbolic_weights, weight_values)
        backend.batch_set_value(weight_value_tuples)
        
        # Perform any layer defined finalization of the layer state.
        for layer in self._flatten_layers():
            layer.finalize_state()