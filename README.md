# IB-MTS

Code of the paper "Multi variables time series information bottleneck"

[Multi variables time series information bottleneck](https://arxiv.org) <br>
[Denis Ullmann](https://orcid.org/0000-0002-7179-005X), [Olga Taran](https://orcid.org/0000-0001-8537-5204), [Slava Voloshynoskyy](https://orcid.org/0000-0003-0416-9674) <br>
University of Geneva <br>
SNSF NRP75 Grant

## Intro

This repository contains all the necessary information to replicate the results obtained in the paper "Multi variables time series information bottleneck"

The part of the code for partial convolution layers from https://github.com/MathiasGruber/PConv-Keras/blob/master/requirements.txt is updated to work on TensorFlow2.

The code is dedicated to Multiple Time Series (MTS) forecasting and includes evaluations with several metrics: 
* MTS metrics: MAE, MAPE, RMSE
* CV metrics: PSNR, SSIM
* Information Theory metrics: Mutual Information I, KL-divergence when the time profiles of the MTS are already clustered (eg. for [IRIS data](https://iris.lmsal.com/data.html)).
* Astrophysics metrics: profile and activity classification error when the MTS data is already labeled or the time profiles of the MTS are already clustered (eg. for [IRIS data](https://iris.lmsal.com/data.html)).

Different concurrent models are evaluated and their codes are present in the repository:
* IB-MTS that makes use of UNet with partial convolution
* [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735), [GRU](
https://doi.org/10.48550/arXiv.1409.1259)
* IB-LSTM, IB-GRU that are made of 2 [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)s or [GRU](
https://doi.org/10.48550/arXiv.1409.1259)s to encode and 2 [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)s or [GRU](
https://doi.org/10.48550/arXiv.1409.1259)s to decode
* [NBeats](https://arxiv.org/abs/1905.10437)


## Contents

*   [Requirements](#requirements)
*   [Install](#install)
*   [Download data](#download-data)
*   [Download weights](#download-weights)
*   [Download test results](#download-test-results)
*   [Usage](#usage)
*   [Brief explanation](#brief-explanation)
*   [License](#license)

## Requirements

Check your available memory, used and downloaded datasets may be significantly big.

Preprocessings, trainings, testings and figure generation were performed on a machine with more that free 15GB and CUDA capable GPUs with more than 10GB RAM each.

You harware and software should be TensorFlow2-friendly (see requirements at [TF installation](https://www.tensorflow.org/install))

You may create a virtual environnement and use `requirements.txt`:

```
$ pip install -r requirements.txt
```

## Install

Use Git to clone this repository into your computer.

```
git clone https://github.com/DenisUllmann/IB-MTS
```

Or pip install from Git:

```
pip install git+https://github.com/DenisUllmann/IB-MTS
```

## Download data

Three datasets are used in the paper (preprocessed data available bellow):

* IRIS data available at [LMSAL](https://iris.lmsal.com/)
* AL data available at [NREL](https://www.nrel.gov/grid/solar-power-data.html)
* PB data available at [CalTrans](https://pems.dot.ca.gov/) or [Zenodo](https://doi.org/10.5281/zenodo.5724362)

To skip the preprocessing part, you may download the preprocessed data instead:

* IRIS preprocessed data (11.9GB) available at [Zenodo](https://doi.org/10.5281/zenodo.7524572)
* AL preprocessed data (34.5MB) available at [Zenodo](https://doi.org/10.5281/zenodo.7674274)
* PB preprocessed data (114.2MB) available at [Zenodo](https://doi.org/10.5281/zenodo.7674366)

You can run `download_data.py` script:

```
$ python download_data.py --data=<data name> --dir=<data dir> --prpr
```

where `<data name>` is `iris`, `al` or `pb`.

Ensure that `<data dir>` and `dataset_address=<data name>_data` parameter of other python files are the same: by default, `<data dir>=dataset_address=<data name>_data`.

Parameter `--prpr` is for preprocessed data, use `--noprpr` if you want to download raw data.

## Download weights

Weights of all forecasting models (IB-MTS, LSTM, IB-LSTM, GRU, IB-GRU, NBeats), of IRIS classifiers and for VGG16 layers used in the loss are available in `h5py` format and can be downloaded with `download_weights.py`:

```
$ python download_weights.py
```

Weights of all forecasting models (IB-MTS, LSTM, IB-LSTM, GRU, IB-GRU, NBeats) are available at [Zenodo](https://doi.org/10.5281/zenodo.7568871) as a compressed .7z format (407.5MB).
By default, they should be downloaded and unziped in the root directory of this project.

Weights for the VGG16 used in the loss are available from an external [Google Drive source](https://drive.google.com/file/d/1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0/view) as a `h5py` format (528MB).
By default, they should be downloaded and unziped in the `vgg_weights` directory of this project.

And weights of the Classifier of solar activities on IRIS data are available at [Zenodo](https://doi.org/10.5281/zenodo.7674521) as a compressed .7z format (46.3MB).
By default, they should be downloaded and unziped in the `classifiers` directory of this project.

## Download test results

You can run `download_testres.py` script:

```
$ python download_testres.py
```

All test results are available at [Zenodo](https://doi.org/10.5281/zenodo.7674553) as a compressed .7z format (108MB).
By default, they should be downloaded and unziped from the root directory of this project.

## Usage

If you would like to directly test the models and skip the preprocessing and training part, follow [From pretrained models and preprocessed data](#From-pretrained-models-and-preprocessed-data).

If you would like to preprocess the data and train the models, follow [From Scratch](#from-scratch).

### From pretrained models and preprocessed data

This part assumes that you downloaded weights and preprocessed data as described in [Download data](#download-data) and [Download weights](#download-weights).

* Generate test results for each model

You can also download test results at Zenodo [Download test results](#Download-test-results) and skip the next line of code. To test your model, you can do it in the following way for iris data and IBMTS model:

```
$ python main.py --model_type=IBMTS --dataset=iris_level_2C --label_length=240 --labels=QS_AR_FL --preload_train --nochange_traindata --test --test_ds=TE_TEL --add_classifier --add_centercount --notrain --nopredict
```

`mode_type` can be `IBMTS`, `LSTMS` (simple LSTM), `LSTM` (named 'IB-LSTM' in the paper), `GRUS` (simple GRU), `GRU` (named 'IB-GRU' in the paper), `NBeats`.

For each 'IRIS', 'AL' or 'PB' evaluated in the paper, here are the corresponding parameters:

***IRIS data:***
>dataset=iris_level_2C\
dataset_address=iris_data\
label_length=240\
labels=QS_AR_FL

***AL data:***
>dataset=al_2C\
dataset_address=al_data\
label_length=137\
labels=AL

***PB data:***
>dataset=pb_2C\
dataset_address=pb_data\
label_length=325\
labels=PB

This may generate a lot of figures and data in the `npz` format: you can also download those results at Zenodo following [Download test results](#Download-test-results).

* Compare models

To generate part of Table 2 and Figures 9, 10:

```
$python compare_mts_plots.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
```

Default paths to model results must be modified in the py file.

To generate other part of Table 2 and Figures 11 and 13:

```
$python compare_mts_plots_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
```

To generate part of Tables 3, 5 and Figures 14, 17:

```
$python compare_cv_plots.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_psnrssim.py --fname=<path for results>
```

To generate other part of Tables 3, 5 and Figures 15, 18:

```
$python compare_cvnn_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_psnrssim_long.py --fname=<path for results>
```

To generate Figure 12:

```
$python durations_counts.py
```

To generate Figure 16 and Table 4:

```
$python compare_centers.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_ib.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_ib_print.py --fname=<path for results>
$python compare_centers_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_ib_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_ib_long_print.py --fname=<path for results>
```

To generate Table 6:

```
$python compare_centers.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_accthss_centers.py --fname=<path for results>
$python compare_centers_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python get_accthss_centerslong.py --fname=<path for results>
```

To generate Figure 19:

```
$python compare_centers.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
```

To generate Table 7:

```
$python get_accthss_class.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python print_act.py --fname=<path for results>
$python get_accthss_class_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python print_act_class_long.py --fname=<path for results>
```

To generate Figure 20:

```
$python compare_feat_plots.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
$python compare_feat_long.py --fname=<path for results> --dataset=<data name> --label_length=<spatial length of data>
```

### From scratch

This part assumes that you downloaded raw or preprocessed data as described in [Download data](#download-data) and that you downloaded VGG16 weights and eventually IRIS Classifiers weights as described in [Download weights](#download-weights).

To train an IBMTS model from scratch on iris data:

```
$ python main.py --model_type=IBMTS --epoch=100 --dataset=iris_level_2C --dataset_address=iris_data --label_length=240 --labels=QS_AR_FL --train --preload_train --nopredict --notest
```

For IRIS data, you must set `preload_train` parameter to `True` because data was already preprocessed and the model should look for this data. When `preload_train` is `True` and no weights are found, it will just start training from scratch. If you start with AL or PB data from scratch (without preprocessing), set `preload_train` to `False` and define the `train_ratio` and `test_ratio` parameters, or set `given_tvt` to `True` if you specified in file names which ones are for train/valid/test.

To test your model, you can do it in the following way for iris data and IBMTS model:

```
$ python main.py --model_type=IBMTS --dataset=iris_level_2C --label_length=240 --labels=QS_AR_FL --preload_train --nochange_traindata --test --test_ds=TE_TEL --add_classifier --add_centercount --notrain --nopredict
```

This may generate a lot of figures and data in the `npz` format. To compare models between then and generate figures present in the paper, follow [From pretrained models and preprocessed data](#From-pretrained-models-and-preprocessed-data).

## Brief explanation

Complete explanation can be found in paper.

![alt text](rdm_img/Evaluations.jpg "Evaluations")


## License
[MIT](https://github.com/DenisUllmann/IB-MTS/blob/main/LICENSE)
