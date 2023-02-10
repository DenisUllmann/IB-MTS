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
* LSTM, GRU
* IB-LSTM, IB-GRU that are made of 2 LSTMs or GRUs to encode and 2 LSTMs or GRUs to decode
* [NBeats](https://arxiv.org/abs/1905.10437)


## Contents

*   [Requirements](#requirements)
*   [Install](#install)
*   [Download data](#download-data)
*   [Download weights](#download-weights)
*   [Usage](#usage)
*   [Brief explanation](#brief-explanation)
*   [License](#license)

## Requirements

Check your available memory, used and downloaded datasets may be significantly big.

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
* PB data available at [CalTrans](https://pems.dot.ca.gov/) or [Zenodo](10.5281/zenodo.5724362)

To skip the preprocessing part, you may download the preprocessed data instead:

* IRIS data available at [Zenodo](10.5281/zenodo.7524572)
* AL data available at [Add](add)
* PB data available at [Add](add)

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

Weights of all forecasting models (IB-MTS, LSTM, IB-LSTM, GRU, IB-GRU, NBeats) are available at [Zenodo](10.5281/zenodo.7568871) as a compressed .7z format.
By default, they should be downloaded and unziped in the root directory of this project.

Weights for the VGG16 used in the loss are available from an external [Google Drive source](https://drive.google.com/file/d/1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0/view) as a `h5py` format.
By default, they should be downloaded and unziped in the `vgg_weights` directory of this project.

And weights of the Classifier of solar activities on IRIS data are available at [Zenodo](add) as a compressed .7z format.
By default, they should be downloaded and unziped in the `classifiers` directory of this project.

## Download test results



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

+ IRIS data:
>dataset=iris_level_2C\
dataset_address=iris_data\
label_length=240\
labels=QS_AR_FL

AL data:\
>dataset=al_2C\
dataset_address=al_data\
label_length=137\
labels=AL

PB data:\
>dataset=pb_2C\
dataset_address=pb_data\
label_length=325\
labels=PB

This may generate a lot of figures and data in the `npz` format: you can also download those results at Zenodo following [Download test results](#Download-test-results).

* Compare models

To generate Table 2:

To generate Figure 9:

To generate Figure 10:

To generate Figure 11:

To generate Figure 12:

To generate Table 3:

To generate Figure 13:

To generate Figure 14:

To generate Figure 15:

To generate Figure 16:

To generate Table 4:

To generate Table 5:

To generate Figure 17:

To generate Figure 18:

To generate Table 6:

To generate Figure 19:

To generate Table 7:

To generate Figure 20:


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

Add


## License
[MIT](https://github.com/DenisUllmann/IB-MTS/blob/main/LICENSE)
