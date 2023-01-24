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

You may create a virtual environnement and use `requirements.txt`:

```
$ pip install -r requirements.txt
```

## Install

Use git to clone this repository into your computer.

```
git clone https://github.com/DenisUllmann/IB-MTS
```

## Download data

Three datasets are used in the paper (preprocessed data available bellow):

* IRIS data available at [LMSAL](https://iris.lmsal.com/)
* AL data available at [Add](add)
* PB data available at [Add](add)

To skip the preprocessing part, you may download the preprocessed data instead:

* IRIS data available at [Zenodo](10.5281/zenodo.7524572)
* AL data available at [Add](add)
* PB data available at [Add](add)


## Download weights

Weights of the IB-MTS model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

Weights of the LSTM model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

Weights of the IB-LSTM model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

Weights of the GRU model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

Weights of the IB-GRU model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

Weights of the Nbeats model:
* On IRIS data [Add](add)
* On AL data [Add](add)
* On PB data [Add](add)

And weights of the Classifier of solar activities on IRIS data: [Add](add)

## Usage

If you would like to preprocess the data and train the models, follow [From Scratch](#from-scratch).

If you would like to directly test the models and skip the preprocessing and training part, follow [From pretrained models and preprocessed data](#From-pretrained-models-and-preprocessed-data).

### From scratch


### From pretrained models and preprocessed data


## Brief explanation

Add


## License
[MIT](https://github.com/DenisUllmann/IB-MTS/blob/main/LICENSE)
