# IB-MTS

Code of the paper "Multi variables time series information bottleneck"

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

Use `requirements.txt`

## Install

Use git to clone this repository into your computer.

```
git clone https://github.com/DenisUllmann/IB-MTS
```

## Download data


## Download weights


## Usage

Use the well known command to copy the template

```bash
# Copy the content
CTRL + C

# Pase into your project
CTRL + V
```

## Brief explanation

Here starts the main content of your README. This is why you did it for in the first place.
To describe to future users of this project (including yourself) everything they need to know
to be able to use it and understand it.

Use visuals to help the reader understand better. An image, diagram, chart or code example says
more than thousand words

![Diagram](doc/diagram.jpg)


## License
[MIT](https://choosealicense.com/licenses/mit/)
