<h1>[CVPR2024] Fully convolutional slice-to-volume reconstruction for single-stack MRI

[![arXiv](https://img.shields.io/badge/arXiv-2312.03102-b31b1b.svg)](https://arxiv.org/abs/2312.03102)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="1000" alt="Screenshot 2024-06-18 at 9 59 33 AM" src="https://github.com/seannz/svr/assets/1659747/ed3a7684-15db-4975-8ff7-53dd6b8eaaf1"></h1>


This is the official repo for the CVPR 2024 paper "Fully Convolutional Slice-to-Volume Reconstruction (FC-SVR) for Single-Stack MRI" by [Sean I Young](https://seaniyoung.com), Yaël Balbastre, Bruce Fischl and others.

<h2>Pre-requisites</h2>

All pre-requisite python packages are listed in `pytorch_1.13.1.yml`. Run `conda env create -f pytorch_1.13.1.yml`.
Also, FreeSurfer version 7 is required to prepare the training dataset. See https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads</br>

<h2>Dataset Preparation</h2>

The Havard CRL fetal atlases can be downloaded from http://crl.med.harvard.edu/research/fetal_brain_atlas. Preprocess the data using `preprocess/crl.py` </br>
The FeTA training and validation volumes can be downloaded from https://doi.org/10.7303/syn25649159. Preprocess the data using `preprocess/feta.py` </br>

<h2>Training</h2>

Run `feta3d_svr_train.sh` to train the svr model on the FeTA 2.1 data. Run `feta3d_inpaint_train.sh` to train the interpolation model.

<h2>Inference</h2>
<img width="1000" alt="Screenshot 2024-06-18 at 10 04 07 AM" src="https://github.com/seannz/svr/assets/1659747/1a42eabd-9ccf-42cc-8e68-8b71536bba5a">

<h2>Pretrained Weights</h2>

The pretrained weights for motion estimation and interpolation networks will be posted here soon!  In the mean time, I can send them to you if you email me at `siyoung` at `mit` dot `edu` 

<h2>Work with us!</h2>

Feel free to reach out to me via e-mail and say hello if you have interesting ideas for  extensions, applications or if you simply just want to chat!
