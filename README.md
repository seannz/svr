<h1>[CVPR2024] Fully Convolutional Slice-to-Volume Reconstruction for Single-Stack MRI
<img width="980" alt="Screenshot 2024-03-26 at 8 30 18 PM" src="https://github.com/seannz/svr/assets/1659747/233b3f47-d10a-4e2d-ae6b-56c8bdb939a4"></h1>

This is the official repo for the CVPR 2024 paper "Fully Convolutional Slice-to-Volume Reconstruction (FC-SVR) for Single-Stack MRI" by [Sean I Young](https://seaniyoung.com), Yaël Balbastre, Bruce Fischl and others. See https://arxiv.org/abs/2312.03102.

<h2>Pre-requisites</h2>

All pre-requisite python packages are listed in `pytorch_1.13.1.yml`. Run `conda env create -f pytorch_1.13.1.yml`.
Also, FreeSurfer version 7 is required to prepare the training dataset. See https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads</br>

<h2>Dataset Preparation</h2>

The Havard CRL fetal atlases can be downloaded from http://crl.med.harvard.edu/research/fetal_brain_atlas. Preprocess the data using `preprocess/crl.py` </br>
The FeTA training and validation volumes can be downloaded from https://doi.org/10.7303/syn25649159. Preprocess the data using `preprocess/feta.py` </br>

<h2>Training</h2>

Run `feta3d_svr_train.sh` to train the svr model on the FeTA 2.1 data. Run `feta3d_inpaint_train.sh` to train the interpolation model.

<h2>Pretrained Weights</h2>

The pretrained weights for motion estimation and interpolation networks will be posted here soon!  In the mean time, I can send them to you if you email me at `siyoung` at `mit` dot `edu` 

<h2>Work with us!</h2>

Feel free to reach out to me via e-mail and say hello if you have interesting ideas for  extensions, applications or if you simply just want to chat!
