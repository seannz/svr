<h1>Fully Convolutional Slice-to-Volume Reconstruction for Single-Stack MRI
<img width="980" alt="Screenshot 2024-03-26 at 8 30 18 PM" src="https://github.com/seannz/svr/assets/1659747/233b3f47-d10a-4e2d-ae6b-56c8bdb939a4"></h1>

This is the official repo for the CVPR 2024 paper "Fully Convolutional Slice-to-Volume Reconstruction (FC-SVR) for Single-Stack MRI" by Sean I Young, Yaël Balbastre, and others. See https://arxiv.org/abs/2312.03102.

<h2>Pre-requisites</h2>
pytorch=1.13.1</br>
torch-interpol=0.2.3</br>
cornucopia=2.0</br>
nibabel=5.0.1</br>
</br>
Also, FreeSurfer version 7 is required to prepare the training dataset. See https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads</br>

<h2>Dataset Preparation</h2>
The Havard CRL fetal atlases can be downloaded from http://crl.med.harvard.edu/research/fetal_brain_atlas. Preprocess the data using preprocess/crl.py </br>
The FeTA training and validation volumes can be downloaded from https://doi.org/10.7303/syn25649159. Preprocess the data using preprocess/feta.py </br>

<h2>Training</h2>
Run feta3d_svr_train.sh to train the svr model on the FeTA 2.1 data. Run feta3d_inpaint_train.sh to train the interpolation model.
