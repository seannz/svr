<h1>Fully Convolutional Slice-to-Volume Reconstruction for Single-Stack MRI</h1>

This is the official repo for the CVPR 2024 paper "Fully Convolutional Slice-to-Volume Reconstruction for Single-Stack SVR (FC-SVR). 

<h2>Pre-requisites</h2>
pytorch=1.13.1</br>
torch-interpol=0.2.3</br>
cornucopia=2.0</br>
nibabel=5.0.1</br>
</br>
Also, FreeSurfer version 7 is required to prepare the training dataset. See https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads</br>

<h2>Dataset Preparation</h2>
The Havard CRL fetal atlases can be downloaded from http://crl.med.harvard.edu/research/fetal_brain_atlas. Preprocess the data using preprocess/crl.py </br>
The FeTA training and validation volumes can be downloaded from http://neuroimaging.ch/feta. Preprocess the data using preprocess/feta.py </br>

<h2>Training</h2>
Run feta3d_svr_train.sh to train the svr model on the FeTA 2.1 data. Run feta3d_inpaint_train.sh to train the interpolation model.
