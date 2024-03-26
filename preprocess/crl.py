import os
import glob
import nibabel as nib
import freesurfer as fs


crl_src = '../CRL_FetalBrainAtlas_2017v3' # folder to read from
crl_dest = '../CRL_FetalBrainAtlas_2017v3_lia' # folder to save to

files = [os.path.basename(f).split('.')[0] for f in glob.glob(os.path.join(crl_src, "*.nii.gz")) if "_" not in os.path.basename(f)]

for f in files:
    os.system('mri_convert %s/%s.nii.gz temp.nii -vs 0.8 0.8 0.8 --in_orientation LPS --out_orientation LIA' % (crl_src, f))
    os.system('mri_convert %s/%s_tissue.nii.gz temp_tissue.nii -vs 0.8 0.8 0.8 -rt nearest --in_orientation LPS --out_orientation LIA' % (crl_src, f))
    os.system('mri_convert %s/%s_regional.nii.gz temp_regional.nii -vs 0.8 0.8 0.8 -rt nearest --in_orientation LPS --out_orientation LIA' % (crl_src, f))

    bbox = fs.Volume.read('temp.nii').bbox()
    fs.Volume.read('temp.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s.nii.gz' % (crl_dest, f))
    fs.Volume.read('temp_tissue.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s_tissue.nii.gz' % (crl_dest, f))
    fs.Volume.read('temp_regional.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s_regional.nii.gz' % (crl_dest, f))

    os.system('rm temp*.nii')
