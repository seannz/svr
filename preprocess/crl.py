import os
import glob
import nibabel as nib
import freesurfer as fs

files = [f.split('.')[0] for f in glob.glob("*.nii.gz") if "_" not in f]
destroot = '../CRL_FetalBrainAtlas_2017v3_lia'

for f in files:
    os.system('mri_convert %s.nii.gz temp.nii -vs 0.8 0.8 0.8 --in_orientation LPS --out_orientation LIA' % f)
    os.system('mri_convert %s_tissue.nii.gz temp_tissue.nii -vs 0.8 0.8 0.8 -rt nearest --in_orientation LPS --out_orientation LIA' % f)
    os.system('mri_convert %s_regional.nii.gz temp_regional.nii -vs 0.8 0.8 0.8 -rt nearest --in_orientation LPS --out_orientation LIA' % f)

    bbox = fs.Volume.read('temp.nii').bbox()
    fs.Volume.read('temp.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s.nii.gz' % (destroot, f))
    fs.Volume.read('temp_tissue.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s_tissue.nii.gz' % (destroot, f))
    fs.Volume.read('temp_regional.nii')[bbox].fit_to_shape([256,256,256]).write('%s/%s_regional.nii.gz' % (destroot, f))
