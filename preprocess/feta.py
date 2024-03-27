import os
import pandas as pd

dst_root = '../CRL_FetalBrainAtlas_2017v3_lia'
src_root = '../feta_2.1'
mov_root = '../feta_2.1_mial'

table = pd.read_csv(os.path.join(src_root, 'participants.tsv'), sep='\t')
subs = np.array(table.participant_id)
ages = np.int8(np.clip(np.round(np.array(table['Gestational age'])),a_min=21,a_max=None))
recs = np.concatenate([np.repeat(np.array('mial'),[40,]),np.repeat(np.array('irtk'),[40,]),np.repeat(np.array('nmic'),[40,])])

for i in range(len(subs)):
    src = '%s/STA%d.nii.gz' % (dst_root, ages[i])
    dst = '%s/%s/anat/%s_rec-%s_T2w_norm.nii' % (src_root, subs[i], subs[i], recs[i]) # use norm only for registration purposes

    command = "mri_robust_register --mov %s --dst %s --lta temp.lta --satit --nosym --iscale --affine" % (src, dst)
    os.system(command)

    command = "mkdir -p %s/%s/anat" % (mov_root, subs[i])
    os.system(command)

    dst = '%s/STA%d.nii.gz' % (dst_root, ages[i])
    src = '%s/%s/anat/%s_rec-%s_T2w.nii' % (src_root, subs[i], subs[i], recs[i])
    sav = '%s/%s/anat/%s_rec-%s_T2w_reg.nii' % (mov_root, subs[i], subs[i], recs[i])
    command = "SUBJECTS_DIR='' mri_vol2vol --mov %s --targ %s --lta-inv temp.lta --o %s" % (src, dst, sav)
    os.system(command)

    dst = '%s/STA%d.nii.gz' % (dst_root, ages[i])
    src = '%s/%s/anat/%s_rec-%s_T2w_norm.nii' % (src_root, subs[i], subs[i], recs[i])
    sav = '%s/%s/anat/%s_rec-%s_T2w_norm_reg.nii' % (mov_root, subs[i], subs[i], recs[i])
    command = "SUBJECTS_DIR='' mri_vol2vol --mov %s --targ %s --lta-inv temp.lta --o %s" % (src, dst, sav)
    os.system(command)

    dst = '%s/STA%d_regional.nii.gz' % (dst_root, ages[i])
    src = '%s/%s/anat/%s_rec-%s_dseg.nii' % (src_root, subs[i], subs[i], recs[i])
    sav = '%s/%s/anat/%s_rec-%s_dseg_reg.nii' % (mov_root, subs[i], subs[i], recs[i])
    command = "SUBJECTS_DIR='' mri_vol2vol --mov %s --targ %s --lta-inv temp.lta --o %s --nearest" % (src, dst, sav)
    os.system(command)
