import os

from surfa.system import fatal
from surfa.core.labels import LabelRecoder
from surfa.core.labels import LabelLookup
from surfa.io.labels import load_label_lookup


def home(require=True):
    """
    The freesurfer installation directory as defined by the FREESURFER_HOME environment variable.

    Parameters
    ----------
    require : bool
        If enabled, throws an error when freesurfer home is not set.

    Returns
    -------
    fshome : str
    """
    fshome = os.environ.get('FREESURFER_HOME')
    if require and fshome is None:
        fatal('FREESURFER_HOME has not been set in the environment')
    return fshome


def subjsdir(path=None):
    """
    The freesurfer subjects directory as defined by the SUBJECTS_DIR environment variable.

    Parameters
    ----------
    path : str
        If provided, sets the new SUBJECTS_DIR.

    Returns
    -------
    dir : str
    """
    if path is not None:
        os.environ['SUBJECTS_DIR'] = path
    sdir = os.environ.get('SUBJECTS_DIR')
    if sdir is None:
        fatal('FREESURFER_HOME has not been set in the environment')
    return sdir


def getfile(subpath):
    """
    Retrieve the complete path of a subfile in the freesurfer home directory.

    Parameters
    ----------
    subpath : str
        File path to append to the extracted freesurfer home.

    Returns
    -------
    path : str
    """
    return os.path.join(home(), subpath)


def labels():
    """
    Standard label lookup for all brain regions.

    Returns
    -------
    labels : LabelLookup
    """
    return load_label_lookup(getfile('FreeSurferColorLUT.txt'))


def destrieux():
    """
    Label lookup for Destrieux cortical atlas parcellations.

    Returns
    -------
    labels : LabelLookup
    """
    labels = LabelLookup()
    labels[0]  = ('Unknown',                   [  0,   0,   0])
    labels[1]  = ('G_and_S_frontomargin',      [ 23, 220,  60])
    labels[2]  = ('G_and_S_occipital_inf',     [ 23,  60, 180])
    labels[3]  = ('G_and_S_paracentral',       [ 63, 100,  60])
    labels[4]  = ('G_and_S_subcentral',        [ 63,  20, 220])
    labels[5]  = ('G_and_S_transv_frontopol',  [ 13,   0, 250])
    labels[6]  = ('G_and_S_cingul-Ant',        [ 26,  60,   0])
    labels[7]  = ('G_and_S_cingul-Mid-Ant',    [ 26,  60,  75])
    labels[8]  = ('G_and_S_cingul-Mid-Post',   [ 26,  60, 150])
    labels[9]  = ('G_cingul-Post-dorsal',      [ 25,  60, 250])
    labels[10] = ('G_cingul-Post-ventral',     [ 60,  25,  25])
    labels[11] = ('G_cuneus',                  [180,  20,  20])
    labels[12] = ('G_front_inf-Opercular',     [220,  20, 100])
    labels[13] = ('G_front_inf-Orbital',       [140,  60,  60])
    labels[14] = ('G_front_inf-Triangul',      [180, 220, 140])
    labels[15] = ('G_front_middle',            [140, 100, 180])
    labels[16] = ('G_front_sup',               [180,  20, 140])
    labels[17] = ('G_Ins_lg_and_S_cent_ins',   [ 23,  10,  10])
    labels[18] = ('G_insular_short',           [225, 140, 140])
    labels[19] = ('G_occipital_middle',        [180,  60, 180])
    labels[20] = ('G_occipital_sup',           [ 20, 220,  60])
    labels[21] = ('G_oc-temp_lat-fusifor',     [ 60,  20, 140])
    labels[22] = ('G_oc-temp_med-Lingual',     [220, 180, 140])
    labels[23] = ('G_oc-temp_med-Parahip',     [ 65, 100,  20])
    labels[24] = ('G_orbital',                 [220,  60,  20])
    labels[25] = ('G_pariet_inf-Angular',      [ 20,  60, 220])
    labels[26] = ('G_pariet_inf-Supramar',     [100, 100,  60])
    labels[27] = ('G_parietal_sup',            [220, 180, 220])
    labels[28] = ('G_postcentral',             [ 20, 180, 140])
    labels[29] = ('G_precentral',              [ 60, 140, 180])
    labels[30] = ('G_precuneus',               [ 25,  20, 140])
    labels[31] = ('G_rectus',                  [ 20,  60, 100])
    labels[32] = ('G_subcallosal',             [ 60, 220,  20])
    labels[33] = ('G_temp_sup-G_T_transv',     [ 60,  60, 220])
    labels[34] = ('G_temp_sup-Lateral',        [220,  60, 220])
    labels[35] = ('G_temp_sup-Plan_polar',     [ 65, 220,  60])
    labels[36] = ('G_temp_sup-Plan_tempo',     [ 25, 140,  20])
    labels[37] = ('G_temporal_inf',            [220, 220, 100])
    labels[38] = ('G_temporal_middle',         [180,  60,  60])
    labels[39] = ('Lat_Fis-ant-Horizont',      [ 61,  20, 220])
    labels[40] = ('Lat_Fis-ant-Vertical',      [ 61,  20,  60])
    labels[41] = ('Lat_Fis-post',              [ 61,  60, 100])
    labels[42] = ('Medial_wall',               [ 25,  25,  25])
    labels[43] = ('Pole_occipital',            [140,  20,  60])
    labels[44] = ('Pole_temporal',             [220, 180,  20])
    labels[45] = ('S_calcarine',               [ 63, 180, 180])
    labels[46] = ('S_central',                 [221,  20,  10])
    labels[47] = ('S_cingul-Marginalis',       [221,  20, 100])
    labels[48] = ('S_circular_insula_ant',     [221,  60, 140])
    labels[49] = ('S_circular_insula_inf',     [221,  20, 220])
    labels[50] = ('S_circular_insula_sup',     [ 61, 220, 220])
    labels[51] = ('S_collat_transv_ant',       [100, 200, 200])
    labels[52] = ('S_collat_transv_post',      [ 10, 200, 200])
    labels[53] = ('S_front_inf',               [221, 220,  20])
    labels[54] = ('S_front_middle',            [141,  20, 100])
    labels[55] = ('S_front_sup',               [ 61, 220, 100])
    labels[56] = ('S_interm_prim-Jensen',      [141,  60,  20])
    labels[57] = ('S_intrapariet_and_P_trans', [143,  20, 220])
    labels[58] = ('S_oc_middle_and_Lunatus',   [101,  60, 220])
    labels[59] = ('S_oc_sup_and_transversal',  [ 21,  20, 140])
    labels[60] = ('S_occipital_ant',           [ 61,  20, 180])
    labels[61] = ('S_oc-temp_lat',             [221, 140,  20])
    labels[62] = ('S_oc-temp_med_and_Lingual', [141, 100, 220])
    labels[63] = ('S_orbital_lateral',         [221, 100,  20])
    labels[64] = ('S_orbital_med-olfact',      [181, 200,  20])
    labels[65] = ('S_orbital-H_Shaped',        [101,  20,  20])
    labels[66] = ('S_parieto_occipital',       [101, 100, 180])
    labels[67] = ('S_pericallosal',            [181, 220,  20])
    labels[68] = ('S_postcentral',             [ 21, 140, 200])
    labels[69] = ('S_precentral-inf-part',     [ 21,  20, 240])
    labels[70] = ('S_precentral-sup-part',     [ 21,  20, 200])
    labels[71] = ('S_suborbital',              [ 21,  20,  60])
    labels[72] = ('S_subparietal',             [101,  60,  60])
    labels[73] = ('S_temporal_inf',            [ 21, 180, 180])
    labels[74] = ('S_temporal_sup',            [223, 220,  60])
    labels[75] = ('S_temporal_transverse',     [221,  60,  60])
    return labels


def dkt():
    """
    Label lookup for DKT cortical atlas parcellations.

    Returns
    -------
    labels : LabelLookup
    """
    labels = LabelLookup()
    labels[0]  = ('unknown',                  [ 25,   5,  25])
    labels[1]  = ('bankssts',                 [ 25, 100,  40])
    labels[2]  = ('caudalanteriorcingulate',  [125, 100, 160])
    labels[3]  = ('caudalmiddlefrontal',      [100,  25,   0])
    labels[4]  = ('corpuscallosum',           [120,  70,  50])
    labels[5]  = ('cuneus',                   [220,  20, 100])
    labels[6]  = ('entorhinal',               [220,  20,  10])
    labels[7]  = ('fusiform',                 [180, 220, 140])
    labels[8]  = ('inferiorparietal',         [220,  60, 220])
    labels[9]  = ('inferiortemporal',         [180,  40, 120])
    labels[10] = ('isthmuscingulate',         [140,  20, 140])
    labels[11] = ('lateraloccipital',         [ 20,  30, 140])
    labels[12] = ('lateralorbitofrontal',     [ 35,  75,  50])
    labels[13] = ('lingual',                  [225, 140, 140])
    labels[14] = ('medialorbitofrontal',      [200,  35,  75])
    labels[15] = ('middletemporal',           [160, 100,  50])
    labels[16] = ('parahippocampal',          [ 20, 220,  60])
    labels[17] = ('paracentral',              [ 60, 220,  60])
    labels[18] = ('parsopercularis',          [220, 180, 140])
    labels[19] = ('parsorbitalis',            [ 20, 100,  50])
    labels[20] = ('parstriangularis',         [220,  60,  20])
    labels[21] = ('pericalcarine',            [120, 100,  60])
    labels[22] = ('postcentral',              [220,  20,  20])
    labels[23] = ('posteriorcingulate',       [220, 180, 220])
    labels[24] = ('precentral',               [ 60,  20, 220])
    labels[25] = ('precuneus',                [160, 140, 180])
    labels[26] = ('rostralanteriorcingulate', [ 80,  20, 140])
    labels[27] = ('rostralmiddlefrontal',     [ 75,  50, 125])
    labels[28] = ('superiorfrontal',          [ 20, 220, 160])
    labels[29] = ('superiorparietal',         [ 20, 180, 140])
    labels[30] = ('superiortemporal',         [140, 220, 220])
    labels[31] = ('supramarginal',            [ 80, 160,  20])
    labels[32] = ('frontalpole',              [100,   0, 100])
    labels[33] = ('temporalpole',             [ 70,  20, 170])
    labels[34] = ('transversetemporal',       [150, 150, 200])
    labels[35] = ('insula',                   [255, 192,  32])
    return labels


def tissue_types():
    """
    Label lookup for generic brain tissue types (including skull and head labels).

    Returns
    -------
    labels : LabelLookup
    """
    labels = LabelLookup()
    labels[0] = ('Unknown',                  [0,   0,   0])
    labels[1] = ('Cortex',                   [205, 62,  78])
    labels[2] = ('Subcortical-Gray-Matter',  [230, 148, 34])
    labels[3] = ('White-Matter',             [245, 245, 245])
    labels[4] = ('CSF',                      [120, 18,  134])
    labels[5] = ('Head',                     [150, 150, 200])
    labels[6] = ('Lesion',                   [255, 165,  0])
    return labels


def tissue_type_recoder(extra=False, lesions=False):
    """
    Return a recoding lookup that converts default brain labels to the
    corresponding tissue-type (includes skull and head labels).

    Parameters
    ----------
    extra : bool
        Include extra-cerebral labels, like skull and eye fluid.
    lesions : bool
        Include lesions as a seperate label.

    Returns
    -------
    recoder : LabelRecoder
    """
    mapping = {
        0:    0,  # Unknown
        2:    3,  # Left-Cerebral-White-Matter
        3:    1,  # Left-Cerebral-Cortex
        4:    4,  # Left-Lateral-Ventricle
        5:    4,  # Left-Inf-Lat-Vent
        7:    3,  # Left-Cerebellum-White-Matter
        8:    2,  # Left-Cerebellum-Cortex
        10:   2,  # Left-Thalamus
        11:   2,  # Left-Caudate
        12:   2,  # Left-Putamen
        13:   2,  # Left-Pallidum
        14:   4,  # 3rd-Ventricle
        15:   4,  # 4th-Ventricle
        16:   3,  # Brain-Stem
        17:   2,  # Left-Hippocampus
        18:   2,  # Left-Amygdala
        24:   4,  # CSF
        25:   6 if lesions else 2,  # Left-Lesion
        26:   2,  # Left-Accumbens-Area
        28:   3,  # Left-VentralDC
        30:   4,  # Left-Vessel
        31:   4,  # Left-Choroid-Plexus
        41:   3,  # Right-Cerebral-White-Matter
        42:   1,  # Right-Cerebral-Cortex
        43:   4,  # Right-Lateral-Ventricle
        44:   4,  # Right-Inf-Lat-Vent
        46:   3,  # Right-Cerebellum-White-Matter
        47:   2,  # Right-Cerebellum-Cortex
        49:   2,  # Right-Thalamus
        50:   2,  # Right-Caudate
        51:   2,  # Right-Putamen
        52:   2,  # Right-Pallidum
        53:   2,  # Right-Hippocampus
        54:   2,  # Right-Amygdala
        75:   6 if lesions else 2,  # Right-Lesion
        58:   2,  # Right-Accumbens-Area
        60:   3,  # Right-VentralDC
        62:   4,  # Right-Vessel
        63:   4,  # Right-Choroid-Plexus
        77:   6 if lesions else 3,  # WM-Hypointensities
        78:   3,  # Left-WM-Hypointensities
        79:   3,  # Right-WM-Hypointensities
        80:   2,  # Non-WM-Hypointensities
        81:   2,  # Left-Non-WM-Hypointensities
        82:   2,  # Right-Non-WM-Hypointensities
        85:   3,  # Optic-Chiasm
        99:   6 if lesions else 2,  # Lesion
        130:  5 if extra else 0,  # Air
        165:  5 if extra else 0,  # Skull
        172:  2,  # Vermis
        174:  3,  # Pons
        251:  3,  # CC_Posterior
        252:  3,  # CC_Mid_Posterior
        253:  3,  # CC_Central
        254:  3,  # CC_Mid_Anterior
        255:  3,  # CC_Anterior
        257:  4,  # CSF-ExtraCerebral
        258:  5 if extra else 0,  # Head-ExtraCerebral
        1001: 1,  # ctx-lh-bankssts
        1002: 1,  # ctx-lh-caudalanteriorcingulate
        1003: 1,  # ctx-lh-caudalmiddlefrontal
        1005: 1,  # ctx-lh-cuneus
        1006: 1,  # ctx-lh-entorhinal
        1007: 1,  # ctx-lh-fusiform
        1008: 1,  # ctx-lh-inferiorparietal
        1009: 1,  # ctx-lh-inferiortemporal
        1010: 1,  # ctx-lh-isthmuscingulate
        1011: 1,  # ctx-lh-lateraloccipital
        1012: 1,  # ctx-lh-lateralorbitofrontal
        1013: 1,  # ctx-lh-lingual
        1014: 1,  # ctx-lh-medialorbitofrontal
        1015: 1,  # ctx-lh-middletemporal
        1016: 1,  # ctx-lh-parahippocampal
        1017: 1,  # ctx-lh-paracentral
        1018: 1,  # ctx-lh-parsopercularis
        1019: 1,  # ctx-lh-parsorbitalis
        1020: 1,  # ctx-lh-parstriangularis
        1021: 1,  # ctx-lh-pericalcarine
        1022: 1,  # ctx-lh-postcentral
        1023: 1,  # ctx-lh-posteriorcingulate
        1024: 1,  # ctx-lh-precentral
        1025: 1,  # ctx-lh-precuneus
        1026: 1,  # ctx-lh-rostralanteriorcingulate
        1027: 1,  # ctx-lh-rostralmiddlefrontal
        1028: 1,  # ctx-lh-superiorfrontal
        1029: 1,  # ctx-lh-superiorparietal
        1030: 1,  # ctx-lh-superiortemporal
        1031: 1,  # ctx-lh-supramarginal
        1032: 1,  # ctx-lh-frontalpole
        1033: 1,  # ctx-lh-temporalpole
        1034: 1,  # ctx-lh-transversetemporal
        1035: 1,  # ctx-lh-insula
        2001: 1,  # ctx-rh-bankssts
        2002: 1,  # ctx-rh-caudalanteriorcingulate
        2003: 1,  # ctx-rh-caudalmiddlefrontal
        2005: 1,  # ctx-rh-cuneus
        2006: 1,  # ctx-rh-entorhinal
        2007: 1,  # ctx-rh-fusiform
        2008: 1,  # ctx-rh-inferiorparietal
        2009: 1,  # ctx-rh-inferiortemporal
        2010: 1,  # ctx-rh-isthmuscingulate
        2011: 1,  # ctx-rh-lateraloccipital
        2012: 1,  # ctx-rh-lateralorbitofrontal
        2013: 1,  # ctx-rh-lingual
        2014: 1,  # ctx-rh-medialorbitofrontal
        2015: 1,  # ctx-rh-middletemporal
        2016: 1,  # ctx-rh-parahippocampal
        2017: 1,  # ctx-rh-paracentral
        2018: 1,  # ctx-rh-parsopercularis
        2019: 1,  # ctx-rh-parsorbitalis
        2020: 1,  # ctx-rh-parstriangularis
        2021: 1,  # ctx-rh-pericalcarine
        2022: 1,  # ctx-rh-postcentral
        2023: 1,  # ctx-rh-posteriorcingulate
        2024: 1,  # ctx-rh-precentral
        2025: 1,  # ctx-rh-precuneus
        2026: 1,  # ctx-rh-rostralanteriorcingulate
        2027: 1,  # ctx-rh-rostralmiddlefrontal
        2028: 1,  # ctx-rh-superiorfrontal
        2029: 1,  # ctx-rh-superiorparietal
        2030: 1,  # ctx-rh-superiortemporal
        2031: 1,  # ctx-rh-supramarginal
        2032: 1,  # ctx-rh-frontalpole
        2033: 1,  # ctx-rh-temporalpole
        2034: 1,  # ctx-rh-transversetemporal
        2035: 1,  # ctx-rh-insula
    }
    return LabelRecoder(mapping, target=tissue_types())
