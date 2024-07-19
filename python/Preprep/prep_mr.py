import pandas as pd
import ants
import os
import glob
import shutil
import tensorflow as tf
import tensorrt as trt
import antspynet

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


pet_type='fet'

dataset_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/Internal_pet2mr/'+pet_type+'/'
#dataset_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/test_suvr/processed/'+pet_type+'/'
dataset_dir = os.listdir(dataset_dir_name)
mask_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/mr_mask/'
#mask_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/test_suvr/mask/'+pet_type+'/'
mask_dir = os.listdir(mask_dir_name)

T1_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C/'
T1_dir = os.listdir(T1_dir_name)
T2_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T2_reg/'
T2_dir = os.listdir(T2_dir_name)

subInfo = pd.read_csv('/home/PJLAB/liumianxin/Desktop/VIEW_codes/subinfo.csv')

T1_out_dir = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C_N4/'

for sub in range(len(dataset_dir)):
    t1path = T1_dir_name + dataset_dir[sub] + '/*T1*.nii*'
    t1path = glob.glob(t1path)
    t2path = T2_dir_name + dataset_dir[sub] + '/T2_reg.nii*'
    t2path = glob.glob(t2path)

    T1 = ants.image_read(t1path[0])
    T2 = ants.image_read(t2path[0])

    T1_corr = ants.n4_bias_field_correction(T1)
    T2_corr = ants.n4_bias_field_correction(T2)

    if not os.path.exists(T1_out_dir + dataset_dir[sub]):
        os.mkdir(T1_out_dir + dataset_dir[sub])

    save_img_path = T1_out_dir + dataset_dir[sub] + '/T1_N4.nii.gz'
    ants.image_write(T1_corr, save_img_path)

    save_img_path = T2_dir_name + dataset_dir[sub] + '/T2_N4.nii.gz'
    ants.image_write(T2_corr, save_img_path)

    probability_brain_mask = antspynet.brain_extraction(T1_corr, modality="t1")
    probability_brain_mask = probability_brain_mask > 0.5
    bet_image = T1_corr * probability_brain_mask
    save_img_path = T1_out_dir + dataset_dir[sub] + '/bet_mask.nii.gz'
    ants.image_write(probability_brain_mask, save_img_path)
    save_img_path = T1_out_dir + dataset_dir[sub] + '/T1C_N4_bet.nii.gz'
    ants.image_write(bet_image, save_img_path)

