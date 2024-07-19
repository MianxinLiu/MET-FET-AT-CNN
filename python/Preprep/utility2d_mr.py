import numpy as np
import torch
import torchio as tio
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as T

from pathlib import Path
import os
import glob
import models


def extract_bboxes(mask):
    """Compute bounding boxes from masks.

    mask: [height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [(x1, x2, y1, y2)].

    """
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    #print("np.any(m, axis=0)", np.any(m, axis=0))
    #print("p.where(np.any(m, axis=0))", np.where(np.any(m, axis=0)))
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([x1, x2, y1, y2])
    return boxes.astype(np.int32)

pet_type='met'

dataset_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/Internal_pet2mr/'+pet_type+'/'
#dataset_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/test_suvr/processed/'+pet_type+'/'
dataset_dir = os.listdir(dataset_dir_name)
mask_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/mr_mask/'
#mask_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/test_suvr/mask/'+pet_type+'/'
mask_dir = os.listdir(mask_dir_name)

T1_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C_N4/'
T1_dir = os.listdir(T1_dir_name)
T2_dir_name = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T2_reg/'
T2_dir = os.listdir(T2_dir_name)

subInfo = pd.read_csv('/home/PJLAB/liumianxin/Desktop/VIEW_codes/subinfo.csv')

subjects = []
mritransform = tio.Compose([
    tio.ToCanonical(),
    #tio.CropOrPad((288,288,148)),
    tio.CopyAffine('Brain'),
    tio.RescaleIntensity(out_min_max=(0, 100),masking_method='Brain'),
    tio.Mask(masking_method='Brain'),
    tio.Resize((188,188,180)),
    #tio.CropOrPad((188,188,148)),
    #tio.ZNormalization(masking_method='Brain'),
])

mritransform2 = tio.Compose([
    tio.ToCanonical(),
    #tio.CropOrPad((288,288,148)),
    tio.CopyAffine('Brain'),
    tio.RescaleIntensity(out_min_max=(0, 100),masking_method='Brain'),
    tio.Mask(masking_method='Brain'),
    # tio.ZNormalization(masking_method='Brain'),
    tio.Resize((188,188,180)),
    #tio.CropOrPad((188,188,148)),
    #tio.RescaleIntensity(out_min_max=(-10, 10),masking_method='Brain'),
])

masktransform = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((188,188,180)),
    #tio.CropOrPad((188,188,148)),
])

pos=0
for sub in range(len(dataset_dir)):
    mripath = dataset_dir_name + dataset_dir[sub] + '/PET2T1.nii*'
    mripath = glob.glob(mripath)
    t1path = T1_dir_name + dataset_dir[sub] + '/*N4.nii.gz'
    t1path = glob.glob(t1path)
    t2path = T2_dir_name + dataset_dir[sub] + '/*N4.nii.gz'
    t2path = glob.glob(t2path)
    maskpath = mask_dir_name + dataset_dir[sub] + '/mask2PET.nii.gz'
    maskpath = glob.glob(maskpath)
    Brainpath = mask_dir_name + dataset_dir[sub] + '/head_mask.nii'
    Brainpath = glob.glob(Brainpath)

    if len(subInfo['H3K27M'][subInfo['name'].values==dataset_dir[sub]].values):
        pet = tio.ScalarImage(mripath[0])
        t1 = tio.ScalarImage(t1path[0])
        t2 = tio.ScalarImage(t2path[0])
        #pet = mritransform(pet)
        Brain = tio.LabelMap(Brainpath[0])
        temp= tio.Subject(pet=pet, Brain=Brain)
        data = mritransform(temp)
        pet = data['pet']
        temp = tio.Subject(t1=t1, t2=t2, Brain=Brain)
        data = mritransform2(temp)
        t1 = data['t1']
        t2 = data['t2']
        mask = tio.LabelMap(maskpath)
        mask = masktransform(mask)

        subject = tio.Subject(
            pet=pet,
            t1=t1,
            t2=t2,
            mask=mask,
            H3K27M = subInfo['H3K27M'][subInfo['name'].values==dataset_dir[sub]].values,
            Ki67 = subInfo['Ki-67'][subInfo['name'].values == dataset_dir[sub]].values,
        )
        print(subInfo['H3K27M'][subInfo['name'].values==dataset_dir[sub]].values)
        pos=pos+subInfo['H3K27M'][subInfo['name'].values==dataset_dir[sub]].values
        subjects.append(subject)
    else:
        print(dataset_dir[sub])

dataset = tio.SubjectsDataset(subjects)

alldata_set = tio.SubjectsDataset(
    dataset)

print('alldata set:', len(alldata_set), 'subjects')

batch_size=1
data_loader = torch.utils.data.DataLoader(
    alldata_set, batch_size=batch_size)

slicenum=[]
for sub,(data) in enumerate(data_loader):
    print(sub)
    if sub<1:
        nonzero_layer=[]
        for z in range(data['mask'][tio.DATA].shape[-1]):
            if torch.sum(torch.squeeze(data['mask']['data'][:,:,:,:,z]))>100:
                nonzero_layer.append(z)
        petslice=data['pet'][tio.DATA][:,:,:,:,nonzero_layer]
        t1slice = data['t1'][tio.DATA][:, :, :, :, nonzero_layer]
        t2slice = data['t2'][tio.DATA][:, :, :, :, nonzero_layer]
        maskslice = data['mask'][tio.DATA][:, :, :, :, nonzero_layer]
        label1 = torch.zeros([len(nonzero_layer)])
        label2 = torch.zeros([len(nonzero_layer)])
        for z in range(len(nonzero_layer)):
            label1[z] = data['H3K27M']
            label2[z] = data['Ki67']
        slice = torch.squeeze(petslice)
        slice = slice.permute(2,0,1)
        slice = torch.unsqueeze(slice,dim=1)
        t1 = torch.squeeze(t1slice)
        t1 = t1.permute(2, 0, 1)
        t1 = torch.unsqueeze(t1, dim=1)
        t2 = torch.squeeze(t2slice)
        t2 = t2.permute(2, 0, 1)
        t2 = torch.unsqueeze(t2, dim=1)
        slice=torch.cat([slice, t1, t2], dim=1)
        mask = torch.squeeze(maskslice)
        mask = mask.permute(2, 0, 1)
        label1_all = label1
        label2_all = label2
        slice_all = slice
        mask_all = mask
        slicenum.append(len(nonzero_layer))
    if sub>=1:
        nonzero_layer = []
        for z in range(data['mask'][tio.DATA].shape[-1]):
            if torch.sum(torch.squeeze(data['mask']['data'][:, :, :, :, z])) >100:
                nonzero_layer.append(z)
        petslice = data['pet'][tio.DATA][:, :, :, :, nonzero_layer]
        t1slice = data['t1'][tio.DATA][:, :, :, :, nonzero_layer]
        t2slice = data['t2'][tio.DATA][:, :, :, :, nonzero_layer]
        maskslice = data['mask'][tio.DATA][:, :, :, :, nonzero_layer]
        label1 = torch.zeros([len(nonzero_layer)])
        label2 = torch.zeros([len(nonzero_layer)])
        for z in range(len(nonzero_layer)):
            label1[z] = data['H3K27M']
            label2[z] = data['Ki67']
        slice = torch.squeeze(petslice)
        slice = slice.permute(2, 0, 1)
        slice = torch.unsqueeze(slice, dim=1)
        t1 = torch.squeeze(t1slice)
        t1 = t1.permute(2, 0, 1)
        t1 = torch.unsqueeze(t1, dim=1)
        t2 = torch.squeeze(t2slice)
        t2 = t2.permute(2, 0, 1)
        t2 = torch.unsqueeze(t2, dim=1)
        slice = torch.cat([slice, t1, t2], dim=1)
        mask = torch.squeeze(maskslice)
        mask = mask.permute(2, 0, 1)
        label1_all = torch.cat([label1_all, label1], dim=0)
        label2_all = torch.cat([label2_all, label2], dim=0)
        slice_all = torch.cat([slice_all, slice], dim=0)
        mask_all = torch.cat([mask_all, mask], dim=0)
        slicenum.append(len(nonzero_layer))

torch.save({'label1_all':label1_all, 'mask_all':mask_all, 'slice_all': slice_all}, '/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_t1t2_v2.pth')
import scipy.io as scio
scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_count_t1t2_v2.mat', {'slicenum':slicenum})

#torch.save({'label1_all':label1_all, 'mask_all':mask_all, 'slice_all': slice_all}, '/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_mask.pth')


data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_t1t2_v2.pth')

slice_all = data['slice_all']
label1_all = data['label1_all']

newslide_all = torch.zeros([1, 3, 188, 188])
newlabel_all = torch.zeros([1])
for n in range(slice_all.shape[0]):
    mask = torch.squeeze(slice_all[n,1,:,:])
    mask = mask.numpy()
    box = extract_bboxes(mask)
    newslide = slice_all[n,:,box[2]:box[3],box[0]:box[1]]
    if newslide.shape[0]!=0 and newslide.shape[1]!=0:
        #mask = torch.squeeze(newslide[:,:])
        #mask = mask.numpy()
        #print(np.sum(mask))
        transform = T.Resize((188,188))
        newslide_all = torch.cat([newslide_all,transform(torch.unsqueeze(newslide,dim=0))], dim=0)
        newlabel_all = torch.cat([newlabel_all, torch.unsqueeze(label1_all[n], dim=0)], dim=0)
    else:
        print('mask is empty')

slice_all = newslide_all[1:,:,:,:]
label_all = newlabel_all[1:]

torch.save({'label1_all':label1_all, 'slice_all': slice_all}, '/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_crop_t1t2_v2.pth')
