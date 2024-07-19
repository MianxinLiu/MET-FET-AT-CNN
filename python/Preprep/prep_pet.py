import os
import numpy as np
import ants
import glob
import numpy.ma as ma
import scipy.stats as stats

datapath='/media/PJLAB\liumianxin/18675978328/Glioma_MET/fet/'
sublist=os.listdir(datapath)
maskpath = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/template/'

# datapath='/media/PJLAB\liumianxin/18675978328/Glioma_MET/case_data/data/'
# sublist=os.listdir('/media/PJLAB\liumianxin/18675978328/Glioma_MET/case_data/data/')
# maskpath = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/template/'


# datapath='/media/PJLAB\liumianxin/T7 Shield/Mianxin/Glioma_new/OSEM_raw/'
# sublist=os.listdir(datapath)
# maskpath = '/media/PJLAB\liumianxin/T7 Shield/Mianxin/Glioma_new/template/'

for i in range(len(sublist)):
    PETfold = glob.glob(datapath+sublist[i]+ '/*.nii')
    if not len(PETfold):
        print(sublist[i])
    save_path = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/case_data/data_suvr/' + sublist[i]
    # save_path = '/media/PJLAB\liumianxin/T7 Shield/Mianxin/Glioma_new/OSEM_suvr/'
    if not os.path.exists(save_path+'mask/fet/'+ sublist[i][0:-4]) and len(PETfold):
        os.mkdir(save_path+ sublist[i] )
        fix_img = ants.image_read(PETfold[0])
        mov_img = ants.image_read(maskpath+'ch2.nii.gz')
        transformants = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='Affine')

        #register and save
        mov_img = ants.image_read(maskpath + 'ch2bet.nii.gz')
        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                             transformlist=transformants['fwdtransforms'], interpolator='bSpline')
        save_img_path = save_path + '/template2T1.nii.gz'
        ants.image_write(reg_img,save_img_path)
        
        headmask = reg_img>5
        save_img_path = save_path +  '/head_mask.nii'
        ants.image_write(headmask, save_img_path)

        mov_img = ants.image_read(maskpath + 'mirror_roi.nii')
        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                        transformlist=transformants['fwdtransforms'], interpolator='nearestNeighbor')
        save_img_path = save_path + '/mask2PET.nii.gz'
        ants.image_write(reg_img, save_img_path)
        
        mov_img = ants.image_read(maskpath + 'CS_re.nii')
        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                        transformlist=transformants['fwdtransforms'], interpolator='nearestNeighbor')
        save_img_path = save_path + '/aal2PET.nii.gz'
        ants.image_write(reg_img, save_img_path)
        CS = reg_img.numpy()
        PET = fix_img
        mask = np.ones(CS.shape)
        mask[CS==1] = 0
        mx = ma.masked_array(PET.numpy(), mask=mask)
        base = mx.mean()
        suvr = PET / base
        save_img_path = save_path +'processed/fet/'+ sublist[i][0:-4] + '/PET_FBP_suvr.nii'
        ants.image_write(suvr, save_img_path)



datapath='/media/PJLAB\liumianxin/18675978328/Glioma_MET/'
sublist=os.listdir('/media/PJLAB\liumianxin/18675978328/Glioma_MET/Internal_pet2mr/met/')

for i in range(len(sublist)):
    T1fold = glob.glob(datapath+ 'T1C/' + sublist[i]+ '/*T1*.nii')
    PETfold = glob.glob(datapath + 'test_suvr/processed/met/' + sublist[i] + '/*FBP*.nii')
    if not len(PETfold):
        print(sublist[i])
    save_path = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/Internal_pet2mr/met/' + sublist[i]
    if not os.path.exists(save_path) and len(PETfold):
        os.mkdir(save_path)
        fix_img = ants.image_read(T1fold[0])
        mov_img = ants.image_read(PETfold[0])
        transformants = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='Affine')

        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                             transformlist=transformants['fwdtransforms'], interpolator='bSpline')
        save_img_path = save_path + '/PET2T1.nii.gz'
        ants.image_write(reg_img,save_img_path)

datapath='/media/PJLAB\liumianxin/18675978328/Glioma_MET/'
sublist=os.listdir('/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C/')

for i in range(len(sublist)):
    T1fold = glob.glob(datapath+ 'T1C/' + sublist[i]+ '/*T1*.nii')
    T2fold = glob.glob(datapath + 'T2FLAIR/' + sublist[i] + '/*T2*.nii')
    if not len(T2fold):
        print(sublist[i])
    save_path = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T2_reg/' + sublist[i]
    if not os.path.exists(save_path) and len(T2fold):
        os.mkdir(save_path)
        fix_img = ants.image_read(T1fold[0])
        mov_img = ants.image_read(T2fold[0])
        transformants = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='Affine')

        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                             transformlist=transformants['fwdtransforms'], interpolator='bSpline')
        save_img_path = save_path + '/T2_reg.nii.gz'
        ants.image_write(reg_img,save_img_path)


datapath='/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C/'
sublist=os.listdir('/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1C/')
maskpath = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/template/'
for i in range(len(sublist)):
    PETfold = glob.glob(datapath+sublist[i]+ '/*.nii')
    if not len(PETfold):
        print(sublist[i])
    save_path = '/media/PJLAB\liumianxin/18675978328/Glioma_MET/T1_reg/' + sublist[i]
    if not os.path.exists(save_path) and len(PETfold):
        os.mkdir(save_path)
        mov_img = ants.image_read(PETfold[0])
        fix_img = ants.image_read(maskpath+'ch2.nii.gz')
        transformants = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='Affine')

        reg_img = ants.apply_transforms(fixed=fix_img, moving=mov_img,
                                             transformlist=transformants['fwdtransforms'], interpolator='bSpline')
        save_img_path = save_path + '/T12template.nii.gz'
        ants.image_write(reg_img,save_img_path)


AAL = ants.image_read('/home/PJLAB/liumianxin/Desktop/atlas_upload/aal.nii.gz')
atlas = ants.image_read('/home/PJLAB/liumianxin/Desktop/atlas_upload/MET_ref/CS.nii')
re_atlas = ants.resample_image_to_target(atlas, AAL, 'nearestNeighbor', 0)
ants.image_write(re_atlas, '/home/PJLAB/liumianxin/Desktop/atlas_upload/MET_ref/CS_re.nii')