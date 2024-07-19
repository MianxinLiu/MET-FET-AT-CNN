from __future__ import print_function
import six
import os
import torch
import radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import SimpleITK as sitk
import numpy as np

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)

data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/met_slice_testset.pth')
mask_all = data['mask_all']
data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/met_slice_crop_testset.pth')
slice_all = data['slice_all']
label_all = data['label1_all']


feature_all=np.zeros((slice_all.shape[0],93))
for s in range(slice_all.shape[0]):
    image = slice_all[s,:,:]
    image = torch.squeeze(image)
    image=image.numpy()
    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetOrigin((0, 0))
    image_sitk.SetSpacing((0.7, 0.7))

    mask = mask_all[s,:,:]
    mask = torch.squeeze(mask)
    mask=mask.numpy()
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.SetOrigin((0, 0))
    mask_sitk.SetSpacing((0.7, 0.7))

    result = extractor.execute(image_sitk, mask_sitk)

    idx=0
    data=[]
    for key, value in six.iteritems(result):
        if idx >= 22:
            data.append(value.item())
        idx += 1
    feature_all[s,:]=data

np.save('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/met_feature_testset.npy', feature_all)
label=label_all.numpy()
np.save('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/met_label_testset.npy', label)

import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


pet = 'fet'
all_f=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_feature.npy')
label=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_label.npy')

y = label
x = all_f

method = 'lr'
qual_all = []
predicted_all = []
test_y_all = []
predict_p = []
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/'+pet+'_sub/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]
    x_train=x[train_idx,:]
    y_train=y[train_idx]
    x_test = x[test_idx, :]
    y_test = y[test_idx]
    # testx, testy = oversample.fit_resample(testx, testy)

    if method == 'svm':
        # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        clf = svm.SVC(class_weight='balanced', kernel='poly', max_iter=5000, probability=True) # max_iter=5000
        clf.fit(x_train, y_train)
        prey = clf.predict(x_test)
        prey_p = clf.predict_proba(x_test)
    if method == 'rf':
        rfc=RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=100)
        rfc.fit(x_train, y_train)
        prey = rfc.predict(x_test)
        prey_p = rfc.predict_proba(x_test)
    if method == 'lr':
        lgc=LogisticRegression(penalty='elasticnet', l1_ratio=0.5, class_weight='balanced',n_jobs=-1, solver='saga')
        lgc.fit(x_train, y_train)
        prey = lgc.predict(x_test)
        prey_p = lgc.predict_proba(x_test)
    correct = (np.array(prey) == np.array(y_test)).sum()
    accuracy = float(correct) / float(len(y_test))
    sens = metrics.recall_score(y_test, prey, pos_label=1)
    spec = metrics.recall_score(y_test, prey, pos_label=0)
    auc = metrics.roc_auc_score(y_test, prey_p[:,1])

    predicted_all = predicted_all + prey.tolist()
    predict_p = predict_p + prey_p[:,1].tolist()
    test_y_all = test_y_all + y_test.tolist()

    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          )
    qual_all.append([accuracy, sens, spec, auc])

    if method == 'svm':
        scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/'+pet+'_SVM_cv' + str(cv) + '.mat', {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})
    if method == 'rf':
        scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/'+pet+'_RF_cv' + str(cv) + '.mat', {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})
    if method == 'lr':
        scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/'+pet+'_LR_cv' + str(cv) + '.mat', {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})

print(qual_all)
print(np.mean(qual_all, axis=0))
print(np.std(qual_all, axis=0))
scio.savemat('./metrics/'+pet+'_'+method+'.mat', {'qual_all': qual_all})
#scio.savemat('metrics_t1_mask_radiomics.mat', {'qual_all':qual_all})



# external
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


pet='fet'
method = 'rf'
all_f=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_feature.npy')
label=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_label.npy')

y_train = label
x_train = all_f

all_f=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_feature_testset.npy')
label=np.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/radiomics_feature/'+pet+'_label_testset.npy')

y_test = label
x_test = all_f

qual_all = []
predicted_all = []
test_y_all = []
predict_p = []

if method == 'svm':
    # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    clf = svm.SVC(class_weight='balanced', kernel='sigmoid', max_iter=5000, probability=True)  # max_iter=5000
    clf.fit(x_train, y_train)
    prey = clf.predict(x_test)
    prey_p = clf.predict_proba(x_test)
if method == 'rf':
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=10)
    rfc.fit(x_train, y_train)
    prey = rfc.predict(x_test)
    prey_p = rfc.predict_proba(x_test)
if method == 'lr':
    lgc = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, class_weight='balanced', n_jobs=-1, solver='saga')
    lgc.fit(x_train, y_train)
    prey = lgc.predict(x_test)
    prey_p = lgc.predict_proba(x_test)

correct = (np.array(prey) == np.array(y_test)).sum()
accuracy = float(correct) / float(len(y_test))
sens = metrics.recall_score(y_test, prey, pos_label=1)
spec = metrics.recall_score(y_test, prey, pos_label=0)
auc = metrics.roc_auc_score(y_test, prey_p[:, 1])

predicted_all = predicted_all + prey.tolist()
predict_p = predict_p + prey_p[:, 1].tolist()
test_y_all = test_y_all + y_test.tolist()

print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      )
qual_all.append([accuracy, sens, spec, auc])

if method == 'svm':
    scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_' + pet + '_SVM.mat',
                 {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})
if method == 'rf':
    scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_' + pet + '_RF.mat',
                 {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})
if method == 'lr':
    scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_' + pet + '_LR.mat',
                 {'predicted_all': prey, 'predict_p': predict_p, 'test_y_all': y_test})

print(qual_all)
print(np.mean(qual_all, axis=0))
print(np.std(qual_all, axis=0))
