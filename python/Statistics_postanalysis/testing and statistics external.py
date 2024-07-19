from torch.utils.data import Dataset, TensorDataset
import torch
import torch.nn as nn
import models
from sklearn import metrics
from sklearn.model_selection import KFold
from random import shuffle
import scipy.io as scio
import numpy as np
import pandas as pd
import torchvision.transforms as T

import scipy.stats as st

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

pet = 'met'

data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet+'_slice_crop_testset.pth') #testset2 and 3 for consecutive (3 for OSEM FET)

slice_all = data['slice_all']
label_all = data['label1_all']

slice_all = torch.unsqueeze(slice_all,1)

dataset_test = TensorDataset(slice_all, label_all)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32) #505 888

predicted_all = []
predict_p = []
test_y_all = []
model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
model.cuda()

model.load_state_dict(torch.load(
    '/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_test/' + 'model_'+pet+'_transfer.pth'))
model.eval()
with torch.no_grad():
    for sub, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.cuda()
        test_y = test_y.long()
        test_y = test_y.cuda()
        test_output = model(test_x)
        test_y = test_y.long()

        predict_p = predict_p + test_output[:, 1].tolist()
        predicted = torch.max(test_output.data, 1)[1]
        correct = (predicted == test_y).sum()
        accuracy = float(correct) / float(predicted.shape[0])
        test_y = test_y.cpu()
        predicted = predicted.cpu()
        predicted_all = predicted_all + predicted.tolist()
        test_y_all = test_y_all + test_y.tolist()

correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
accuracy = float(correct) / float(len(test_y_all))
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
auc = metrics.roc_auc_score(test_y_all, predict_p)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      )
scio.savemat('./results/Testset_'+pet+'_transfer.mat', {'predicted_all': predicted_all, 'predict_p':predict_p, 'test_y_all': test_y_all})



# testset
import numpy as np
from sklearn.metrics import roc_curve, auc

sub_pred_all = []
sub_pro_all = []
sub_label_all = []

pet = 'met'
method = 'transfer'

mat_file = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/' + pet + '_sub/slice2sub_testset.mat')
slice2sub = mat_file['index_idx']

thesub = slice2sub[0]
uniquesub = np.unique(thesub)

result_file = scio.loadmat(
    '/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_'+pet+ '_' + method + '.mat')
predicted_all = result_file['predict_p']

sub_pred = np.zeros(len(uniquesub))
sub_pro = np.zeros(len(uniquesub))
sub_label = np.zeros(len(uniquesub))

for i, unique_sub in enumerate(uniquesub):
    sub_pos = np.where(thesub == unique_sub)[0]
    sub_label[i] = result_file['test_y_all'][0][sub_pos][1]
    sub_pro[i] = (np.sum(predicted_all[0][sub_pos]) / len(sub_pos))
    if sub_pro[i]>0.5:
        sub_pred[i] = 1
    else:
        sub_pred[i] = 0

sub_pred_all.extend(sub_pred)
sub_pro_all.extend(sub_pro)
sub_label_all.extend(sub_label)

# Save the results
scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_'+pet+'_'+method+'_ind.mat', {'sub_pro_all':sub_pro_all, 'sub_pred_all':sub_pred_all, 'sub_label_all':sub_label_all})


temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_met_transfer_ind.mat')
predicted_all= temp['sub_pred_all'][0]
predict_p= temp['sub_pro_all'][0]
test_y_all= temp['sub_label_all'][0]
correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
accuracy = float(correct) / float(len(test_y_all))
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
auc = metrics.roc_auc_score(test_y_all, predict_p)
f1 = metrics.f1_score(test_y_all, predicted_all)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      '|F1:', f1,
      )

# hold-out set
temp1 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_transfer_ind.mat')
temp2 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_met_transfer_ind_tmp.mat')

predicted_all= np.concatenate([temp1['sub_pred_all'][0],temp2['sub_pred_all'][0]])
predict_p =  np.concatenate([temp1['sub_pro_all'][0],temp2['sub_pro_all'][0]])
test_y_all=  np.concatenate([temp1['sub_label_all'][0],temp2['sub_label_all'][0]])

# consecutive test set
# temp1 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_transfer2_ind2.mat')
# temp2 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_transfer3_ind3.mat') # OSEM
# temp3 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_met_transfer2_ind2.mat')
#
# predicted_all= np.concatenate([temp1['sub_pred_all'][0],temp2['sub_pred_all'][0], temp3['sub_pred_all'][0]])
# predict_p =  np.concatenate([temp1['sub_pro_all'][0],temp2['sub_pro_all'][0], temp3['sub_pro_all'][0]])
# test_y_all=  np.concatenate([temp1['sub_label_all'][0],temp2['sub_label_all'][0], temp3['sub_label_all'][0]])


correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
accuracy = float(correct) / float(len(test_y_all))
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
auc = metrics.roc_auc_score(test_y_all, predict_p)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      )
threshold = Find_Optimal_Cutoff(test_y_all,predict_p)
print(threshold)

scio.savemat('./results/ConTestset_'+pet+'_transfer.mat', {'predicted_all': predicted_all, 'predict_p':predict_p, 'test_y_all': test_y_all})

import compare_auc_delong_xu as delong
temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_test_ind.mat')
label= temp['sub_label_all'][0]
pred1= temp['sub_pred_all'][0]

temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_transfer_ind.mat')
pred2= temp['sub_pred_all'][0]
p=delong.delong_roc_test(label, pred1, pred2)

from scipy.special import exp10
print(p/2)


import compare_auc_delong_xu as delong
temp1 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_LR_ind.mat')
temp2 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_met_LR_ind.mat')

pred1 = np.concatenate([temp1['sub_pro_all'][0],temp2['sub_pro_all'][0]])
label = np.concatenate([temp1['sub_label_all'][0],temp2['sub_label_all'][0]])
auc = metrics.roc_auc_score(label, pred1)
print(auc)

temp1 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_fet_transfer_ind.mat')
temp2 = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/Testset_met_transfer_ind.mat')

pred2 =  np.concatenate([temp1['sub_pro_all'][0],temp2['sub_pro_all'][0]])
auc = metrics.roc_auc_score(label, pred2)
print(auc)

p=delong.delong_roc_test(label, pred1, pred2)

from scipy.special import exp10
print(p/2)
auc_1, auc_cov_1 = delong.delong_roc_variance(label, pred1)
auc_2, auc_cov_2 = delong.delong_roc_variance(label, pred2)



## Clinical feasibility test

data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/fet_slice_crop_nogene.pth')

slice_all = data['slice_all']

slice_all = torch.unsqueeze(slice_all,1)

dataset_test = TensorDataset(slice_all)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)

predicted_all = []
predict_p = []

model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
model.cuda()
model.load_state_dict(torch.load(
    '/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_test/' + 'model_fet_transfer.pth'))
model.eval()

with torch.no_grad():
    for sub, (test_x) in enumerate(test_loader):
        test_x = test_x[0].cuda()
        test_output = model(test_x)

        predict_p = predict_p + test_output[:, 1].tolist()
        predicted = torch.max(test_output.data, 1)[1]
        predicted = predicted.cpu()
        predicted_all = predicted_all + predicted.tolist()

scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/nogene_fet_transfer.mat', {'predicted_all': predicted_all, 'predict_p':predict_p})

import numpy as np
from sklearn.metrics import roc_curve, auc

sub_pred_all = []
sub_pro_all = []

pet = 'fet'
mat_file = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/' + pet + '_sub/slice2sub_nogene.mat')
slice2sub = mat_file['index_idx']

thesub = slice2sub[0]
uniquesub = np.unique(thesub)

result_file = scio.loadmat(
    '/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/nogene_fet_transfer.mat')
predicted_all = result_file['predict_p']

sub_pred = np.zeros(len(uniquesub))
sub_pro = np.zeros(len(uniquesub))

for i, unique_sub in enumerate(uniquesub):
    sub_pos = np.where(thesub == unique_sub)[0]
    sub_pro[i] = (np.sum(predicted_all[0][sub_pos]) / len(sub_pos))
    if sub_pro[i]>0.5:
        sub_pred[i] = 1
    else:
        sub_pred[i] = 0

sub_pred_all.extend(sub_pred)
sub_pro_all.extend(sub_pro)
print(sub_pro_all)