from torch.utils.data import Dataset, TensorDataset
import torch
import torch.nn as nn
import models
from sklearn import metrics
from sklearn.model_selection import KFold
from random import shuffle
import scipy.io as scio
import numpy as np
import torchvision.transforms as T

import scipy.stats as st


# Slice metrics
pet_type = 'fet'
posfix = 'transfer'
data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_crop.pth')

slice_all = data['slice_all']
label1_all = data['label1_all']
slice_all = torch.unsqueeze(slice_all,1)

y = label1_all
x = slice_all

qual_all = []
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/'+pet_type+'_sub/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]
    qualified = []
    dataset_test = TensorDataset(x[test_idx, :, :], y[test_idx])
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
    model.cuda()
    model.load_state_dict(torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_sub/'+ 'model_cv' + str(cv) + '_'+pet_type+'_'+posfix+'.pth'))
    # model.load_state_dict(torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_sub/' 'model_cv' + str(cv) + '_met_fet_directmix.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []
    model.eval()
    with torch.no_grad():
        for sub, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.cuda()
            test_y = test_y.long()
            test_y = test_y.cuda()
            test_output = model(test_x) 
            test_y = test_y.long()

            predicted = torch.max(test_output.data, 1)[1]
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
    # test_auc.append(accuracy)
    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
    # sen.append(sens)
    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
    # spe.append(spec)
    auc = metrics.roc_auc_score(test_y_all, predict_p)
    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          )
    qual_all.append([accuracy, sens, spec, auc])
    scio.savemat('./results/'+pet_type+'_'+posfix+'_cv'+str(cv)+'.mat', {'predicted_all': predicted_all, 'predict_p':predict_p, 'test_y_all':test_y_all})

print(qual_all)
print(np.mean(qual_all, axis=0))
print(np.std(qual_all, axis=0))
scio.savemat('./metrics/'+pet_type+'_'+posfix+'.mat', {'qual_all': qual_all})
performance=qual_all
st.t.interval(alpha=0.95, df=len(qual_all)-1, loc=np.mean(qual_all, axis=0), scale=st.sem(qual_all, axis=0))


## Ind metrics
temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/fet_LR_ind.mat')
predicted_p= temp['sub_pro_all'][0]
predicted_all= temp['sub_pred_all'][0]
test_y_all= temp['sub_label_all'][0]
correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
accuracy = float(correct) / float(len(test_y_all))
# test_auc.append(accuracy)
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
# sen.append(sens)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
# spe.append(spec)
auc = metrics.roc_auc_score(test_y_all, predicted_p)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      )


# Delong's test
import compare_auc_delong_xu as delong
temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/fet_LR_ind.mat')
label= temp['sub_label_all'][0]
pred1= temp['sub_pro_all'][0]

temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/fet_transfer_ind.mat')
pred2= temp['sub_pro_all'][0]
p=delong.delong_roc_test(label, pred1, pred2)

from scipy.special import exp10
print(p/2)


## Internal & t1t2

import numpy as np
from sklearn.metrics import roc_curve, auc

sub_pred_all = []
sub_pro_all = []
sub_label_all = []

pet = 'fet'
method = 'LR'

for cv in range(1, 6):
    # mat_file = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/'+pet+'_t1t2_v2/slice2sub.mat')
    mat_file = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/' + pet + '_sub/slice2sub.mat')
    slice2sub = mat_file['index_idx']

    shuffled_index_cv = scio.loadmat(
        # '/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/'+pet+'_t1t2_v2/shuffled_index_cv'+str(cv)+'.mat')
        '/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/' + pet + '_sub/shuffled_index_cv' + str(cv) + '.mat')
    test_idx = shuffled_index_cv['test_idx']

    thesub = slice2sub[0,test_idx]
    uniquesub = np.unique(thesub)

    # result_file = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/T1T2_'+method+'_cv'+str(cv)+'.mat')
    result_file = scio.loadmat(
        '/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/'+pet+ '_' + method + '_cv' + str(cv) + '.mat')
    predicted_all = result_file['predict_p']

    sub_pred = np.zeros(len(uniquesub))
    sub_pro = np.zeros(len(uniquesub))
    sub_label = np.zeros(len(uniquesub))

    for i, unique_sub in enumerate(uniquesub):
        sub_pos = np.where(thesub == unique_sub)[1]
        sub_label[i] = result_file['test_y_all'][0][sub_pos[0]]
        sub_pro[i] = (np.sum(predicted_all[0][sub_pos]) / len(sub_pos))
        if sub_pro[i]>0.5:
            sub_pred[i] = 1
        else:
            sub_pred[i] = 0

    sub_pred_all.extend(sub_pred)
    sub_pro_all.extend(sub_pro)
    sub_label_all.extend(sub_label)

# Save the results
# scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/T1T2_'+method+'_ind.mat', {'sub_pro_all':sub_pro_all, 'sub_pred_all':sub_pred_all, 'sub_label_all':sub_label_all})
scio.savemat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/results/'+pet+'_'+method+'_ind.mat', {'sub_pro_all':sub_pro_all, 'sub_pred_all':sub_pred_all, 'sub_label_all':sub_label_all})

# # Calculate accuracy
accuracy = np.sum(np.array(sub_pred_all) == np.array(sub_label_all)) / len(sub_pred_all)
print(f'Accuracy: {accuracy}')
#
# # Calculate AUC
# fpr, tpr, _ = roc_curve(sub_label_all, sub_pro_all, pos_label=1)
# roc_auc = auc(fpr, tpr)
# print(f'AUC: {roc_auc}')

pet_type='fet'
input = 't2_addition'

data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_crop_t1t2_v2.pth')

slice_all = data['slice_all']
label1_all = data['label1_all']
if input=='fet' or input=='fet_addition':
    slice_all = slice_all[:,0,:,:]
if input == 't1' or input=='t1_addition':
    slice_all = slice_all[:, 1, :, :]
if input == 't2' or input=='t2_addition':
    slice_all = slice_all[:, 2, :, :]
slice_all=torch.unsqueeze(slice_all,dim=1)
y = label1_all
x = slice_all

y = label1_all
x = slice_all
kf=KFold(n_splits=5)

qual_all = []
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/fet_t1t2_v2/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]
    qualified = []
    dataset_test = TensorDataset(x[test_idx, :, :], y[test_idx])
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
    model.cuda()
    model.load_state_dict(torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_t1t2_v2/fet_'+ 'model_cv' + str(cv) + '_'+input+'.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []
    model.eval()
    with torch.no_grad():
        for sub, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.cuda()
            test_y = test_y.long()
            test_y = test_y.cuda()
            test_output = model(test_x)  # model being an instance of torch.nn.Module
            test_y = test_y.long()

            predicted = torch.max(test_output.data, 1)[1]
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
    # test_auc.append(accuracy)
    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
    # sen.append(sens)
    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
    # spe.append(spec)
    auc = metrics.roc_auc_score(test_y_all, predict_p)
    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          )
    qual_all.append([accuracy, sens, spec, auc])
    scio.savemat('./results/T1T2_'+input+'_cv'+str(cv)+'.mat', {'predicted_all': predicted_all, 'predict_p':predict_p, 'test_y_all':test_y_all})

print(qual_all)
print(np.mean(qual_all, axis=0))
print(np.std(qual_all, axis=0))
scio.savemat('./metrics/t1t2_'+input+'_v2.mat', {'qual_all': qual_all})

predicted_all = []
predict_p = []
test_y_all = []

qual_all = []
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/fet_t1t2/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]
    qualified = []
    dataset_test = TensorDataset(x[test_idx, :, :], y[test_idx])
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
    model.cuda()
    model.load_state_dict(torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_t1t2/fet_'+ 'model_cv' + str(cv) + '_fet_addition.pth'))

    model.eval()
    with torch.no_grad():
        for sub, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.cuda()
            test_y = test_y.long()
            test_y = test_y.cuda()
            test_output = model(test_x)  # model being an instance of torch.nn.Module
            test_y = test_y.long()

            predicted = torch.max(test_output.data, 1)[1]
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
# test_auc.append(accuracy)
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
# sen.append(sens)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
# spe.append(spec)
auc = metrics.roc_auc_score(test_y_all, predict_p)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      )

