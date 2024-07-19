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

transform_set = [T.RandomAffine(degrees=(-20,20),translate=(0.1, 0.1)),
                 T.RandomHorizontalFlip(0.2),
                 T.RandomVerticalFlip(0.2),
                 #T.ColorJitter(),
                 #T.RandomRotation(30),
                 #T.GaussianBlur(3,3),
                 ]
transform = T.RandomChoice(transform_set)

data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/met_slice_crop.pth')

slice_all = data['slice_all']
label1_all = data['label1_all']
#label2_all = data['label2_all']
slice_all = torch.unsqueeze(slice_all,1)
petidx1 = torch.zeros(label1_all.shape[0])

data2 = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/fet_slice_crop.pth')

slice_all2 = data2['slice_all']
label1_all2 = data2['label1_all']
#label2_all2 = data2['label2_all']
slice_all2 = torch.unsqueeze(slice_all2,1)
petidx2 = torch.ones(label1_all2.shape[0])

slice_all = torch.cat([slice_all, slice_all2], dim=0)
label1_all = torch.cat([label1_all, label1_all2], dim=0)
petidx_all = torch.cat([petidx1, petidx2], dim=0)


x = slice_all
y = label1_all

kf=KFold(n_splits=5)

EPOCH = 15

qual_all = []

target_pet=1


for cv in [1,2,3,4,5]:
    temp = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/fet_met_sub/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]

    qualified = []
    while not qualified:
        lr = 0.001
        auc_baseline = expect[cv-1]
        dataset_train = TensorDataset(x[train_idx, :, :, :], y[train_idx], petidx_all[train_idx])
        dataset_test = TensorDataset(x[test_idx, :, :, :], y[test_idx], petidx_all[test_idx])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

        y_train = y[train_idx]
        pet_train = petidx_all[train_idx]
        ratio = y[train_idx].sum() / (y[train_idx].shape[0] - y[train_idx].sum())
        if ratio < 1:
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])
        ratio = y_train[pet_train==0].sum() / (y_train[pet_train==0].shape[0] - y_train[pet_train==0].sum())
        if ratio < 1:
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])
        criterion1 = nn.CrossEntropyLoss(weight)  

        y_train = y[train_idx]
        pet_train = petidx_all[train_idx]
        ratio = y_train[pet_train == 1].sum() / (y_train[pet_train == 1].shape[0] - y_train[pet_train == 1].sum())
        if ratio < 1:
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])
        criterion2 = nn.CrossEntropyLoss(weight)  

        ratio = y[train_idx].sum() / (y[train_idx].shape[0] - y[train_idx].sum())
        if ratio < 1:
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])
        criterion_all=nn.CrossEntropyLoss(weight)

        model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
        for epoch in range(EPOCH):
            for sub,(b_x, b_y, b_pet) in enumerate(train_loader):
                model.train()
                b_x = transform(b_x)
                b_x = b_x.cuda()
                b_y = b_y.long()
                b_y = b_y.cuda()
                b_pet = b_pet.long()
                b_pet = b_pet.cuda()

                if epoch <= 8:
                    output = model(b_x)
                    if target_pet == 0:
                       loss = criterion1(output, b_y)
                    else:
                       loss = criterion2(output, b_y)
                    predicted = torch.max(output.data, 1)[1]

                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    predicted = torch.max(output.data, 1)[1]
                    correct = (predicted == b_y).sum()
                    accuracy = float(correct) / float(b_x.shape[0])
                    if sub == 0:
                        print('Epoch:', epoch + 1, 'Batch:', sub + 1, '|train diag loss:', loss.data.item(),
                              '|train accuracy:',
                              accuracy
                              )

                else:
                    if b_y[b_pet == target_pet].shape[0] is not 0:
                        output1 = model(b_x[b_pet == target_pet, :, :, :])
                        if target_pet==0:
                            loss = criterion1(output1, b_y[b_pet == target_pet])
                        else:
                            loss = criterion2(output1, b_y[b_pet == target_pet])
                        predicted1 = torch.max(output1.data, 1)[1]
                        count1 = (predicted1 == b_y[b_pet == target_pet]).sum()

                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if sub == 0:
                        print('Epoch:', epoch + 1, 'Batch:', sub + 1, '|train diag loss:', loss.data.item(),
                              '|train target pet accuracy:', (count1 / float(b_y[b_pet == target_pet].shape[0])).item()
                              )

                #train_loss.append([loss.data.item(),accuracy])
                if epoch>=8:
                    predicted_all = []
                    predict_p = []
                    test_y_all= []

                    model.eval()
                    with torch.no_grad():
                        for sub,(test_x, test_y, test_pet) in enumerate(test_loader):

                            test_x = test_x.cuda()
                            test_y = test_y.long()
                            test_y = test_y.cuda()
                            test_pet = test_pet.long()
                            test_pet = test_pet.cuda()
                            if test_y[test_pet == target_pet].shape[0] is not 0:
                                test_output = model(test_x[test_pet == target_pet, :, :, :])
                                predicted = torch.max(test_output.data, 1)[1]
                                test_y=test_y[test_pet == target_pet]

                                test_y = test_y.cpu()
                                predicted = predicted.cpu()

                                predicted_all = predicted_all + predicted.tolist()
                                test_y_all = test_y_all + test_y.tolist()
                                predict_p = predict_p + test_output[:, 1].tolist()


                    correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                    accuracy = float(correct) / float(len(test_y_all))
                    # test_auc.append(accuracy)
                    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                    # sen.append(sens)
                    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                    # spe.append(spec)
                    auc = metrics.roc_auc_score(test_y_all, predict_p)
                    print('target pet: |test accuracy:', accuracy,
                          '|test sen:', sens,
                          '|test spe:', spec,
                          '|test auc:', auc,
                          )
                    if auc >= auc_baseline and sens > 0.70 and spec > 0.70:
                        auc_baseline = auc
                        if target_pet==0:
                            torch.save(model.state_dict(), '/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_sub/'+ 'model_cv' + str(cv) + '_met_transfer.pth')
                        else:
                            torch.save(model.state_dict(),
                                       '/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_sub/' + 'model_cv' + str(
                                           cv) + '_fet_transfer.pth')
                        print('got one model with |test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', sens,
                              )
                        qualified.append([accuracy, sens, spec, auc])
    qual_all.append(qualified[-1])

print(qual_all)
print(np.mean(qual_all, axis=0))
print(np.std(qual_all, axis=0))

#scio.savemat('train_loss.mat', {'train_loss':train_loss})