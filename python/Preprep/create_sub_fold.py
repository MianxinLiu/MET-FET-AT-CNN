import os
import numpy as np
import scipy.io as scio

savepath = '/home/PJLAB/liumianxin/Desktop/VIEW_codes/kfold/fet_sub/'
pet_data = scio.loadmat('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/fet_count_testset3.mat')

slicenum = pet_data['slicenum'][0].tolist()
tmp = np.cumsum(slicenum)
start_point = [0] + tmp[:-1].tolist()

index_idx = [i for i in range(1, len(slicenum) + 1) for _ in range(slicenum[i - 1])]
scio.savemat(os.path.join(savepath, f'slice2sub_testset3.mat'), {'index_idx':index_idx})

step = int(len(slicenum) * 0.2)+1
index = np.random.permutation(len(slicenum))

cut = list(range(0, len(slicenum), step))

for k in range(1, 6):
    if k != 5:
        pos = slice(cut[k - 1], cut[k])
    else:
        pos = slice(cut[k - 1], len(index))

    tmp2 = np.zeros(len(slicenum))
    tmp2[index[pos]] = 1
    trainsub = np.where(tmp2 == 0)[0]
    testsub = np.where(tmp2 == 1)[0]
    print(trainsub)
    print(testsub)

    train_idx = []
    test_idx = []

    for j in trainsub:
        train_idx.extend(range(start_point[j], start_point[j] + slicenum[j]))
    for j in testsub:
        test_idx.extend(range(start_point[j], start_point[j] + slicenum[j]))

    scio.savemat(os.path.join(savepath, f'shuffled_index_cv{k}'), {'train_idx':train_idx, 'test_idx':test_idx})
