from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torch.utils.data import Dataset, TensorDataset
import torch
import models
from sklearn.model_selection import KFold
import scipy.io as scio
import numpy as np

pet_type = 'met'
posfix = 'transfer'
data = torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/datasets/'+pet_type+'_slice_crop.pth')

slice_all = data['slice_all']
label1_all = data['label1_all']
slice_all = torch.unsqueeze(slice_all,1)

y = label1_all
x = slice_all

qual_all = []
cv = 2
temp = scio.loadmat('./kfold/'+pet_type+'_sub/shuffled_index_cv' + str(cv))
train_idx = temp['train_idx'][0]
test_idx = temp['test_idx'][0]
qualified = []
# dataset_test = TensorDataset(x[test_idx, :, :], y[test_idx])
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
xx = x[test_idx, :, :,:]
yy = y[test_idx]
model = models.Conv2D_resnet(num_classes=2, in_ch=slice_all.shape[1])
model.cuda()
model.load_state_dict(torch.load('/home/PJLAB/liumianxin/Desktop/VIEW_codes/models_sub/'+ 'model_cv' + str(cv) + '_'+pet_type+'_'+posfix+'.pth'))
model.eval()
dataset_test = TensorDataset(x[test_idx, :, :], y[test_idx])
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
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
TN = np.where((np.array(predicted_all) == np.array(test_y_all)) & (np.array(test_y_all)== 0))
TP = np.where((np.array(predicted_all) == np.array(test_y_all)) & (np.array(test_y_all)== 1))


target_layers = [model.conv_layer3.conv1]
input_tensor = xx[TP[0][30:40],:,:,:] # Create an input tensor image for your model..

# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cams = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)

# In this example grayscale_cam has only one image in the batch:
from PIL import Image
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2

transform = T.ToPILImage()
index = 6
grayscale_cam = grayscale_cams[index]
print(np.mean(grayscale_cam))
# gray_img = np.array(transform(xx[1,:,:]))
gray_img = np.array(torch.squeeze(input_tensor[index,:,:,:])/torch.max(torch.squeeze(input_tensor[index,:,:,:]))*255)
gray_img = gray_img/255
print(np.mean(gray_img))
rgb_img = cv2.merge([gray_img, gray_img, gray_img])

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
camimg = np.uint8(255*grayscale_cam)
camimg = cv2.merge([camimg, camimg, camimg])
images = np.hstack((np.uint8(255*rgb_img), visualization))
plot = Image.fromarray(images)
plot.show()
plot.save('./figures/MET_3.png')
# You can also get the model outputs without having to re-inference
model_outputs = cam.outputs