from sklearn.metrics import accuracy_score
from data_loader import get_loader
from config import get_config
from solver import Solver

import torch
import models

import mne

train_config = get_config(mode='train')
test_config = get_config(mode='test')

device= "cpu"



"""
Class activation topography (CAT) for EEG model visualization, combining class activity map and topography
Code: Class activation map (CAM) and then CAT

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam

Salute every open-source researcher and developer!
"""


import argparse
import os
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
from torch.backends import cudnn
# from tSNE import plt_tsne
# from grad_cam.utils import GradCAM, show_cam_on_image
from utils import GradCAM, show_cam_on_image

cudnn.benchmark = False
cudnn.deterministic = True


def get_mean_grayscale_cam(data):
    test = torch.as_tensor(data, dtype=torch.float32)
    test = torch.autograd.Variable(test, requires_grad=True)

    grayscale_cam = cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]

    mean_all_cam = np.mean(grayscale_cam, axis=1)


    return grayscale_cam, mean_all_cam
# keep the overall model class, omitted here
class ViT(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            # ... the model
        )



nSub = 1
target_category = 2  # set the class (class activation mapping)

# ! A crucial step for adaptation on Transformer
# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    # result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return tensor#result




model = getattr(models, train_config.model_name)(train_config)

from torchsummary import summary

# Suppose your EEGNet input is (1, 1, 64, 128)
# s=summary(model, input_size=(1, 64, 159))
# print('s:', s)
# exit()


# # used for cnn model without transformer
# model.load_state_dict(torch.load('./model/model_cnn.pth', map_location=device))
# target_layers = [model[0].projection]  # set the layer you want to visualize, you can use torchsummary here to find the layer index
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

CAND_CHANNELS=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 
'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz','O2','PO10','AF7','AF3',
'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5',
'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']

inter1020_montage = mne.channels.make_standard_montage('standard_1020')

cand_channels = [ch for ch in inter1020_montage.ch_names if ch in CAND_CHANNELS ]
info = mne.create_info(ch_names = cand_channels, sfreq=256., ch_types='eeg')


fig, axes = plt.subplots(5,8, figsize=(12,7))

col_pos=0
for i in range(1,16):
    if i not in [1,2,3,5,7,10,12,15]:
        continue
    train_config.subject=test_config.subject=i
    load_path_root='/home/Hyunwook/codes/BrainCon-revision/checkpoints'
    load_file_name='model_'+str(train_config.subject)+'.std'

    load_path = os.path.join(load_path_root, load_file_name)

    model.load_state_dict(torch.load(load_path, map_location=device))
    target_layers = [model.conv3]  # set the target layer 
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)


    all_cam = []


    test_data_loader = get_loader(test_config, shuffle=False)

    label_featureList_map={}


    for batch in test_data_loader:
        features, labels = batch
        # print(features.shape, labels.shape)
        for j in range(labels.shape[0]):
            label = int(labels[j].detach().cpu().numpy())

            # print(features[j].shape, label)
            if label not in label_featureList_map:
                label_featureList_map[label]=[]

            label_featureList_map[label].append(features[j])


    for j in range(5):
        label_featureList_map[j]=np.asarray(label_featureList_map[j])

    # for j in range(5):
    #     print(label_featureList_map[j].shape)


    grayscale_cam, hello_mean_grayscale_cam = get_mean_grayscale_cam(label_featureList_map[0])
    _, helpme_mean_grayscale_cam = get_mean_grayscale_cam(label_featureList_map[1])
    _, stop_mean_grayscale_cam = get_mean_grayscale_cam(label_featureList_map[2])
    _, thankyou_mean_grayscale_cam = get_mean_grayscale_cam(label_featureList_map[3])
    _, yes_mean_grayscale_cam = get_mean_grayscale_cam(label_featureList_map[4])

    evoked = mne.EvokedArray(grayscale_cam, info)
    evoked.set_montage(inter1020_montage)


    im, cn = mne.viz.plot_topomap(hello_mean_grayscale_cam, evoked.info, show=False, axes=axes[0,col_pos], res=1200)
    im, cn = mne.viz.plot_topomap(helpme_mean_grayscale_cam, evoked.info, show=False, axes=axes[1,col_pos], res=1200)
    im, cn = mne.viz.plot_topomap(stop_mean_grayscale_cam, evoked.info, show=False, axes=axes[2,col_pos], res=1200)
    im, cn = mne.viz.plot_topomap(thankyou_mean_grayscale_cam, evoked.info, show=False, axes=axes[3,col_pos], res=1200)
    im, cn = mne.viz.plot_topomap(yes_mean_grayscale_cam, evoked.info, show=False, axes=axes[4,col_pos], res=1200)
    col_pos+=1

# plt.subplot(212)
# im2, cn2 = mne.viz.plot_topomap(mean_all_cam, evoked.info, show=False, axes=axes[1,0], res=1200)
axes[0,0].set_title('Sub01')
axes[0,1].set_title('Sub02')
axes[0,2].set_title('Sub03')
axes[0,3].set_title('Sub05')
axes[0,4].set_title('Sub07')
axes[0,5].set_title('Sub10')
axes[0,6].set_title('Sub12')
axes[0,7].set_title('Sub15')


axes[0,0].set_ylabel('Hello')
axes[1,0].set_ylabel('Help me')
axes[2,0].set_ylabel('Stop')
axes[3,0].set_ylabel('Thank you')
axes[4,0].set_ylabel('Yes')


cbar_ax = fig.add_axes([0.39, -0.05, 0.6, 0.03])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')            
# # Adjust subplot layout manually

fig.subplots_adjust(
    # top=1.0,  # Space at the top
    bottom=0.2,  # Space at the bottom (enough for the color bar)
    # left=0.1,  # Space on the left
    # right=0.9,  # Space on the right
    hspace=0.1,  # Space between rows
    wspace=1e-2   # Space between columns
)
plt.tight_layout()

plt.savefig('topo.png', bbox_inches='tight' )


