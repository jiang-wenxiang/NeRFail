import os
import time
from pathlib import Path

import configargparse
import cv2
import numpy as np
import torch
import tqdm
import wandb
from scipy.spatial import distance
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision.models.inception import Inception3
from torchvision.transforms import Resize

import sys
sys.path.append('/data/data_jiangwenxiang/adversarial_attacks_nerf')

from MyDataset import gauss_dataset, gauss_weight_dataset
from model.GaussNet import gauss_net, create_gauss_w
from model.MyModel import MyCNN


parser = configargparse.ArgumentParser()
parser.add_argument('--label', type=str, default="lego", help='object name')
args = parser.parse_args()

scene_name = args.label

base_mask_image_number = 3

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

index_and_dist_base = "../Create_spatial_point_set/logs/blender_paper_" + scene_name + "/index_and_dist/"

gauss_weight_base = "../Create_spatial_point_set/logs/blender_paper_" + scene_name + "/index_and_weight/"

test_number = 200
val_number = 100
train_number = 100

test_dir = 'test'
val_dir = 'val'
train_dir = 'train'

test_index_and_dist_path = os.path.join(index_and_dist_base, test_dir)
val_index_and_dist_path = os.path.join(index_and_dist_base, val_dir)
train_index_and_dist_path = os.path.join(index_and_dist_base, train_dir)

test_gauss_weight_path = os.path.join(gauss_weight_base, test_dir)
val_gauss_weight_path = os.path.join(gauss_weight_base, val_dir)
train_gauss_weight_path = os.path.join(gauss_weight_base, train_dir)

for dir in [test_gauss_weight_path, val_gauss_weight_path, train_gauss_weight_path]:
    if not os.path.exists(dir):
        os.makedirs(dir)

test_index_and_dist_name_list = [os.path.join(test_index_and_dist_path, str(i) + ".pth") for i in range(test_number)]
val_index_and_dist_name_list = [os.path.join(val_index_and_dist_path, str(i) + ".pth") for i in range(val_number)]
train_index_and_dist_name_list = [os.path.join(train_index_and_dist_path, str(i) + ".pth") for i in range(train_number)]

test_gauss_weight_name_list = [os.path.join(test_gauss_weight_path, str(i) + ".pth") for i in range(test_number)]
val_gauss_weight_name_list = [os.path.join(val_gauss_weight_path, str(i) + ".pth") for i in range(val_number)]
train_gauss_weight_name_list = [os.path.join(train_gauss_weight_path, str(i) + ".pth") for i in range(train_number)]

all_index_and_dist_name_list = test_index_and_dist_name_list + val_index_and_dist_name_list + train_index_and_dist_name_list
all_gauss_weight_name_list = test_gauss_weight_name_list + val_gauss_weight_name_list + train_gauss_weight_name_list

dataset = gauss_weight_dataset(all_index_and_dist_name_list, all_gauss_weight_name_list, device)
# dataset = gauss_weight_dataset(test_index_and_dist_name_list, test_gauss_weight_name_list, device)

dataloader = DataLoader(dataset=dataset, batch_size=1)

since = time.time()

c = 0.02

net = create_gauss_w(device, c)
v_list = []

for img_index_and_dist, gauss_weight_names in tqdm.tqdm(dataloader):
    # print(gauss_weight_names)

    # attack forward
    i_w, dist = net(img_index_and_dist)
    i_ws = i_w.chunk(len(gauss_weight_names), dim=0)

    v = torch.mean(torch.square(dist))
    v_list.append(v.item())

    for i_ws_, gauss_weight_name in zip(i_ws, gauss_weight_names):
        i_ws_ = i_ws_.squeeze(0)
        torch.save(i_ws_, gauss_weight_name)

v = sum(v_list) / len(v_list)
print("v :", v)
