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
from torchvision import models
from torchvision.models import vgg16, alexnet, vit_b_16

from torchvision.models.inception import Inception3
from torchvision.transforms import Resize

import deepfool
from MyDataset import gauss_dataset, gauss_dataset_rand_select, universal_2D_dataset, universal_2D_dataset_rand_select
from model.GaussNet import gauss_net, gauss_get_r, gauss_get_img, universal_2D_net
from model.MyModel import MyCNN
from model_test import test_for_inception

# attack config
parser = configargparse.ArgumentParser()
parser.add_argument('--e', type=int, default=32)
parser.add_argument('--m1', type=int, default=8)
parser.add_argument('--m2', type=int, default=100)
parser.add_argument('--targeted_attack', type=bool, default=False)
parser.add_argument('--attack_target_label_int', type=int, default=4)
parser.add_argument('--label', type=str, default='lego')
parser.add_argument('--model_name', type=str, default='inception')

args = parser.parse_args()

# attack config
targeted_attack = args.targeted_attack
attack_target_label_int = args.attack_target_label_int

epsilon = args.e
m1 = args.m1
m2 = args.m2
attack_epochs = 100

# model_name = "my_model"
model_name = args.model_name
# model_name = "vgg16"
# model_name = "resnet50"
# model_name = "efficientnet"
# model_name = "mobilenet"
# model_name = "vit_b_16"
scene_name = args.label
# scene_name = "ship"

simple_rate = 1

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# other config
expname = scene_name + "_attack_UAP"
basedir = "./output/"+model_name+"/attack"

c = 0.02
num_classes = 8
batch_size = 1
dist_step_len = 800

df_max_iter = 1000

attack_target_label = torch.tensor([attack_target_label_int], device=device)

if not targeted_attack:
    attack_target_label_int = "n"

dataname = scene_name
database = "./data/nerf_synthetic"

rand_init_mask = False
zero_init_mask = True

change_target_base = "./data/nerf_synthetic"

test_index_and_weight_dir = 'test'
train_index_and_weight_dir = 'train'
val_index_and_weight_dir = 'val'

test_img_from_dir = "test"
train_img_from_dir = "train"
val_img_from_dir = "val"

test_img_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("test")
train_img_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("train")
val_img_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("val")

test_img_mask_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("test")
train_img_mask_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("train")
val_img_mask_save_to_dir = 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("val")

change_target_path = os.path.join(change_target_base, dataname, 'test')
save_target_path = os.path.join(basedir, dataname, 'Universal_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/', 'base_masks')
# wandb_outputs = os.path.join(basedir, dataname, 'wandb')

change_target_img_path = os.path.join(save_target_path, 'universal.pth')

test_img_from_path = os.path.join(database, dataname, test_img_from_dir)
train_img_from_path = os.path.join(database, dataname, train_img_from_dir)
val_img_from_path = os.path.join(database, dataname, val_img_from_dir)

test_img_save_to_path = os.path.join(basedir, dataname, test_img_save_to_dir)
train_img_save_to_path = os.path.join(basedir, dataname, train_img_save_to_dir)
val_img_save_to_path = os.path.join(basedir, dataname, val_img_save_to_dir)

test_img_mask_save_to_path = os.path.join(basedir, dataname, test_img_mask_save_to_dir)
train_img_mask_save_to_path = os.path.join(basedir, dataname, train_img_mask_save_to_dir)
val_img_mask_save_to_path = os.path.join(basedir, dataname, val_img_mask_save_to_dir)

zero_s = ["000", "00", "0", ""]

test_number = 200
train_number = 100
val_number = 100

for dir in [test_img_save_to_path, train_img_save_to_path, val_img_save_to_path,
            test_img_mask_save_to_path, train_img_mask_save_to_path, val_img_mask_save_to_path,
            save_target_path]:
    if not os.path.exists(dir):
        os.makedirs(dir)

test_img_name_list = [os.path.join(test_img_from_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_name_list = [os.path.join(train_img_from_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_name_list = [os.path.join(val_img_from_path, "r_" + str(i) + ".png") for i in range(val_number)]

test_img_save_to_name_list = [os.path.join(test_img_save_to_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_save_to_name_list = [os.path.join(train_img_save_to_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_save_to_name_list = [os.path.join(val_img_save_to_path, "r_" + str(i) + ".png") for i in range(val_number)]

test_img_mask_save_to_name_list = [os.path.join(test_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_mask_save_to_name_list = [os.path.join(train_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_mask_save_to_name_list = [os.path.join(val_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(val_number)]

all_img_name_list = test_img_name_list + train_img_name_list
all_img_save_to_name_list = test_img_save_to_name_list + train_img_save_to_name_list
all_img_mask_save_to_name_list = test_img_mask_save_to_name_list + train_img_mask_save_to_name_list

# part 1 end

if model_name == "my_model":
    # init my model CNN
    model = MyCNN(num_classes=num_classes)
elif model_name == "inception":
    model = Inception3(num_classes=num_classes, init_weights=False)
elif model_name == "vgg16":
    model = vgg16(num_classes=num_classes, init_weights=False)
elif model_name == "alexnet":
    model = alexnet(num_classes=num_classes)
elif model_name == "vit_b_16":
    model = vit_b_16(num_classes=num_classes)
elif model_name == "resnet50":
    model = models.resnet50(num_classes=num_classes)
elif model_name == "densenet121":
    model = models.densenet121(num_classes=num_classes)
elif model_name == "mobilenet":
    model = models.mobilenet_v2(num_classes=num_classes)
elif model_name == "efficientnet":
    model = models.efficientnet_b0(num_classes=num_classes)
elif model_name == "swin_b":
    model = models.swin_b(num_classes=num_classes)

# change device
model.to(device)

# load pretrain weight
if model_name == "my_model":
    weights_path = Path("./model/weights/my_model_"+str(num_classes)+"_best.pth")
elif model_name == "inception":
    weights_path = Path("./model/weights/inception_"+str(num_classes)+"_best.pth")
else:
    weights_path = Path("./model/weights/"+model_name+"_" + str(num_classes) + "_best.pth")


pretrain_model = torch.load(weights_path, map_location=device)
model.load_state_dict(pretrain_model)

# init attack net
net = universal_2D_net(device, c, model, model_name)

net.to(device)

dataset = universal_2D_dataset(all_img_name_list+val_img_name_list,
                               all_img_save_to_name_list+val_img_save_to_name_list,
                               all_img_mask_save_to_name_list+val_img_mask_save_to_name_list, device)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

dataset_rand_select = universal_2D_dataset_rand_select(all_img_name_list, all_img_save_to_name_list, all_img_mask_save_to_name_list, device, select_rate=simple_rate)
dataloader_rand_select = DataLoader(dataset=dataset_rand_select, batch_size=batch_size)

# train
since = time.time()
val_acc_history = []

print("Start attack "+model_name+"!")

criterion = nn.CrossEntropyLoss()

# best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# sample execution (requires torchvision)

dataset_type = 'test'

# use wandb to log
# wandb.init(project="images-classification", name=expname, dir=wandb_outputs)

change_target_tensor = torch.zeros((800, 800, 3)).to(device)

change_target_tensor_init = change_target_tensor.detach().clone()
change_target_tensor_best = change_target_tensor.detach().clone()

def project_perturbation(data_point, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(1, data_point / torch.norm(perturbation.flatten(1)))
    elif p == torch.inf:
        perturbation = torch.clamp(perturbation, -data_point, data_point)
    return perturbation

attack_epoch_acc_best = 0
attack_epoch_loss_best = 0

if targeted_attack:
    attack_epoch_acc_best = 0
else:
    attack_epoch_acc_best = 10000

epoch=0

while epoch < attack_epochs:
    print("Attack ...... ["+str(epoch)+"/"+str(attack_epochs)+"]")

    model.train(False)
    net.train(False)

    for param in model.parameters():
        param.requires_grad = False
    for param in net.parameters():
        param.requires_grad = False

    running_loss = 0.0
    running_corrects = 0

    attack_loss = 0.0
    attack_corrects = 0

    tensor_not_changed = True

    now_dataloader = dataloader_rand_select
    # save the images at the last epoch
    if epoch == (attack_epochs - 1):
        now_dataloader = dataloader

    for index, ori_img, img_save_to_name, img_mask_save_to_name in tqdm.tqdm(now_dataloader):

        # save the images at the last epoch
        if epoch == (attack_epochs - 1):
            change_target_tensor = change_target_tensor_best
            change_target_tensor.requires_grad = False

        # attack forward
        x, r, cla, ori_img, ori_cla = net(change_target_tensor, ori_img)

        attack_target_label_r = attack_target_label.broadcast_to([ori_cla.size()[0], ])
        # normal testing
        loss = criterion(ori_cla, attack_target_label_r)
        _, preds = torch.max(ori_cla, 1)

        # statistics
        running_loss += loss.item() * len(ori_img)
        running_corrects += torch.sum(preds == attack_target_label_r)

        ae_loss = criterion(cla, attack_target_label_r)
        _, ae_preds = torch.max(cla, 1)

        # attack statistics
        attack_loss += ae_loss.item() * len(ori_img)
        attack_corrects += torch.sum(ae_preds == attack_target_label_r)

        if epoch < (attack_epochs - 1):
            ori_img_list = ori_img.chunk(len(ori_img), dim=0)

            for i in range(len(ori_img)):
                cla_same = False

                for i in range(len(ori_img)):
                    pre_cla = torch.argmax(cla[i])
                    pre_ori_cla = torch.argmax(ori_cla[i])
                    if pre_cla == pre_ori_cla:
                        cla_same = True
                        break

                if cla_same:

                    net_input = (change_target_tensor, ori_img)

                    # Finding a new minimal perturbation with deepfool to fool the network on this image
                    if targeted_attack:
                        dr, iter_k, label, k_i, pert_image = deepfool.deepfool(net_input, epsilon, net, max_iter=df_max_iter, target_label = attack_target_label_int, m1=m1, m2=m2, universal_2d=True)
                    else:
                        dr, iter_k, label, k_i, pert_image = deepfool.deepfool(net_input, epsilon, net, max_iter=df_max_iter, m1=m1, m2=m2, universal_2d=True)

                    # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                    if iter_k < df_max_iter:
                        print("iter_k < df_max_iter")
                        change_target_tensor += dr
                        tensor_not_changed = False
                        change_target_tensor = project_perturbation(epsilon, torch.inf, change_target_tensor)


        elif epoch == (attack_epochs - 1):
            gauss_mask_img_tensor = x.chunk(len(ori_img), dim=0)
            gauss_img_tensor = r.chunk(len(ori_img), dim=0)
            ori_img_tensor = ori_img.chunk(len(ori_img), dim=0)

            for gauss_img, gauss_mask_img, img_save_file_name, img_mask_save_file_name, chunk_ori_img in \
                    zip(gauss_img_tensor, gauss_mask_img_tensor, img_save_to_name, img_mask_save_to_name, ori_img_tensor):
                cv2.imwrite(img_save_file_name, gauss_img.squeeze(0).cpu().detach().numpy())
                cv2.imwrite(img_mask_save_file_name, gauss_mask_img.squeeze(0).cpu().detach().numpy())
                cv2.imwrite(img_save_file_name.replace(".png", "_ori.png"), chunk_ori_img.squeeze(0).cpu().detach().numpy())

            tensor_not_changed = False

    if tensor_not_changed:
        epoch = attack_epochs - 1
    else:
        epoch += 1

    epoch_loss = running_loss / len(now_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(now_dataloader.dataset)

    attack_epoch_loss = attack_loss / len(now_dataloader.dataset)
    attack_epoch_acc = attack_corrects.double() / len(now_dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('attack', attack_epoch_loss, attack_epoch_acc))

    if targeted_attack:
        if attack_epoch_acc > attack_epoch_acc_best:
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            change_target_tensor_best = change_target_tensor.clone().detach()
    else:
        if attack_epoch_acc < attack_epoch_acc_best:
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            change_target_tensor_best = change_target_tensor.clone().detach()

    # wandb.log({'test Loss': epoch_loss, 'test Acc': epoch_acc}, step=epoch)
    # wandb.log({'attack Loss': attack_epoch_loss, 'attack Acc': attack_epoch_acc}, step=epoch)

torch.save(change_target_tensor, change_target_img_path)

time_elapsed = time.time() - since
print('Attack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

test_for_inception(print_e=epsilon, model_name=model_name, method_name="Universal_2D",
                   target_class_idx=attack_target_label_int,
                   m1=m1, m2=m2, setname="test", step=0, scene_name=scene_name)

test_for_inception(print_e=epsilon, model_name=model_name, method_name="Universal_2D",
                   target_class_idx=attack_target_label_int,
                   m1=m1, m2=m2, setname="val", step=0, scene_name=scene_name)
