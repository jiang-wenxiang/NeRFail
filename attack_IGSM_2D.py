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
from torchvision.models import vgg16, alexnet, vit_b_16, resnet50

from torchvision.models.inception import Inception3
from torchvision.transforms import Resize

from MyDataset import gauss_dataset, gauss_dataset_rand_select, universal_2D_dataset, universal_2D_dataset_rand_select
from model.GaussNet import gauss_net, universal_2D_net
from model.MyModel import MyCNN
from model_test import test_for_inception

# attack config
parser = configargparse.ArgumentParser()
parser.add_argument('--e', type=int, default=32)
parser.add_argument('--base_mask_image_number', type=int, default=3)
parser.add_argument('--a', type=int, default=2)
parser.add_argument('--targeted_attack', type=bool, default=False)
parser.add_argument('--attack_target_label_int', type=int, default=4)
parser.add_argument('--label', type=str, default='lego')
parser.add_argument('--model_name', type=str, default='inception')

args = parser.parse_args()

targeted_attack = args.targeted_attack
attack_target_label_int = args.attack_target_label_int

a = args.a
epsilon = args.e
attack_epochs = 100

# model_name = "my_model"
model_name = args.model_name
# model_name = "mobilenet"
# model_name = "vit_b_16"

scene_name = args.label

base_mask_image_number = args.base_mask_image_number
simple_rate = 1

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# other config
expname = scene_name + "_attack_IGSM"
basedir = "./output/"+model_name+"/attack"

c = 0.02
# if base_mask_image_number == 3 and scene_name == "lego":
#     c = 0.08638298 # 3 image lego
# elif base_mask_image_number == 4 and scene_name == "lego":
#     c = 0.06204213  # 4 image lego

beta = 0
num_classes = 8
batch_size = 4
dist_step_len = 800
attack_target_label = torch.tensor([attack_target_label_int], device=device)

if not targeted_attack:
    attack_target_label_int = "n"

dataname = scene_name
database = "./data/nerf_synthetic"

rand_init_mask = False
zero_init_mask = True

change_target_base = "./data/nerf_synthetic"

index_and_weight_base = "./Create_spatial_point_set/logs/blender_paper_" + scene_name + "/index_and_weight/"

test_index_and_weight_dir = 'test'
train_index_and_weight_dir = 'train'
val_index_and_weight_dir = 'val'

test_img_from_dir = "test"
train_img_from_dir = "train"
val_img_from_dir = "val"

test_img_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/{}'.format("test")
train_img_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/{}'.format("train")
val_img_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/{}'.format("val")

test_img_mask_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/attack_masks/{}'.format("test")
train_img_mask_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/attack_masks/{}'.format("train")
val_img_mask_save_to_dir = 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/attack_masks/{}'.format("val")

change_target_path = os.path.join(change_target_base, dataname, 'test')
save_target_path = os.path.join(basedir, dataname, 'IGSM_2D_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_a_'+str(a)+'/attack_masks/', 'base_masks')
# wandb_outputs = os.path.join(basedir, dataname, 'wandb')

test_index_and_weight_path = os.path.join(index_and_weight_base, test_index_and_weight_dir)
train_index_and_weight_path = os.path.join(index_and_weight_base, train_index_and_weight_dir)
val_index_and_weight_path = os.path.join(index_and_weight_base, val_index_and_weight_dir)

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

test_index_and_weight_name_list = [os.path.join(test_index_and_weight_path, str(i) + ".pth") for i in range(test_number)]
train_index_and_weight_name_list = [os.path.join(train_index_and_weight_path, str(i) + ".pth") for i in range(train_number)]
val_index_and_weight_name_list = [os.path.join(val_index_and_weight_path, str(i) + ".pth") for i in range(val_number)]

test_img_name_list = [os.path.join(test_img_from_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_name_list = [os.path.join(train_img_from_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_name_list = [os.path.join(val_img_from_path, "r_" + str(i) + ".png") for i in range(val_number)]

test_img_save_to_name_list = [os.path.join(test_img_save_to_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_save_to_name_list = [os.path.join(train_img_save_to_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_save_to_name_list = [os.path.join(val_img_save_to_path, "r_" + str(i) + ".png") for i in range(val_number)]

test_img_mask_save_to_name_list = [os.path.join(test_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(test_number)]
train_img_mask_save_to_name_list = [os.path.join(train_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(train_number)]
val_img_mask_save_to_name_list = [os.path.join(val_img_mask_save_to_path, "r_" + str(i) + ".png") for i in range(val_number)]

all_index_and_weight_name_list = test_index_and_weight_name_list + train_index_and_weight_name_list
all_img_name_list = test_img_name_list + train_img_name_list
all_img_save_to_name_list = test_img_save_to_name_list + train_img_save_to_name_list
all_img_mask_save_to_name_list = test_img_mask_save_to_name_list + train_img_mask_save_to_name_list

if scene_name == "lego" and base_mask_image_number == 2:
    change_target_img_index_list = [75, 125]
elif scene_name == "lego" and base_mask_image_number == 3:
    change_target_img_index_list = [50, 75, 125]
elif scene_name == "lego" and base_mask_image_number == 4:
    change_target_img_index_list = [50, 75, 100, 125]
elif scene_name == "ship" and base_mask_image_number == 2:
    change_target_img_index_list = [50, 100]
elif scene_name == "hotdog" and base_mask_image_number == 4:
    change_target_img_index_list = [50, 75, 100, 125]
elif scene_name == "chair" and base_mask_image_number == 4:
    change_target_img_index_list = [50, 75, 100, 125]
elif base_mask_image_number == 3:
    change_target_img_index_list = [50, 75, 125]
else:
    change_target_img_index_list = [50, 75, 125]

change_target_img_name_list = [os.path.join(change_target_path, "r_" + str(i) + ".png") for i in change_target_img_index_list]
save_target_img_name_list = [os.path.join(save_target_path, str(i) + ".png") for i in change_target_img_index_list]

change_target_img_list = [torch.tensor(cv2.imread(change_target_img_name, cv2.IMREAD_UNCHANGED)).to(device) for change_target_img_name in change_target_img_name_list]
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
    model = resnet50(num_classes=num_classes)
elif model_name == "mobilenet":
    model = models.mobilenet_v2(num_classes=num_classes)
elif model_name == "efficientnet":
    model = models.efficientnet_b0(num_classes=num_classes)

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

dataset = universal_2D_dataset(all_img_name_list + val_img_name_list,
                               all_img_save_to_name_list + val_img_save_to_name_list,
                               all_img_mask_save_to_name_list + val_img_mask_save_to_name_list,
                               device)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

dataset_rand_select = universal_2D_dataset(all_img_name_list + val_img_name_list,
                                           all_img_save_to_name_list + val_img_save_to_name_list,
                                           all_img_mask_save_to_name_list + val_img_mask_save_to_name_list,
                                           device)

dataloader_rand_select = DataLoader(dataset=dataset_rand_select, batch_size=batch_size)

# train
since = time.time()
val_acc_history = []

print("Start attack "+model_name+"!")

criterion = nn.CrossEntropyLoss()
img_rgba_loss = torch.nn.MSELoss()

# best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# sample execution (requires torchvision)

dataset_type = 'test'

# use wandb to log
# wandb.init(project="images-classification", name=expname, dir=wandb_outputs)

change_target_tensor_all = torch.zeros((len(dataloader.dataset), 800, 800, 3)).to(device)

change_target_tensor_init_all = change_target_tensor_all.clone().detach()
change_target_tensor_best_all = change_target_tensor_all.clone().detach()

attack_epoch_acc_best = 0
attack_epoch_loss_best = 0

if targeted_attack:
    attack_epoch_acc_best = 0
else:
    attack_epoch_acc_best = 10000

for epoch in range(attack_epochs):
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
    beta_attack_loss = 0.0
    attack_img_loss = 0.0
    beta_attack_img_loss = 0.0
    attack_total_loss = 0.0
    attack_corrects = 0

    now_dataloader = dataloader_rand_select
    # save the images at the last epoch
    if epoch == (attack_epochs - 1):
        now_dataloader = dataloader

    for index, ori_img, img_save_to_name, img_mask_save_to_name in tqdm.tqdm(dataloader):
        change_target_tensor = change_target_tensor_all[index, :, :, :]
        change_target_tensor_init = change_target_tensor_init_all[index, :, :, :]
        change_target_tensor_best = change_target_tensor_best_all[index, :, :, :]

        if epoch < (attack_epochs - 1):
            change_target_tensor = Variable(change_target_tensor, requires_grad=True)
            change_target_tensor.requires_grad = True

        # save the images at the last epoch
        elif epoch == (attack_epochs - 1):
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
        img_pixel_loss = img_rgba_loss(r, ori_img)
        _, ae_preds = torch.max(cla, 1)

        beta_1 = 1 - beta
        if beta < 0:
            beta_1 = 1 + beta

        total_loss = (beta_1 * ae_loss) + (beta * img_pixel_loss)

        # attack statistics
        attack_loss += ae_loss.item() * len(ori_img)
        beta_attack_loss += ((1-beta) * ae_loss.item()) * len(ori_img)
        attack_img_loss += img_pixel_loss.item() * len(ori_img)
        beta_attack_img_loss += (beta * img_pixel_loss.item()) * len(ori_img)
        attack_total_loss += total_loss.item() * len(ori_img)
        attack_corrects += torch.sum(ae_preds == attack_target_label_r)

        if epoch < (attack_epochs - 1):
            # loss backward
            total_loss.backward(retain_graph=True)

            # attack and limit dont change alpha out of mask
            # change_target_tensor_size = change_target_tensor[:, :, :, :3].size()
            # change_target_tensor_alpha = change_target_tensor[:, :, :, 3].unsqueeze(-1)\
            #     .broadcast_to(change_target_tensor_size)

            # attack
            if targeted_attack:
                change_target_tensor = change_target_tensor - a * torch.sign(change_target_tensor.grad.data).to(device)
                # if zero_init_mask:
                #     tensor_0 = torch.ones_like(change_target_tensor_rgba[:, :, :, :3]) * 100
                # else:
                # tensor_0 = torch.zeros_like(change_target_tensor_rgba[:, :, :, :3])
                # change_target_tensor_rgb = torch.where(change_target_tensor_alpha > 0, change_target_tensor_rgba[:, :, :, :3], tensor_0)
                # change_target_tensor = torch.cat([change_target_tensor_rgb, change_target_tensor_alpha[:, :, :, 0].unsqueeze(-1)], dim=-1)
            else:
                change_target_tensor = change_target_tensor + a * torch.sign(change_target_tensor.grad.data).to(device)
                # if zero_init_mask:
                #     tensor_0 = torch.ones_like(change_target_tensor_rgba[:, :, :, :3]) * 100
                # else:
                # tensor_0 = torch.zeros_like(change_target_tensor_rgba[:, :, :, :3])
                # change_target_tensor_rgb = torch.where(change_target_tensor_alpha > 0, change_target_tensor_rgba[:, :, :, :3], tensor_0)
                # change_target_tensor = torch.cat([change_target_tensor_rgb, change_target_tensor_alpha[:, :, :, 0].unsqueeze(-1)], dim=-1)

            # attack 0-255 limit
            # temp = torch.cat([change_target_tensor.unsqueeze(0),
            #                   torch.ones(change_target_tensor.size()).unsqueeze(0).to(device)], 0)
            # change_target_tensor = torch.max(temp, dim=0)[0]
            #
            # temp = torch.cat([change_target_tensor.unsqueeze(0),
            #                   torch.ones(change_target_tensor.size()).unsqueeze(0).to(device) * 255], 0)
            # change_target_tensor = torch.min(temp, dim=0)[0]

            # attack +- epsilon limit
            change_target_tensor_max = change_target_tensor_init + epsilon
            change_target_tensor_min = change_target_tensor_init - epsilon

            temp = torch.cat([change_target_tensor.unsqueeze(0), change_target_tensor_min.unsqueeze(0)], 0)
            change_target_tensor = torch.max(temp, dim=0)[0]

            temp = torch.cat([change_target_tensor.unsqueeze(0), change_target_tensor_max.unsqueeze(0)], 0)
            change_target_tensor = torch.min(temp, dim=0)[0]

            change_target_tensor_all[index, :, :, :] = change_target_tensor.clone().detach()

        elif epoch == (attack_epochs - 1):
            gauss_mask_img_tensor = x.chunk(len(ori_img), dim=0)
            gauss_img_tensor = r.chunk(len(ori_img), dim=0)
            ori_img_tensor = ori_img.chunk(len(ori_img), dim=0)

            for gauss_img, gauss_mask_img, img_save_file_name, img_mask_save_file_name, chunk_ori_img in \
                    zip(gauss_img_tensor, gauss_mask_img_tensor, img_save_to_name, img_mask_save_to_name, ori_img_tensor):
                cv2.imwrite(img_save_file_name, gauss_img.squeeze(0).cpu().detach().numpy())
                cv2.imwrite(img_mask_save_file_name, gauss_mask_img.squeeze(0).cpu().detach().numpy())
                cv2.imwrite(img_save_file_name.replace(".png", "_ori.png"), chunk_ori_img.squeeze(0).cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    attack_epoch_loss = attack_loss / len(dataloader.dataset)
    beta_attack_epoch_loss = beta_attack_loss / len(dataloader.dataset)
    attack_epoch_acc = attack_corrects.double() / len(dataloader.dataset)
    attack_epoch_img_loss = attack_img_loss / len(dataloader.dataset)
    beta_attack_epoch_img_loss = beta_attack_img_loss / len(dataloader.dataset)
    attack_epoch_total_loss = attack_total_loss / len(dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('attack', attack_epoch_loss, attack_epoch_acc))
    print('{} Beta Loss: {:.4f} Beta Img Loss: {:.4f}'.format('attack', beta_attack_epoch_loss, beta_attack_epoch_img_loss))
    print('{} Img Loss: {:.4f} Total Loss: {:.4f}'.format('attack', attack_epoch_img_loss, attack_epoch_total_loss))

    if targeted_attack:
        if attack_epoch_acc > attack_epoch_acc_best:
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            change_target_tensor_best_all = change_target_tensor_all.clone().detach()
    else:
        if attack_epoch_acc < attack_epoch_acc_best:
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            change_target_tensor_best_all = change_target_tensor_all.clone().detach()

    # wandb.log({'test Loss': epoch_loss, 'test Acc': epoch_acc}, step=epoch)
    # wandb.log({'attack Loss': attack_epoch_loss, 'attack Acc': attack_epoch_acc,
    #            '(1-beta) x attack-Loss': beta_attack_epoch_loss, 'beta x attack-img-Loss': beta_attack_epoch_img_loss,
    #            'attack img Loss': attack_epoch_img_loss, 'attack total Loss': attack_epoch_total_loss}, step=epoch)

time_elapsed = time.time() - since
print('Attack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

test_for_inception(print_e=epsilon, model_name=model_name, base_mask_image_number=base_mask_image_number,
                   scene_name=scene_name, target_class_idx=attack_target_label_int,
                   method_name='IGSM_2D', setname="test", step=0)

test_for_inception(print_e=epsilon, model_name=model_name, base_mask_image_number=base_mask_image_number,
                   scene_name=scene_name, target_class_idx=attack_target_label_int,
                   method_name='IGSM_2D', setname="val", step=0)
