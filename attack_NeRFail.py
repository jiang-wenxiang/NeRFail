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
from MyDataset import gauss_dataset, gauss_dataset_rand_select
from model.GaussNet import gauss_net, gauss_get_r, gauss_get_img
from model.MyModel import MyCNN
from model_test import test_for_inception

# attack config
parser = configargparse.ArgumentParser()
parser.add_argument('--e', type=int, default=32)
parser.add_argument('--base_mask_image_number', type=int, default=3)
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
m1_list = [0, args.m1]
m1 = args.m1
m2 = args.m2
attack_epochs = 100

m2_max_limit = 1000000

# model_name = "my_model"
model_name = args.model_name
# model_name = "vit_b_16"

# model_name = "vgg16"
# model_name = "alexnet"

# model_name = "resnet50"
# model_name = "densenet121"
# model_name = "mobilenet"
# model_name = "efficientnet"
# model_name = "swin_b"

scene_name = args.label

base_mask_image_number = args.base_mask_image_number
simple_rate = 1

accumulated_incomplete_attack = False

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# other config
expname = scene_name + "_attack_NeRFail"
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

index_and_weight_base = "./Create_spatial_point_set/logs/blender_paper_" + scene_name + "/index_and_weight/"

test_index_and_weight_dir = 'test'
train_index_and_weight_dir = 'train'
val_index_and_weight_dir = 'val'

test_img_from_dir = "test"
train_img_from_dir = "train"
val_img_from_dir = "val"

test_img_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("test")
train_img_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("train")
val_img_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format("val")

test_img_mask_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("test")
train_img_mask_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("train")
val_img_mask_save_to_dir = 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/{}'.format("val")

change_target_path = os.path.join(change_target_base, dataname, 'test')
save_target_path = os.path.join(basedir, dataname, 'NeRFail_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(attack_target_label_int)+'_e_'+str(epsilon)+'_m_'+str(m1)+'_'+str(m2)+'/attack_masks/', 'base_masks')
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
change_target_tensor = torch.stack(change_target_img_list)
net = gauss_net(device, c, model, model_name, epsilon)

net.to(device)

dataset = gauss_dataset(all_index_and_weight_name_list + val_index_and_weight_name_list,
                        all_img_name_list + val_img_name_list,
                        all_img_save_to_name_list + val_img_save_to_name_list,
                        all_img_mask_save_to_name_list + val_img_mask_save_to_name_list,
                        device)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

dataset_rand_select = gauss_dataset_rand_select(all_index_and_weight_name_list, all_img_name_list,
                                                all_img_save_to_name_list, all_img_mask_save_to_name_list,
                                                device, select_rate=simple_rate)
dataloader_rand_select = DataLoader(dataset=dataset_rand_select, batch_size=batch_size, shuffle=True)

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

change_target_tensor = torch.tensor(change_target_tensor, dtype=torch.float)

if rand_init_mask:
    change_target_tensor_rand = torch.rand_like(change_target_tensor) * epsilon
    target_alpha = change_target_tensor[:, :, :, 3].unsqueeze(-1)
    change_target_tensor = torch.where(target_alpha > 0, change_target_tensor_rand, change_target_tensor)

if zero_init_mask:
    change_target_tensor_zero = torch.zeros_like(change_target_tensor)
    target_alpha = change_target_tensor[:, :, :, 3]
    change_target_tensor_zero[:, :, :, 3] = target_alpha
    change_target_tensor = change_target_tensor_zero

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

m1_best = 0
m2_best = 0

if targeted_attack:
    attack_epoch_acc_best = 0
else:
    attack_epoch_acc_best = 10000

epoch = 0
need_to_log_dict = {}

# m1_index = 1
# m1 = m1_list[m1_index]

while epoch < attack_epochs:
    print("Attack ...... ["+str(epoch)+"/"+str(attack_epochs)+"]")
    since_epoch = time.time()

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

    no_attack_number_after_last_m2_change = 0
    attack_number_all_after_last_m2_change = 0
    # reset m2
    m2 = args.m2

    tensor_not_changed = True

    now_dataloader = dataloader_rand_select
    # save the images at the last epoch
    if epoch == (attack_epochs - 1):
        now_dataloader = dataloader

    for index, ori_img, img_index_and_weight, img_save_to_name, img_mask_save_to_name in tqdm.tqdm(now_dataloader):

        # save the images at the last epoch
        if epoch == (attack_epochs - 1):
            change_target_tensor = change_target_tensor_best
            change_target_tensor.requires_grad = False
            print("best m1="+str(m1_best))
            print("best m2=" + str(m2_best))
            print("best acc=" + str(attack_epoch_acc_best))

        net.open_update_epsilon_3d()

        # attack forward
        x, r, cla, ori_img, ori_cla = net(change_target_tensor, img_index_and_weight, ori_img)

        attack_target_label_r = attack_target_label.broadcast_to([ori_cla.size()[0], ])
        # normal testing
        loss = criterion(ori_cla, attack_target_label_r)
        _, preds = torch.max(ori_cla, 1)

        # statistics
        running_loss += loss.item() * len(ori_img)
        running_corrects += torch.sum(preds == attack_target_label_r)

        # if targeted_attack:
        #     cla[:, :int(attack_target_label)] = cla[:, :int(attack_target_label)] + m1  # add m1
        #     cla[:, int(attack_target_label) + 1:] = cla[:, int(attack_target_label) + 1:] + m1  # add m1
        # else:
        #     cla[:, int(attack_target_label)] = cla[:, int(attack_target_label)] + m1

        ae_loss = criterion(cla, attack_target_label_r)
        _, ae_preds = torch.max(cla, 1)

        # attack statistics
        attack_loss += ae_loss.item() * len(ori_img)
        attack_corrects += torch.sum(ae_preds == attack_target_label_r)

        if epoch < (attack_epochs - 1):
            net.close_update_epsilon_3d()

            ori_img_list = ori_img.chunk(len(ori_img), dim=0)
            img_index_and_weight_list = img_index_and_weight.chunk(len(ori_img), dim=0)

            for i in range(len(ori_img)):
                cla_same = False

                for i in range(len(ori_img)):
                    pre_cla = torch.argmax(cla[i])
                    pre_ori_cla = torch.argmax(ori_cla[i])
                    if pre_cla == pre_ori_cla:
                        cla_same = True
                        break

                if cla_same:

                    net_input = (change_target_tensor, img_index_and_weight, ori_img)

                    # Finding a new minimal perturbation with deepfool to fool the network on this image
                    if targeted_attack:
                        dr, iter_k, label, k_i, pert_image = deepfool.deepfool(net_input, epsilon, net, max_iter=df_max_iter, target_label=attack_target_label_int, m1=m1, m2=m2)
                    else:
                        dr, iter_k, label, k_i, pert_image = deepfool.deepfool(net_input, epsilon, net, max_iter=df_max_iter, m1=m1, m2=m2)

                    # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                    if iter_k < df_max_iter or accumulated_incomplete_attack:
                        print("iter_k < df_max_iter")
                        change_target_tensor += dr
                        tensor_not_changed = False
                        attack_number_all_after_last_m2_change += 1
                    elif m2 < m2_max_limit:
                        no_attack_number_after_last_m2_change += 1
                        attack_number_all_after_last_m2_change += 1
                        if (attack_number_all_after_last_m2_change > 10 and
                                (no_attack_number_after_last_m2_change/attack_number_all_after_last_m2_change) > 0.5):
                            m2 = m2 * 10
                            print("m2 is too small, times 10 to " + str(m2))
                            no_attack_number_after_last_m2_change = 0
                            attack_number_all_after_last_m2_change = 0

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
        # if m1_index < len(m1_list) - 1:
        #     m1_index += 1
        #     m1 = m1_list[m1_index]
        #     print("m1 from "+str(m1_list[m1_index-1])+" to "+str(m1_list[m1_index]))
        #     epoch = attack_epochs - 1
        # else:

        if m1_list[0] < m1 - 1 and epoch == 0:
            m1_list[1] = m1
            m1 = int((m1 + m1_list[0]) / 2)
            print("m1 from " + str(m1_list[0]) + " to " + str(m1_list[1]) + ", m1=" + str(m1))
            m2 = args.m2
            epoch = 0
        elif m1_list[0] < m1 and epoch == 0:
            m1_list[1] = m1
            m1 = m1_list[0]
            print("m1 from " + str(m1_list[0]) + " to " + str(m1_list[1]) + ", m1=" + str(m1))
            m2 = args.m2
            epoch = 0
        else:
            epoch = attack_epochs - 1
    elif epoch == attack_epochs - 1:
        if m1 < m1_list[1] - 1:
            m1_list[0] = m1
            m1 = int((m1 + m1_list[1]) / 2)
            print("m1 from " + str(m1_list[0]) + " to " + str(m1_list[1]) + ", m1=" + str(m1))
            m2 = args.m2
            epoch = 0
        elif m1 < m1_list[1]:
            m1_list[0] = m1
            m1 = m1_list[1]
            print("m1 from " + str(m1_list[0]) + " to " + str(m1_list[1]) + ", m1=" + str(m1))
            m2 = args.m2
            epoch = 0
        else:
            epoch += 1
    else:
        epoch += 1

    end_epoch = time.time()

    epoch_loss = running_loss / len(now_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(now_dataloader.dataset)

    attack_epoch_loss = attack_loss / len(now_dataloader.dataset)
    attack_epoch_acc = attack_corrects.double() / len(now_dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('attack', attack_epoch_loss, attack_epoch_acc))

    need_to_log_dict["m1-"+str(m1)+" epoch-" + str(epoch)] = 'time: {} Acc: {:.4f}'.format(end_epoch-since_epoch, attack_epoch_acc)

    net.print_epsilon()
    net.epsilon_3d_zero()

    if targeted_attack:
        if (attack_epoch_acc >= attack_epoch_acc_best and m1 == m1_best) or (m1 > m1_best and attack_epoch_acc > 0):
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            m1_best = m1
            m2_best = m2
            change_target_tensor_best = change_target_tensor.clone().detach()
    else:
        if (attack_epoch_acc <= attack_epoch_acc_best and m1 == m1_best) or (m1 > m1_best and attack_epoch_acc < 1):
            attack_epoch_loss_best = attack_epoch_loss
            attack_epoch_acc_best = attack_epoch_acc
            m1_best = m1
            m2_best = m2
            change_target_tensor_best = change_target_tensor.clone().detach()

    # wandb.log({'test Loss': epoch_loss, 'test Acc': epoch_acc}, step=epoch)
    # wandb.log({'attack Loss': attack_epoch_loss, 'attack Acc': attack_epoch_acc}, step=epoch)

change_target_tensor = change_target_tensor.to(torch.device("cpu"))
gauss_img_tensor = change_target_tensor.chunk(len(change_target_img_list), dim=0)

for i in range(len(change_target_img_list)):
    cv2.imwrite(save_target_img_name_list[i], gauss_img_tensor[i].squeeze(0).cpu().detach().numpy())

time_elapsed = time.time() - since
print('Attack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

test_for_inception(print_e=epsilon, model_name=model_name, base_mask_image_number=base_mask_image_number,
                   scene_name=scene_name, target_class_idx=attack_target_label_int,
                   m1=args.m1, m2=args.m2, setname="test", method_name='NeRFail',  step=0, something_need_log=need_to_log_dict)

test_for_inception(print_e=epsilon, model_name=model_name, base_mask_image_number=base_mask_image_number,
                   scene_name=scene_name, target_class_idx=attack_target_label_int,
                   m1=args.m1, m2=args.m2, setname="val", method_name='NeRFail', step=0)
