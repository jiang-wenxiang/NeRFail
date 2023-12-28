import math
import os
import time
from pathlib import Path

import configargparse
import numpy as np
import torch.nn.functional as F
import torch
import tqdm
from torch import nn
from torchvision import models
from torchvision.models import alexnet, vgg16, vit_b_16, resnet50, densenet121, efficientnet_b0, swin_b
from torchvision.models.inception import Inception3
from torchvision.transforms import Resize

from MyDataset import MySimpleDataset
from model.MyModel import MyCNN
from tools.send_e_mail import send_dict

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_psnr(target, ref):
    # target_data = np.array(target, dtype=np.float64)  # 将图像格式转为 float64
    # ref_data = np.array(ref, dtype=np.float64)
    # 直接相减，求差值
    diff = ref - target
    # 按第三个通道顺序把三维矩阵拉平
    diff = torch.flatten(diff)
    # 计算MSE值
    rmse = torch.sqrt(torch.mean(diff ** 2.))
    # 精度
    eps = torch.finfo(torch.float64).eps
    if rmse == 0:
        rmse = eps
    return 20*math.log10(255.0/rmse)

def test_for_inception(scene_name = "lego", model_name = "vgg16", now_class_idx = 10, attack_epochs = 100,
                       print_a = 2, print_e = 32, target_class_idx = "n", base_mask_image_number = 3, m1=0, m2=0,
                       setname = "test", method_name = "NeRFail", step=0, something_need_log: dict = None):
    # print config

    # test config
    three_channels_list = []
    class_list = {"chair": 0, "drums": 1, "ficus": 2, "hotdog": 3, "lego": 4, "materials": 5, "mic": 6, "ship": 7}
    # class_list = {"animal": 0, "bottled": 1, "box": 2, "car": 3, "cup": 4, "house_furnishings": 5, "icon": 6,
    #               "office_supplies": 7, "person": 8, "plate": 9, "shoe": 10, "toy": 11, "wear":12}

    only_print_class = [class_list[scene_name], ]
    # if setname == "test":
    #     ori_img_from = {class_list[scene_name]: "./data/"+setname+"_ori_img/" + scene_name}
    # else:
    #     ori_img_from = {class_list[scene_name]: "./data/"+setname+"/" + scene_name}

    ori_img_from = {class_list[scene_name]: "./data/nerf_synthetic/" + scene_name + "/" + setname}

    if method_name is None:
        only_print_class = []
        ori_img_from = {}

    if step == 0:
        step_name = "attack"
    elif step == 1:
        step_name = "nerf"
    elif step == 2:
        step_name = "defense"
    elif step == 3:
        step_name = "nerf_defense"

    _3_channels = True
    resize_frame = True
    load_later = False

    if method_name is None:
        print_tab = "test for model: " + model_name
    else:
        if method_name == "IGSM":
            method_name = "NeRFail-S"
        elif method_name == "Universal":
            method_name = "NeRFail"

        print_method_name = method_name

        print_tab = print_method_name + " " + step_name + " "+model_name+" "+str(base_mask_image_number)+\
                    "P "+str(class_list[scene_name]) +" to "+str(target_class_idx)+\
                    " and a="+str(print_a)+", e="+str(print_e)+", m1="+str(m1)+", m2="+str(m2)+", set="+setname

    if method_name is None:
        test_dir_change_dict = None
    # test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/attack/' + scene_name +
    #                            '/IGSM_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(target_class_idx)+'_e_'+str(print_e)+'_a_'+str(print_a)+'/{}'.format("test")}

    if method_name == "NeRFail":
        test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/'+step_name+'/' + scene_name +
                                   '/'+method_name+'_'+str(base_mask_image_number)+'P_'+str(attack_epochs)
                                   +'_to_'+str(target_class_idx)+'_e_'+str(print_e)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format(setname)}
    elif method_name == "NeRFail_S":
        test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/'+step_name+'/' + scene_name +
                                   '/'+method_name+'_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+
                                   str(target_class_idx)+'_e_'+str(print_e)+'_a_'+str(print_a)+'/{}'.format(setname)}

    elif method_name == "IGSM_2D":
        test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/'+step_name+'/' + scene_name +
                                   '/'+method_name+'_'+str(attack_epochs)+'_to_'+str(target_class_idx)+
                                   '_e_'+str(print_e)+'_a_'+str(print_a)+'/{}'.format(setname)}

    elif method_name == "No_attack" and step == 0:
        test_dir_change_dict = None

    elif method_name == "No_attack":
        test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/'+step_name+'/' + scene_name +
                                   '/no_attack/{}'.format(setname)}

    elif method_name == "Universal_2D":
        test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/'+step_name+'/' + scene_name +
                                    '/'+'Universal_2D_'+str(attack_epochs)+'_to_'+str(target_class_idx)+'_e_'+
                                   str(print_e)+'_m_'+str(m1)+'_'+str(m2)+'/{}'.format(setname)}

    # test_dir_change_dict = {class_list[scene_name]: './output/'+model_name+'/nerf/' + scene_name +
    #                            '/IGSM_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_'+str(target_class_idx)+'_e_'+str(print_e)+'_a_'+str(print_a)+'/{}'.format("test")}

    num_classes = len(class_list.keys())
    # inception v3 model
    if model_name == "inception":
        # inception v3 model
        model = Inception3(num_classes=num_classes, init_weights=False)
    elif model_name == "vgg16":
        model = vgg16(num_classes=num_classes, init_weights=False)
    elif model_name == "alexnet":
        model = alexnet(num_classes=num_classes)
    elif model_name == "vit_b_16":
        model = vit_b_16(num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    elif model_name == "densenet121":
        model = densenet121(num_classes=num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = efficientnet_b0(num_classes=num_classes)
    elif model_name == "swin_b":
        model = swin_b(num_classes=num_classes)
    else:
        model = MyCNN(num_classes=num_classes)

    model.to(device)

    # config
    data_dir = Path("./data/nerf_synthetic")
    test_dir_name = model_name
    batch_size = 1

    # load pretrain weight
    weights_path = Path("./model/weights/"+model_name+"_"+str(num_classes)+"_best.pth")
    print(weights_path.exists())
    print(weights_path.absolute())
    pretrain_model = torch.load(weights_path.absolute(), map_location=device)
    model.load_state_dict(pretrain_model)

    # test
    since = time.time()
    val_acc_history = []

    print("Start test for AE model = "+model_name+" model!")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # sample execution (requires torchvision)

    print("Initializing Datasets and Dataloaders...")

    torch_resize_299 = Resize([299, 299])
    torch_resize_224 = Resize([224, 224])
    torch_resize_800 = Resize([800, 800])

    resize_func = None

    if model_name == "vit_b_16":
        # resize image from 800x800 to 224x224
        resize_func = torch_resize_224
    elif model_name == "my_model":
        resize_func = torch_resize_800
    else:
        # resize image from 800x800 to 299x299
        resize_func = torch_resize_299

    # Create test datasets
    image_datasets = {x: MySimpleDataset(data_dir, x, test_dir_change_dict=test_dir_change_dict,
                                         ori_img_from=ori_img_from, device=device, _3_channels=_3_channels,
                                         resize_frame=resize_frame, load_later=load_later, resize_frame_size=resize_func) for x in ['test']}
    # Create test dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ['test']}

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    class_image_length = {}

    running_loss_dict = {}
    running_corrects_dict = {}
    wrong_class_label = {}

    e_dict = {}

    i = -1

    # Iterate over data.
    for dataload in tqdm.tqdm(dataloaders['test']):
        inputs, labels, ori_img = dataload

        if int(labels.data) not in ori_img_from.keys():
            ori_img = None
        else:
            ori_img = ori_img.to(device)

        inputs = inputs.to(device)
        labels = labels.to(device)

        if inputs.size()[1] != 3:
            inputs_size = inputs.size()
            inputs_alpha = inputs[:, 3, :, :].unsqueeze(1).broadcast_to([inputs_size[0], 3,
                                                                         inputs_size[2], inputs_size[3]])
            inputs_rgb = inputs[:, :3, :, :]
            tensor_255 = torch.ones_like(inputs_rgb) * 255
            inputs = torch.where(inputs_alpha > 0, inputs_rgb, tensor_255)

        if ori_img is not None:
            if ori_img.size()[1] != 3:
                ori_img_size = ori_img.size()
                ori_img_alpha = ori_img[:, 3, :, :].unsqueeze(1).broadcast_to([ori_img_size[0], 3,
                                                                               ori_img_size[2], ori_img_size[3]])
                ori_img_rgb = ori_img[:, :3, :, :]
                tensor_255 = torch.ones_like(ori_img_rgb) * 255
                ori_img = torch.where(ori_img_alpha > 0, ori_img_rgb, tensor_255)

        if ori_img is not None:
            dist_mask = inputs - ori_img
            dist_mask = torch.abs(dist_mask)
            dist_mask_size = dist_mask.size()

            e_max = torch.max(dist_mask)
            e_avg = torch.sum(dist_mask) / (dist_mask_size[0] * dist_mask_size[1] * dist_mask_size[2] * dist_mask_size[3])
            e_min = torch.min(dist_mask)
            d_l2 = torch.dist(inputs, ori_img, p=2)
            d_l0 = torch.count_nonzero(dist_mask)
            psnr = get_psnr(inputs, ori_img)

            if str(int(labels.data)) in e_dict.keys():
                e_dict[str(int(labels.data))]["e_max"] = torch.max(torch.stack([e_dict[str(int(labels.data))]["e_max"], e_max]))
                e_dict[str(int(labels.data))]["e_avg"] += e_avg * inputs.size(0)
                e_dict[str(int(labels.data))]["e_min"] = torch.min(torch.stack([e_dict[str(int(labels.data))]["e_min"], e_min]))
                e_dict[str(int(labels.data))]["d_l2"] += d_l2
                e_dict[str(int(labels.data))]["d_l0"] += d_l0
                e_dict[str(int(labels.data))]["psnr_list"].append(psnr)

            else:
                e_dict[str(int(labels.data))] = {"e_max": e_max, "e_avg": e_avg, "e_min": e_min,
                                                 "d_l2": d_l2, "d_l0": d_l0, "psnr_list": [psnr]}

        i += 1
        with torch.set_grad_enabled(False):

            if model_name == "vit_b_16":
                # resize image from 800x800 to 224x224
                inputs = torch_resize_224(inputs)
            elif model_name == "my_model":
                pass
            else:
                # resize image from 800x800 to 299x299
                inputs = torch_resize_299(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if str(int(labels.data)) in running_loss_dict.keys():
            running_loss_dict[str(int(labels.data))] += loss.item() * inputs.size(0)
            running_corrects_dict[str(int(labels.data))] += torch.sum(preds == labels.data)
            class_image_length[str(int(labels.data))] += inputs.size(0)
        else:
            running_loss_dict[str(int(labels.data))] = loss.item() * inputs.size(0)
            running_corrects_dict[str(int(labels.data))] = torch.sum(preds == labels.data)
            class_image_length[str(int(labels.data))] = inputs.size(0)

        if preds != labels.data:

            if len(only_print_class) > 0 and int(labels.data) in only_print_class:
                if setname == "test":
                    print_list = []
                    # print_list = [0, 13, 25, 38, 50, 63, 75, 88, 59]
                    if (i % 200) in print_list:
                        print("test data index=r_" + str(i % 200) + ", target=" + str(labels.data) + " and output=" + str(preds))
                        prod = (torch.exp(outputs) / torch.exp(outputs).sum())
                        print("prod_max: ", torch.max(prod))
                else:
                    print_list = []
                    if (i % 100) in print_list:
                        print("val data index=r_" + str(i % 100) + ", target=" + str(labels.data) + " and output=" + str(preds))
                        prod = (torch.exp(outputs) / torch.exp(outputs).sum())
                        print("prod_max: ", torch.max(prod))

            elif len(only_print_class) == 0:
                print("test data index=r_" + str(i % 200) + ", target=" + str(labels.data) + " and output=" + str(preds))

        if int(labels.data) in only_print_class:
            if str(int(labels.data)) in wrong_class_label.keys():
                if str(int(preds)) in wrong_class_label[str(int(labels.data))].keys():
                    wrong_class_label[str(int(labels.data))][str(int(preds))] += 1
                else:
                    wrong_class_label[str(int(labels.data))][str(int(preds))] = 1
            else:
                wrong_class_label[str(int(labels.data))] = {str(int(preds)): 1}

    epoch_loss = running_loss / len(dataloaders['test'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(test_dir_name, epoch_loss, epoch_acc))

    msg_dict = {test_dir_name: 'Loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc)}

    print("------------------------"+print_tab+"-----------------------------")
    if len(only_print_class) > 0:
        for key in only_print_class:
            wrong_class_labels = wrong_class_label[str(key)]
            number = 0
            # msg_dict = {}
            for wk in wrong_class_labels:
                number += wrong_class_labels[wk]

            if str(key) not in wrong_class_labels.keys():
                no_target_attack_success = "100.0 %"
                msg_dict = {"attack acc": no_target_attack_success}
                print("no targeted attack: " + no_target_attack_success)
            else:
                no_target_attack_success = str((1 - wrong_class_labels[str(key)] / number) * 100) + " %"
                msg_dict = {"attack acc": no_target_attack_success}
                print("no targeted attack: " + no_target_attack_success)

            for wk in wrong_class_labels:
                msg_dict[str(key) + " to " + str(wk)] = str((wrong_class_labels[wk]/number) * 100) + " %"
                print("ground truth class label: " + str(key) + ", now class label: "+wk+"  ---  "+str((wrong_class_labels[wk]/number) * 100)+" %")

            if str(key) in e_dict.keys():
                print(">>>>>>>>  e min: ", float(e_dict[str(key)]["e_min"]))
                print(">>>>>>>>  e avg: ", float(e_dict[str(key)]["e_avg"] / number))
                print(">>>>>>>>  e max: ", float(e_dict[str(key)]["e_max"]))
                print(">>>>>>>>  d l2 norm: ", float(e_dict[str(key)]["d_l2"] / number))
                print(">>>>>>>>  d l0 norm: ", float(e_dict[str(key)]["d_l0"] / number))
                print(">>>>>>>>  psnr min: ", min(e_dict[str(key)]["psnr_list"]))
                print(">>>>>>>>  psnr max: ", max(e_dict[str(key)]["psnr_list"]))
                print(">>>>>>>>  psnr avg: ", sum(e_dict[str(key)]["psnr_list"]) / number)

                # send an e-mail
                msg_dict.update({
                    "e min": float(e_dict[str(key)]["e_min"]),
                    "e avg": float(e_dict[str(key)]["e_avg"] / number),
                    "e max": float(e_dict[str(key)]["e_max"]),
                    "d l2 norm": float(e_dict[str(key)]["d_l2"] / number),
                    "d l0 norm": float(e_dict[str(key)]["d_l0"] / number),
                    "psnr min": min(e_dict[str(key)]["psnr_list"]),
                    "psnr max": max(e_dict[str(key)]["psnr_list"]),
                    "psnr avg": sum(e_dict[str(key)]["psnr_list"]) / number,
                })

            if something_need_log is not None:
                msg_dict.update(something_need_log)

            send_dict(print_tab, msg_dict)

    else:
        for i in range(len(running_loss_dict.keys())):
            key = str(i)
            class_loss = running_loss_dict[key] / class_image_length[key]
            class_acc = running_corrects_dict[key].double() / class_image_length[key]
            print('class {} Loss: {:.6f} Acc: {:.6f}'.format(key, class_loss, class_acc))
            msg_dict[key] = 'Loss: {:.6f} Acc: {:.6f}'.format(class_loss, class_acc)

        if something_need_log is not None:
            msg_dict.update(something_need_log)

        send_dict(print_tab, msg_dict)
    print("------------------------"+print_tab+"-----------------------------")

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--e', type=int, default=32)
    parser.add_argument('--a', type=int, default=2)
    parser.add_argument('--m1', type=int, default=8)
    parser.add_argument('--m2', type=int, default=100)
    parser.add_argument('--base_mask_image_number', type=int, default=3)
    parser.add_argument('--setname', type=str, default='test')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--method_name', type=str, default=None)
    parser.add_argument('--label', type=str, default='lego')
    parser.add_argument('--model_name', type=str, default='inception')
    parser.add_argument('--target_class_idx', default='n')

    args = parser.parse_args()

    test_for_inception(print_e=args.e, print_a=args.a, model_name=args.model_name,
                       base_mask_image_number=args.base_mask_image_number, m1=args.m1,
                       m2=args.m2, setname=args.setname, step=args.step,
                       method_name=args.method_name, target_class_idx=args.target_class_idx,
                       scene_name=args.label)

    # print_e_list = [32, 32, 32, 32]
    # base_mask_image_number_list = [3, 3, 3, 3]
    # m1_list = [8, 8, 8, 8]
    # m2_list = [100, 100, 100, 100]
    # setname_list = ['test', 'val', 'test', 'val']
    # # setname_list = ['val']
    # step_list = [0, 0, 1, 1]

    # method_name = "Universal_2D"
    # method_name = "Universal"
    # method_name = "IGSM"
    # method_name = "IGSM_2D"
    # method_name = "No_attack"
    # method_name = "No_attack"

    # model_name = "inception"
    # model_name = "alexnet"
    # model_name = "resnet50"
    # model_name = "vgg16"
    # model_name = "inception"
    # model_name = "vit_b_16"
    # model_name = "mobilenet"
    # model_name = "efficientnet"

    # target_class_idx = 'n'
    # target_class_idx = 0
    # scene_name = "lego"
    # scene_name = "ficus"

    # for print_e, base_mask_image_number, m1, m2, setname, step in zip(print_e_list, base_mask_image_number_list, m1_list,
    #                                                             m2_list, setname_list, step_list):
    #     test_for_inception(print_e=print_e, model_name=model_name, base_mask_image_number=base_mask_image_number, m1=m1, m2=m2,
    #                        setname=setname, step=step, method_name=method_name, target_class_idx=target_class_idx,
    #                        scene_name=scene_name)


    # test_for_inception(model_name="inception", step=1, method_name="No_attack", setname="val")
    # test_for_inception(model_name="vgg16", step=1, method_name="No_attack", setname="val")
    # test_for_inception(model_name="alexnet", step=1, method_name="No_attack", setname="val")
    # test_for_inception(model_name="vit_b_16", step=1, method_name="No_attack", setname="val")


    # test_for_inception(model_name="resnet50", method_name=None)
    # test_for_inception(model_name="efficientnet", method_name=None)
    # test_for_inception(model_name="densenet121", method_name=None)
    # test_for_inception(model_name="mobilenet", method_name=None)
    # test_for_inception(model_name="swin_b", method_name=None)
    # test_for_inception(model_name="inception", method_name=None)
