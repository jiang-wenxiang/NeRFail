## This file source is https://github.com/BXuan694/Universal-Adversarial-Perturbation/blob/master/deepfool.py
## This file is not the scope of the original paper of this project

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy


def deepfool(net_input, e, net, num_classes = 8, max_iter = 20, target_label:int = None,
             overshoot: float=0.02, m1: float=1, m2: float=30, universal_2d = False):

    """
       :param image:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    if universal_2d:
        spatial_rgb, ori_img = net_input
    else:
        spatial_rgb, weight_and_index, ori_img = net_input

    spatial_rgb_0 = spatial_rgb.clone().detach()
    spatial_rgb = Variable(spatial_rgb, requires_grad=True)

    if universal_2d:
        _, _, cla, _, ori_cla = net(spatial_rgb, ori_img)
    else:
        _, _, cla, _, ori_cla = net(spatial_rgb, weight_and_index, ori_img)

    cla_max_0, cla_max_index_0 = torch.max(cla, 1)
    ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

    rot = torch.zeros_like(spatial_rgb)

    grad_cla_s_0 = torch.autograd.grad(cla_max_0, spatial_rgb, retain_graph=False, create_graph=False)[0]

    loop_i = 0

    while loop_i < max_iter:

        spatial_rgb = Variable(spatial_rgb, requires_grad=True)

        if universal_2d:
            _, _, cla, _, ori_cla = net(spatial_rgb, ori_img)
        else:
            _, _, cla, _, ori_cla = net(spatial_rgb, weight_and_index, ori_img)

        if target_label is None:
            cla[:, int(ori_cla_max_index)] = cla[:, int(ori_cla_max_index)] + m1  # add m1
        else:
            cla[:, :int(target_label)] = cla[:, :int(target_label)] + m1  # add m1
            cla[:, int(target_label) + 1:] = cla[:, int(target_label) + 1:] + m1  # add m1

        cla_max, cla_max_index = torch.max(cla, 1)
        ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

        if (target_label is None) and (cla_max_index != ori_cla_max_index):
            break

        if (target_label is not None) and (cla_max_index == target_label):
            break

        min_value = torch.inf
        dr = torch.zeros_like(rot)

        if target_label is None:
            for k in range(num_classes):
                if k == ori_cla_max_index:
                    continue

                grad_cla_s_k = torch.autograd.grad(cla[:, k], spatial_rgb, retain_graph=True, create_graph=True)[0]
                grad_cla_s_o = torch.autograd.grad(cla[:, int(ori_cla_max_index)], spatial_rgb, retain_graph=True, create_graph=True)[0]

                f_prime = cla[:, k] - (cla[:, int(ori_cla_max_index)] + m2)
                grad_cla_s_prime = grad_cla_s_k - grad_cla_s_o

                value_r = torch.abs(f_prime) / (torch.norm(grad_cla_s_prime) + 0.0001)

                if value_r < min_value:
                    dr = ((torch.abs(f_prime) / ((torch.norm(grad_cla_s_prime) ** 2)+0.0001)) * grad_cla_s_prime)
                    min_value = value_r

        else:
            k = target_label
            grad_cla_s_k = torch.autograd.grad(cla[:, k], spatial_rgb, retain_graph=True, create_graph=True)[0]
            grad_cla_s_o = torch.autograd.grad(cla[:, int(ori_cla_max_index)], spatial_rgb, retain_graph=True, create_graph=True)[0]

            f_prime = cla[:, k] - (cla[:, int(ori_cla_max_index)] + m2)  # add m2
            grad_cla_s_prime = grad_cla_s_k - grad_cla_s_o

            dr = ((torch.abs(f_prime) / ((torch.norm(grad_cla_s_prime) ** 2)+0.0001)) * grad_cla_s_prime)

        rot = (rot + dr).detach()

        spatial_rgb = (spatial_rgb_0 + (overshoot*rot)).detach()
        spatial_rgb = torch.clamp(spatial_rgb, -255, 255)

        # alpha channel is not changed
        if not universal_2d:
            spatial_rgb = torch.cat([spatial_rgb[:, :, :, :3], spatial_rgb_0[:, :, :, 3].unsqueeze(-1)], -1)

        loop_i += 1

    rot = (spatial_rgb - spatial_rgb_0).detach()

    return rot, loop_i, ori_cla_max_index, cla_max_index, spatial_rgb


def deepfool_2D_universal(net_input, e, net, num_classes = 8, max_iter = 20, target_label:int = None, overshoot: float=0.02, m1: float=1, m2: float=30):

    """
       :param image:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    ori_img, spatial_rgb = net_input

    spatial_rgb_0 = spatial_rgb.clone().detach()
    spatial_rgb = Variable(spatial_rgb, requires_grad=True)

    _, _, cla, _, ori_cla = net(ori_img, spatial_rgb)

    cla_max_0, cla_max_index_0 = torch.max(cla, 1)
    ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

    rot = torch.zeros_like(spatial_rgb)

    grad_cla_s_0 = torch.autograd.grad(cla_max_0, spatial_rgb, retain_graph=False, create_graph=False)[0]

    loop_i = 0

    while loop_i < max_iter:

        spatial_rgb = Variable(spatial_rgb, requires_grad=True)

        _, _, cla, _, ori_cla = net(ori_img, spatial_rgb)

        cla[:, int(ori_cla_max_index)] = cla[:, int(ori_cla_max_index)] + m1 # add

        cla_max, cla_max_index = torch.max(cla, 1)
        ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

        if cla_max_index != ori_cla_max_index:
            break

        min_value = torch.inf
        dr = torch.zeros_like(rot)

        if target_label is None:
            for k in range(num_classes):
                if k == ori_cla_max_index:
                    continue

                grad_cla_s_k = torch.autograd.grad(cla[:, k], spatial_rgb, retain_graph=True, create_graph=True)[0]
                grad_cla_s_o = torch.autograd.grad(cla[:, int(ori_cla_max_index)], spatial_rgb, retain_graph=True, create_graph=True)[0]

                f_prime = cla[:, k] - (cla[:, int(ori_cla_max_index)] + m2)
                grad_cla_s_prime = grad_cla_s_k - grad_cla_s_o

                value_r = torch.abs(f_prime) / (torch.norm(grad_cla_s_prime) + 0.0001)

                if value_r < min_value:
                    dr = (torch.abs(f_prime) / ((torch.norm(grad_cla_s_prime) ** 2)+0.0001)) * grad_cla_s_prime
                    min_value = value_r

        else:
            k = target_label
            grad_cla_s_k = torch.autograd.grad(cla[:, k], spatial_rgb, retain_graph=True, create_graph=True)[0]
            grad_cla_s_o = torch.autograd.grad(cla[:, int(ori_cla_max_index)], spatial_rgb, retain_graph=True, create_graph=True)[0]

            f_prime = cla[:, k] -  (cla[:, int(ori_cla_max_index)] + m2)
            grad_cla_s_prime = grad_cla_s_k - grad_cla_s_o

            dr = ((torch.abs(f_prime) / ((torch.norm(grad_cla_s_prime) ** 2)+0.0001)) * grad_cla_s_prime)

        rot = (rot + dr).detach()
        spatial_rgb = (spatial_rgb_0 + (overshoot*rot)).detach()

        loop_i += 1

    return (overshoot*rot), loop_i, ori_cla_max_index, cla_max_index, spatial_rgb