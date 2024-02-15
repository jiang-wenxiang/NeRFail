## This file source is https://github.com/BXuan694/Universal-Adversarial-Perturbation/blob/master/deepfool.py
## This file is not the scope of the original paper of this project

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy


def deepfool(net_input, e, net, num_classes = 8, max_iter = 20, target_label:int = None,
             overshoot: float=0.02, m1: float=1, m2: float=30, universal_2d = False, clip_e = False):

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

    # setting r approx
    # r_approx_max = (e * 0.1)
    # r_approx_loss_func = torch.nn.MSELoss()

    # # setting optimizer when r approx
    # learning_rata = 10000

    # # setting lower and upper limit for spatial rgb tensor
    # spatial_rgb_alpha = spatial_rgb_0[:, :, :, 3].unsqueeze(-1)
    # spatial_rgb_size = spatial_rgb_0.size()
    #
    # spatial_rgb_zreos = torch.zeros_like(spatial_rgb_0)
    #
    # # spatial_rgb_alpha_mins = torch.zeros_like(spatial_rgb_0) + (1/255)
    # spatial_rgb_alpha_maxs = torch.ones_like(spatial_rgb_0) * 255
    #
    # spatial_rgb_max = torch.ones_like(spatial_rgb_0) * e
    # spatial_rgb_min = torch.ones_like(spatial_rgb_0) * (-e)
    #
    # spatial_rgb_alpha_broadcast = torch.broadcast_to(spatial_rgb_alpha, spatial_rgb_size)
    #
    # max_alpha = torch.where(spatial_rgb_alpha > 0, spatial_rgb_alpha_maxs[:, :, :, 3].unsqueeze(-1), spatial_rgb_zreos[:, :, :, 3].unsqueeze(-1))
    # spatial_rgb_max = torch.where(spatial_rgb_alpha_broadcast > 0, spatial_rgb_max, spatial_rgb_zreos)
    # spatial_rgb_max = torch.cat((spatial_rgb_max[:, :, :, :3], max_alpha), -1)
    #
    # min_alpha = max_alpha # attack do not change alpha channel
    # spatial_rgb_min = torch.where(spatial_rgb_alpha_broadcast > 0, spatial_rgb_min, spatial_rgb_zreos)
    # spatial_rgb_min = torch.cat((spatial_rgb_min[:, :, :, :3], min_alpha), -1)

    if universal_2d:
        _, _, cla, _, ori_cla = net(spatial_rgb, ori_img)
    else:
        _, _, cla, _, ori_cla = net(spatial_rgb, weight_and_index, ori_img)

    cla_max_0, cla_max_index_0 = torch.max(cla, 1)
    ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

    rot = torch.zeros_like(spatial_rgb)

    grad_cla_s_0 = torch.autograd.grad(cla_max_0, spatial_rgb, retain_graph=False, create_graph=False)[0]

    # r_k = r_0.clone().detach()
    #
    # # setting the lower and upper limit for r
    # r_k_init = r_0.clone().detach()
    # r_k_alpha = r_k_init[:, :, :, 3].unsqueeze(-1)
    # r_k_size = r_k.size()
    #
    # r_k_zreos = torch.zeros_like(r_k)
    # r_k_max = torch.ones_like(r_k) * e
    # r_k_min = torch.ones_like(r_k) * (-e)
    #
    # r_k_alpha_broadcast = torch.broadcast_to(r_k_alpha, r_k_size)
    #
    # r_k_max = torch.where(r_k_alpha_broadcast > 0, r_k_max, r_k_zreos)
    # r_k_min = torch.where(r_k_alpha_broadcast > 0, r_k_min, r_k_zreos)

    loop_i = 0

    while loop_i < max_iter:

        # if torch.isnan(torch.sum(spatial_rgb)):
        #     print("spatial_rgb nan before attack! ")

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

        if clip_e:
            spatial_rgb = torch.clamp(spatial_rgb, -e, e)
            spatial_rgb = torch.cat([spatial_rgb[:, :, :, :3], spatial_rgb_0[:, :, :, 3].unsqueeze(-1)], -1)

        # spatial_rgb = torch.clamp(spatial_rgb, min=spatial_rgb_min)
        # spatial_rgb = torch.clamp(spatial_rgb, max=spatial_rgb_max)

        loop_i += 1

    # d_spa = (spatial_rgb - spatial_rgb_0).detach()

    # loop_j = 0
    # d_spa = torch.zeros_like(spatial_rgb_0)
    # spatial_rgb = Variable(spatial_rgb, requires_grad=True)
    # # optimizer = torch.optim.Adam([spatial_rgb], lr=learning_rata)
    #
    # while loop_i < max_iter and loop_j < r_approx_max_iter:
    #
    #     spatial_rgb = Variable(spatial_rgb, requires_grad=True)
    #     # optimizer.zero_grad()
    #
    #     r_0 = net_r(spatial_rgb, dist_and_index_list)
    #
    #     r_approx = torch.max(torch.abs(r_k[:, :, :, :3] - r_0[:, :, :, :3]))
    #     if r_approx <= r_approx_max:
    #         break
    #
    #     approx_loss = r_approx_loss_func(r_0, r_k)
    #     approx_loss.backward()
    #
    #     # optimizer.step()
    #
    #     spatial_rgb_grad = spatial_rgb.grad.data
    #
    #     spatial_rgb = (spatial_rgb - (learning_rata * spatial_rgb_grad)).detach()
    #     # spatial_rgb = torch.clamp(spatial_rgb, spatial_rgb_min, spatial_rgb_max).detach()
    #     # spatial_rgb = torch.cat([spatial_rgb[:, :, :, :3], spatial_rgb_alpha], -1)
    #
    #     d_spa = (spatial_rgb - spatial_rgb_0).detach()
    #
    #     loop_j += 1
    #
    #     # if loop_j == 1:
    #     #     print("----------"+str(loop_j)+"-----------")
    #     #     print(r_approx)
    #     #     print(approx_loss)
    #     #     print(grad_max)
    #     #
    #     if loop_j == r_approx_max_iter:
    #         print("----------"+str(loop_j)+"-----------")
    #         print(r_approx)
    #         print(approx_loss)
    #         # print(grad_max)

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

    # setting r approx
    # r_approx_max = (e * 0.1)
    # r_approx_loss_func = torch.nn.MSELoss()

    # # setting optimizer when r approx
    # learning_rata = 10000

    # # setting lower and upper limit for spatial rgb tensor
    # spatial_rgb_alpha = spatial_rgb_0[:, :, :, 3].unsqueeze(-1)
    # spatial_rgb_size = spatial_rgb_0.size()
    #
    # spatial_rgb_zreos = torch.zeros_like(spatial_rgb_0)
    #
    # # spatial_rgb_alpha_mins = torch.zeros_like(spatial_rgb_0) + (1/255)
    # spatial_rgb_alpha_maxs = torch.ones_like(spatial_rgb_0) * 255
    #
    # spatial_rgb_max = torch.ones_like(spatial_rgb_0) * e
    # spatial_rgb_min = torch.ones_like(spatial_rgb_0) * (-e)
    #
    # spatial_rgb_alpha_broadcast = torch.broadcast_to(spatial_rgb_alpha, spatial_rgb_size)
    #
    # max_alpha = torch.where(spatial_rgb_alpha > 0, spatial_rgb_alpha_maxs[:, :, :, 3].unsqueeze(-1), spatial_rgb_zreos[:, :, :, 3].unsqueeze(-1))
    # spatial_rgb_max = torch.where(spatial_rgb_alpha_broadcast > 0, spatial_rgb_max, spatial_rgb_zreos)
    # spatial_rgb_max = torch.cat((spatial_rgb_max[:, :, :, :3], max_alpha), -1)
    #
    # min_alpha = max_alpha # attack do not change alpha channel
    # spatial_rgb_min = torch.where(spatial_rgb_alpha_broadcast > 0, spatial_rgb_min, spatial_rgb_zreos)
    # spatial_rgb_min = torch.cat((spatial_rgb_min[:, :, :, :3], min_alpha), -1)

    _, _, cla, _, ori_cla = net(ori_img, spatial_rgb)

    cla_max_0, cla_max_index_0 = torch.max(cla, 1)
    ori_cla_max, ori_cla_max_index = torch.max(ori_cla, 1)

    rot = torch.zeros_like(spatial_rgb)

    grad_cla_s_0 = torch.autograd.grad(cla_max_0, spatial_rgb, retain_graph=False, create_graph=False)[0]

    # r_k = r_0.clone().detach()
    #
    # # setting the lower and upper limit for r
    # r_k_init = r_0.clone().detach()
    # r_k_alpha = r_k_init[:, :, :, 3].unsqueeze(-1)
    # r_k_size = r_k.size()
    #
    # r_k_zreos = torch.zeros_like(r_k)
    # r_k_max = torch.ones_like(r_k) * e
    # r_k_min = torch.ones_like(r_k) * (-e)
    #
    # r_k_alpha_broadcast = torch.broadcast_to(r_k_alpha, r_k_size)
    #
    # r_k_max = torch.where(r_k_alpha_broadcast > 0, r_k_max, r_k_zreos)
    # r_k_min = torch.where(r_k_alpha_broadcast > 0, r_k_min, r_k_zreos)

    loop_i = 0

    while loop_i < max_iter:

        # if torch.isnan(torch.sum(spatial_rgb)):
        #     print("spatial_rgb nan before attack! ")

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
        # spatial_rgb = torch.clamp(spatial_rgb, min=spatial_rgb_min)
        # spatial_rgb = torch.clamp(spatial_rgb, max=spatial_rgb_max)

        loop_i += 1

    # d_spa = (spatial_rgb - spatial_rgb_0).detach()

    # loop_j = 0
    # d_spa = torch.zeros_like(spatial_rgb_0)
    # spatial_rgb = Variable(spatial_rgb, requires_grad=True)
    # # optimizer = torch.optim.Adam([spatial_rgb], lr=learning_rata)
    #
    # while loop_i < max_iter and loop_j < r_approx_max_iter:
    #
    #     spatial_rgb = Variable(spatial_rgb, requires_grad=True)
    #     # optimizer.zero_grad()
    #
    #     r_0 = net_r(spatial_rgb, dist_and_index_list)
    #
    #     r_approx = torch.max(torch.abs(r_k[:, :, :, :3] - r_0[:, :, :, :3]))
    #     if r_approx <= r_approx_max:
    #         break
    #
    #     approx_loss = r_approx_loss_func(r_0, r_k)
    #     approx_loss.backward()
    #
    #     # optimizer.step()
    #
    #     spatial_rgb_grad = spatial_rgb.grad.data
    #
    #     spatial_rgb = (spatial_rgb - (learning_rata * spatial_rgb_grad)).detach()
    #     # spatial_rgb = torch.clamp(spatial_rgb, spatial_rgb_min, spatial_rgb_max).detach()
    #     # spatial_rgb = torch.cat([spatial_rgb[:, :, :, :3], spatial_rgb_alpha], -1)
    #
    #     d_spa = (spatial_rgb - spatial_rgb_0).detach()
    #
    #     loop_j += 1
    #
    #     # if loop_j == 1:
    #     #     print("----------"+str(loop_j)+"-----------")
    #     #     print(r_approx)
    #     #     print(approx_loss)
    #     #     print(grad_max)
    #     #
    #     if loop_j == r_approx_max_iter:
    #         print("----------"+str(loop_j)+"-----------")
    #         print(r_approx)
    #         print(approx_loss)
    #         # print(grad_max)

    return (overshoot*rot), loop_i, ori_cla_max_index, cla_max_index, spatial_rgb