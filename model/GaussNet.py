import torch
from torch import nn
from torchvision.transforms import Resize


# part 2: gauss net to compute mask for img
class gauss_net(nn.Module):
    def __init__(self, device, c, model, model_name, epsilon=None):
        super(gauss_net, self).__init__()
        self.top_number = 8
        self.c = torch.nn.Parameter(torch.tensor([c]), requires_grad=False)
        self.device = device
        self.model = model
        self.model_name = model_name
        self.torch_resize_299 = Resize([299, 299])
        self.torch_resize_224 = Resize([224, 224])
        # self.epsilon = epsilon
        self.theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float, device=device)
        self.w = 299
        self.h = 299

        self.epsilon = epsilon

        self.epsilon_3d_max = 0
        self.epsilon_3d_min = 0

        self.update_epsilon_3d = True

    def epsilon_3d_zero(self):
        self.epsilon_3d_max = 0
        self.epsilon_3d_min = 0

    def close_update_epsilon_3d(self):
        self.update_epsilon_3d = False

    def open_update_epsilon_3d(self):
        self.update_epsilon_3d = True

    def print_epsilon(self):
        print("epsilon_3d_min: ", self.epsilon_3d_min)
        print("epsilon_3d_max: ", self.epsilon_3d_max)

    def forward(self, spatial_rgb, weight_and_index_list, ori_img, zero_init_mask: bool = False):

        # if zero_init_mask:
        #     s_zero = torch.reshape(spatial_rgb, (-1, 4))
        #     s = s_zero.clone()
        #     s[:, :3] = s_zero[:, :3] - 100
        # else:
        s = torch.reshape(spatial_rgb, (-1, 4))

        ori_img = torch.tensor(ori_img, dtype=torch.float)

        w = weight_and_index_list[:, 0, :, :, :]
        x = weight_and_index_list[:, 1, :, :, :]

        shape_temp = [x.size()[0], x.size()[1], x.size()[2], x.size()[3], 4]
        x = torch.reshape(x, (-1,)).unsqueeze(-1)
        x = torch.broadcast_to(x, (x.size()[0], 4)).type(torch.long)
        x = s.gather(dim=0, index=x)

        # alpha = x[:, 3].unsqueeze(-1) / 255
        # epsilon_x_max = float(torch.max(x[:, :3] * alpha).data)
        # epsilon_x_min = float(torch.min(x[:, :3] * alpha).data)

        x = torch.reshape(x, shape_temp)
        w = torch.broadcast_to(w.unsqueeze(-1), shape_temp)

        # d = -(torch.square(d / self.c) / 2)
        # alpha = x[:, :, :, :, 3].unsqueeze(-1) / 255
        # d = torch.exp(d).unsqueeze(-1)
        # ds = torch.sum(d, dim=-2).unsqueeze(-1)
        # ds = torch.broadcast_to(ds, (x.size()))
        # dq = ds + 0.001
        # zeros = torch.zeros_like(ds).to(self.device)

        # x = torch.where(ds > 0, (x * d) / dq, zeros)
        x = x * w

        x = torch.sum(x, dim=-2)

        alpha = x[:, :, :, 3].unsqueeze(-1) / 255

        ori_img_alpha = ori_img[:, :, :, 3].unsqueeze(-1)

        if self.update_epsilon_3d:
            # test the epsilon when create attack img
            alpha_x_size = alpha.broadcast_to(x[:, :, :, :3].size())
            zreos = torch.zeros_like(x[:, :, :, :3])

            x_zeros = torch.where(alpha_x_size > 0, x[:, :, :, :3], zreos)

            epsilon_x_max = float(torch.max(x_zeros[:, :, :, :3] * alpha).data)
            epsilon_x_min = float(torch.min(x_zeros[:, :, :, :3] * alpha).data)
            # test the epsilon end

            if epsilon_x_max > self.epsilon_3d_max:
                self.epsilon_3d_max = epsilon_x_max
            if epsilon_x_min < self.epsilon_3d_min:
                self.epsilon_3d_min = epsilon_x_min

        # clip to -e - e
        if self.epsilon is None:
            x_rgb = ori_img[:, :, :, :3] + x[:, :, :, :3] * alpha
        else:
            x_c = torch.clip((x[:, :, :, :3] * alpha), -self.epsilon, self.epsilon)
            x_rgb = ori_img[:, :, :, :3] + x_c

        zeros = torch.zeros_like(x_rgb).to(self.device)
        x_rgb = torch.where(ori_img_alpha > 0, x_rgb, zeros)
        x_a = ori_img[:, :, :, 3]

        x_rgba = torch.cat([x_rgb, x_a.unsqueeze(-1)], dim=-1)

        # clip to 0 - 255
        x_rgba = torch.clip(x_rgba,  min=0, max=255)

        cla_x = x_rgba.transpose(2, 3).transpose(1, 2)
        cla_ori_img = ori_img.transpose(2, 3).transpose(1, 2)

        # cla_x = torch.floor(cla_x)
        # cla_ori_img = torch.floor(cla_ori_img)

        # cla_x from rgba img trans to rgb img
        cla_x_size = cla_x.size()
        cla_x_alpha = cla_x[:, 3, :, :].unsqueeze(1).broadcast_to([cla_x_size[0], 3,
                                                                   cla_x_size[2],
                                                                   cla_x_size[3]])
        cla_x_rgb = cla_x[:, :3, :, :]

        tensor_255 = torch.ones_like(cla_x_rgb) * 255
        cla_x_3channel = torch.where(cla_x_alpha > 0, cla_x_rgb, tensor_255)

        # cla_ori_img from rgba img trans to rgb img
        cla_ori_img_size = cla_ori_img.size()
        cla_ori_img_alpha = cla_ori_img[:, 3, :, :].unsqueeze(1).broadcast_to([cla_ori_img_size[0], 3,
                                                                               cla_ori_img_size[2],
                                                                               cla_ori_img_size[3]])
        cla_ori_img_rgb = cla_ori_img[:, :3, :, :]

        tensor_255 = torch.ones_like(cla_ori_img_rgb) * 255
        cla_ori_img_3channel = torch.where(cla_ori_img_alpha > 0, cla_ori_img_rgb, tensor_255)

        if self.model_name == "my_model":
            pass
        elif self.model_name == "vit_b_16":
            cla_x_3channel = self.torch_resize_224(cla_x_3channel)
            cla_ori_img_3channel = self.torch_resize_224(cla_ori_img_3channel)
        else:
            cla_x_3channel = self.torch_resize_299(cla_x_3channel)
            cla_ori_img_3channel = self.torch_resize_299(cla_ori_img_3channel)

        cla = self.model(cla_x_3channel)
        ori_cla = self.model(cla_ori_img_3channel)

        return x, x_rgba, cla, ori_img, ori_cla

class create_gauss_w(nn.Module):
    def __init__(self, device, c):
        super(create_gauss_w, self).__init__()
        self.top_number = 8
        self.device = device
        self.c = c


    def forward(self, dist_and_index_list):

        dist = dist_and_index_list[:, 0, :, :, :].unsqueeze(1)
        x = dist_and_index_list[:, 1, :, :, :].unsqueeze(1)

        d = -(torch.square(dist / self.c) / 2)
        d = torch.exp(d)
        ds = torch.sum(d, dim=-1).unsqueeze(-1)
        ds = torch.broadcast_to(ds, (d.size()))
        dq = ds + 0.001
        zeros = torch.zeros_like(ds).to(self.device)

        w = torch.where(ds > 0, d / dq, zeros)
        # x = x * d

        i_w = torch.cat([w, x], 1)

        return i_w, dist


class gauss_get_r(nn.Module):
    def __init__(self, device, c, model, model_name):
        super(gauss_get_r, self).__init__()
        self.top_number = 8
        self.c = torch.nn.Parameter(torch.tensor([c]), requires_grad=False)
        self.device = device
        self.model = model
        self.model_name = model_name
        self.torch_resize_299 = Resize([299, 299])
        self.torch_resize_224 = Resize([224, 224])
        self.theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float, device=device)
        self.w = 299
        self.h = 299

        self.epsilon_3d_max = 0
        self.epsilon_3d_min = 0

        self.update_epsilon_3d = True

    def epsilon_3d_zero(self):
        self.epsilon_3d_max = 0
        self.epsilon_3d_min = 0

    def close_update_epsilon_3d(self):
        self.update_epsilon_3d = False

    def open_update_epsilon_3d(self):
        self.update_epsilon_3d = True

    def print_epsilon(self):
        print("epsilon_3d_min: ", self.epsilon_3d_min)
        print("epsilon_3d_max: ", self.epsilon_3d_max)

    def forward(self, spatial_rgb, dist_and_index_list):

        s = torch.reshape(spatial_rgb, (-1, 4))

        d = dist_and_index_list[:, 0, :, :, :]
        x = dist_and_index_list[:, 1, :, :, :]

        shape_temp = [x.size()[0], x.size()[1], x.size()[2], x.size()[3], 4]
        x = torch.reshape(x, (-1,)).unsqueeze(-1)
        x = torch.broadcast_to(x, (x.size()[0], 4)).type(torch.long)
        x = s.gather(dim=0, index=x)

        x = torch.reshape(x, shape_temp)
        d = -(torch.square(d / self.c) / 2)
        d = torch.exp(d).unsqueeze(-1)
        ds = torch.sum(d, dim=-2).unsqueeze(-1)
        ds = torch.broadcast_to(ds, (x.size()))
        dq = ds + 0.001
        zeros = torch.zeros_like(ds).to(self.device)

        x = torch.where(ds > 0, (x * d) / dq, zeros)
        # x = x * d

        x = torch.sum(x, dim=-2)

        alpha = x[:, :, :, 3].unsqueeze(-1) / 255

        if self.update_epsilon_3d:
            # test the epsilon when create attack img
            alpha_x_size = alpha.broadcast_to(x[:, :, :, :3].size())
            zreos = torch.zeros_like(x[:, :, :, :3])

            x_zeros = torch.where(alpha_x_size > 0, x[:, :, :, :3], zreos)

            epsilon_x_max = float(torch.max(x_zeros[:, :, :, :3] * alpha).data)
            epsilon_x_min = float(torch.min(x_zeros[:, :, :, :3] * alpha).data)
            # test the epsilon end

            if epsilon_x_max > self.epsilon_3d_max:
                self.epsilon_3d_max = epsilon_x_max
            if epsilon_x_min < self.epsilon_3d_min:
                self.epsilon_3d_min = epsilon_x_min

        return x


class gauss_get_img(nn.Module):
    def __init__(self, device, c, model, model_name):
        super(gauss_get_img, self).__init__()
        self.top_number = 8
        self.c = torch.nn.Parameter(torch.tensor([c]), requires_grad=False)
        self.device = device
        self.model = model
        self.model_name = model_name
        self.torch_resize_299 = Resize([299, 299])
        self.torch_resize_224 = Resize([224, 224])
        self.theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float, device=device)
        self.w = 299
        self.h = 299


    def forward(self, ori_img, r):
        ori_img = torch.tensor(ori_img, dtype=torch.float)

        alpha = r[:, :, :, 3].unsqueeze(-1) / 255
        ori_img_alpha = ori_img[:, :, :, 3].unsqueeze(-1)

        x_rgb = ori_img[:, :, :, :3] + r[:, :, :, :3] * alpha

        zeros = torch.zeros_like(x_rgb).to(self.device)
        x_rgb = torch.where(ori_img_alpha > 0, x_rgb, zeros)
        x_a = ori_img[:, :, :, 3]

        x_rgba = torch.cat([x_rgb, x_a.unsqueeze(-1)], dim=-1)

        cla_x = x_rgba.transpose(2, 3).transpose(1, 2)
        cla_ori_img = ori_img.transpose(2, 3).transpose(1, 2)

        # cla_x from rgba img trans to rgb img
        cla_x_size = cla_x.size()
        cla_x_alpha = cla_x[:, 3, :, :].unsqueeze(1).broadcast_to([cla_x_size[0], 3,
                                                                   cla_x_size[2],
                                                                   cla_x_size[3]])
        cla_x_rgb = cla_x[:, :3, :, :]
        tensor_255 = torch.ones_like(cla_x_rgb) * 255
        cla_x_3channel = torch.where(cla_x_alpha > 0, cla_x_rgb, tensor_255)

        # cla_ori_img from rgba img trans to rgb img
        cla_ori_img_size = cla_ori_img.size()
        cla_ori_img_alpha = cla_ori_img[:, 3, :, :].unsqueeze(1).broadcast_to([cla_ori_img_size[0], 3,
                                                                               cla_ori_img_size[2],
                                                                               cla_ori_img_size[3]])
        cla_ori_img_rgb = cla_ori_img[:, :3, :, :]
        tensor_255 = torch.ones_like(cla_ori_img_rgb) * 255
        cla_ori_img_3channel = torch.where(cla_ori_img_alpha > 0, cla_ori_img_rgb, tensor_255)

        if self.model_name == "inception" or self.model_name == "vgg16" or self.model_name == "alexnet":
            cla_x_3channel = self.torch_resize_299(cla_x_3channel)
            cla_ori_img_3channel = self.torch_resize_299(cla_ori_img_3channel)
        elif self.model_name == "vit_b_16":
            cla_x_3channel = self.torch_resize_224(cla_x_3channel)
            cla_ori_img_3channel = self.torch_resize_224(cla_ori_img_3channel)
        else:
            cla_x_3channel = self.torch_resize_299(cla_x_3channel)
            cla_ori_img_3channel = self.torch_resize_299(cla_ori_img_3channel)

        cla = self.model(cla_x_3channel)
        ori_cla = self.model(cla_ori_img_3channel)

        return r, x_rgba, cla, ori_img, ori_cla


class universal_2D_net(nn.Module):
    def __init__(self, device, c, model, model_name):
        super(universal_2D_net, self).__init__()
        self.top_number = 8
        self.c = torch.nn.Parameter(torch.tensor([c]), requires_grad=False)
        self.device = device
        self.model = model
        self.model_name = model_name
        self.torch_resize_299 = Resize([299, 299])
        self.torch_resize_224 = Resize([224, 224])
        self.theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float, device=device)


    def forward(self,  r, ori_img):
        ori_img = torch.tensor(ori_img, dtype=torch.float)

        if r.size() != ori_img.size():
            br = r.unsqueeze(0)
            br = br.broadcast_to(ori_img.size())

            x_rgb = ori_img + br
        else:
            x_rgb = ori_img + r

        x_rgb = torch.clip(x_rgb, min=0, max=255)

        cla_x = x_rgb.transpose(2, 3).transpose(1, 2)
        cla_ori_img = ori_img.transpose(2, 3).transpose(1, 2)

        if self.model_name == "inception" or self.model_name == "vgg16" or self.model_name == "alexnet":
            cla_x = self.torch_resize_299(cla_x)
            cla_ori_img = self.torch_resize_299(cla_ori_img)
        elif self.model_name == "vit_b_16":
            cla_x = self.torch_resize_224(cla_x)
            cla_ori_img = self.torch_resize_224(cla_ori_img)
        else:
            cla_x = self.torch_resize_299(cla_x)
            cla_ori_img = self.torch_resize_299(cla_ori_img)

        cla = self.model(cla_x)
        ori_cla = self.model(cla_ori_img)

        return r, x_rgb, cla, ori_img, ori_cla
