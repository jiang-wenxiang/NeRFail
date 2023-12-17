import os
import random
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Resize
from torchvision import transforms as tt


def find_classes(directory: Path) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class MySimpleDataset(Dataset):
    def __init__(self, root: Path, set_name: str, class_names_path: Path = None, test_dir_change_dict: dict = None, ori_img_from: dict=None,
                 _3_channels=False, resize_frame=False, resize_frame_size=Resize([800, 800]), device='cpu', load_later=False):
        self.classes, self.class_to_idx = find_classes(root)
        if class_names_path is not None:
            with open(class_names_path, 'w') as f:
                for i_classes in self.classes:
                    f.write(str(self.class_to_idx[i_classes]) + " -- "+i_classes+"\n")
                f.close()

        self.root = root
        self.set_name = set_name
        self.load_later = load_later
        self._3_channels = _3_channels

        self.resize_frame = resize_frame
        self.resize_frame_size = resize_frame_size

        self.img_set = {}
        self.ori_img_set = {}
        self.label_set = {}

        self.device = device

        self.now_index = 0
        self.test_dir_change_dict = test_dir_change_dict

        self.ori_img_from = ori_img_from

        for class_name in self.classes:
            self.read_data_from_one_class(class_name, set_name)

        self.length = len(self.img_set)

    def read_data_from_one_class(self, class_name: str, set_name: str):
        class_path = os.path.join(self.root, class_name, set_name)
        ori_path = None
        class_idx = self.class_to_idx[class_name]
        if self.test_dir_change_dict is not None:
            if class_idx in self.test_dir_change_dict.keys():
                class_path = self.test_dir_change_dict[class_idx]

        if self.ori_img_from is not None:
            if class_idx in self.ori_img_from.keys():
                ori_path = self.ori_img_from[class_idx]

        img_files = os.listdir(class_path)

        start_idx = self.now_index

        for img_file in img_files:
            if (len(img_file.split("_")) < 3) and ("ori" not in img_file):
                if len(img_file.split("_")) == 2:
                    img_index = int(img_file.split("_")[1].replace(".png", "").replace(".jpg", "").replace(".jpeg", ""))
                else:
                    img_index = int(img_file.split("_")[0].replace(".png", "").replace(".jpg", "").replace(".jpeg", ""))
                img_idx = start_idx + img_index
                img_path = os.path.join(class_path, img_file)

                if self.load_later:
                    self.img_set[img_idx] = img_path
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    tensor_img = torch.tensor(img, dtype=torch.float).to(self.device)

                    tensor_img = tensor_img.transpose(1, 2).transpose(0, 1)

                    if self._3_channels:
                        channels = tensor_img.size()[0]
                        if channels == 4:
                            tensor_img_size = tensor_img.size()
                            tensor_img_alpha = tensor_img[3, :, :].unsqueeze(0).broadcast_to([3, tensor_img_size[1], tensor_img_size[2]])
                            tensor_img_rgb = tensor_img[:3, :, :]
                            tensor_img_white = torch.ones_like(tensor_img_rgb) * 255
                            tensor_img = torch.where(tensor_img_alpha > 0, tensor_img_rgb, tensor_img_white)

                    if self.resize_frame:
                        tensor_img = self.resize_frame_size(tensor_img)

                    self.img_set[img_idx] = tensor_img.detach().clone()

                self.label_set[img_idx] = class_idx
                self.now_index += 1

                if ori_path is not None:
                    ori_img_file = "r_" + str(img_index) + ".png"
                    ori_img_path = os.path.join(ori_path, ori_img_file)
                    if self.load_later:
                        self.ori_img_set[img_idx] = ori_img_path
                    else:
                        ori_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
                        tensor_ori_img = torch.tensor(ori_img, dtype=torch.float).to(self.device)

                        tensor_ori_img = tensor_ori_img.transpose(1, 2).transpose(0, 1)

                        if self._3_channels:
                            channels = tensor_ori_img.size()[0]
                            if channels == 4:
                                tensor_ori_img_size = tensor_ori_img.size()
                                tensor_ori_img_alpha = tensor_ori_img[3, :, :].unsqueeze(0).broadcast_to(
                                    [3, tensor_ori_img_size[1], tensor_ori_img_size[2]])
                                tensor_ori_img_rgb = tensor_ori_img[:3, :, :]
                                tensor_ori_img_white = torch.ones_like(tensor_ori_img_rgb) * 255
                                tensor_ori_img = torch.where(tensor_ori_img_alpha > 0, tensor_ori_img_rgb, tensor_ori_img_white)

                        if self.resize_frame:
                            tensor_ori_img = self.resize_frame_size(tensor_ori_img)

                        self.ori_img_set[img_idx] = tensor_ori_img.detach().clone()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.ori_img_from is None:
            if self.load_later:
                img_path = self.img_set[idx]
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                tensor_img = torch.tensor(img, dtype=torch.float).to(self.device)

                tensor_img = tensor_img.transpose(1, 2).transpose(0, 1)

                if self._3_channels:
                    tensor_img_size = tensor_img.size()
                    if tensor_img_size[0] == 4:
                        tensor_img_alpha = tensor_img[3, :, :].unsqueeze(0).broadcast_to([3, tensor_img_size[1], tensor_img_size[2]])
                        tensor_img_rgb = tensor_img[:3, :, :]
                        tensor_img_white = torch.ones_like(tensor_img_rgb) * 255
                        tensor_img = torch.where(tensor_img_alpha > 0, tensor_img_rgb, tensor_img_white)

                if self.resize_frame:
                    tensor_img = self.resize_frame_size(tensor_img)

                return tensor_img.detach().clone(), self.label_set[idx]
            else:
                return self.img_set[idx], self.label_set[idx]

        if idx in self.ori_img_set.keys():
            return self.img_set[idx], self.label_set[idx], self.ori_img_set[idx]

        return self.img_set[idx], self.label_set[idx], self.img_set[idx]


class gauss_dataset(Dataset):
    def __init__(self, all_index_and_dist_name_list, all_img_name_list, all_img_save_to_name_list, all_img_mask_save_to_name_list, device):
        self.length = len(all_img_name_list)
        self.all_index_and_dist_name_list = all_index_and_dist_name_list
        self.all_img_name_list = all_img_name_list
        self.all_img_save_to_name_list = all_img_save_to_name_list
        self.all_img_mask_save_to_name_list = all_img_mask_save_to_name_list
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ori_img = torch.tensor(cv2.imread(self.all_img_name_list[index], cv2.IMREAD_UNCHANGED)).to(self.device)
        img_index_and_dist = torch.load(self.all_index_and_dist_name_list[index], map_location=self.device)
        img_save_to_name = self.all_img_save_to_name_list[index]
        img_mask_save_to_name = self.all_img_mask_save_to_name_list[index]
        return index, ori_img, img_index_and_dist, img_save_to_name, img_mask_save_to_name


class gauss_dataset_rand_select(Dataset):
    def __init__(self, all_index_and_dist_name_list, all_img_name_list, all_img_save_to_name_list, all_img_mask_save_to_name_list, device, select_rate=0.2):
        rand_seed = 1003
        random.seed(rand_seed)

        ori_length = len(all_img_name_list)
        ori_index_list = range(ori_length)
        self.length = int(ori_length * select_rate)

        rand_select_index = random.sample(ori_index_list, self.length)
        self.all_index_and_dist_name_list = [all_index_and_dist_name_list[i] for i in rand_select_index]
        self.all_img_name_list = [all_img_name_list[i] for i in rand_select_index]
        self.all_img_save_to_name_list = [all_img_save_to_name_list[i] for i in rand_select_index]
        self.all_img_mask_save_to_name_list = [all_img_mask_save_to_name_list[i] for i in rand_select_index]
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ori_img = torch.tensor(cv2.imread(self.all_img_name_list[index], cv2.IMREAD_UNCHANGED)).to(self.device)
        img_index_and_dist = torch.load(self.all_index_and_dist_name_list[index], map_location=self.device)
        img_save_to_name = self.all_img_save_to_name_list[index]
        img_mask_save_to_name = self.all_img_mask_save_to_name_list[index]
        return index, ori_img, img_index_and_dist, img_save_to_name, img_mask_save_to_name



class universal_2D_dataset(Dataset):
    def __init__(self, all_img_name_list, all_img_save_to_name_list, all_img_mask_save_to_name_list, device):
        self.length = len(all_img_name_list)
        self.all_img_name_list = all_img_name_list
        self.all_img_save_to_name_list = all_img_save_to_name_list
        self.all_img_mask_save_to_name_list = all_img_mask_save_to_name_list
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ori_img = torch.tensor(cv2.imread(self.all_img_name_list[index], cv2.IMREAD_UNCHANGED)).to(self.device)

        # cla_ori_img from rgba img trans to rgb img
        cla_ori_img_size = ori_img.size()
        cla_ori_img_alpha = ori_img[:, :, 3].unsqueeze(-1).broadcast_to([cla_ori_img_size[0],
                                                                         cla_ori_img_size[1], 3])
        cla_ori_img_rgb = ori_img[:, :, :3]
        tensor_255 = torch.ones_like(cla_ori_img_rgb) * 255
        ori_img_3channel = torch.where(cla_ori_img_alpha > 0, cla_ori_img_rgb, tensor_255)

        img_save_to_name = self.all_img_save_to_name_list[index]
        img_mask_save_to_name = self.all_img_mask_save_to_name_list[index]
        return index, ori_img_3channel, img_save_to_name, img_mask_save_to_name


class universal_2D_dataset_rand_select(Dataset):
    def __init__(self, all_img_name_list, all_img_save_to_name_list, all_img_mask_save_to_name_list, device, select_rate=0.2):
        rand_seed = 1003
        random.seed(rand_seed)

        ori_length = len(all_img_name_list)
        ori_index_list = range(ori_length)
        self.length = int(ori_length * select_rate)

        rand_select_index = random.sample(ori_index_list, self.length)
        self.all_img_name_list = [all_img_name_list[i] for i in rand_select_index]
        self.all_img_save_to_name_list = [all_img_save_to_name_list[i] for i in rand_select_index]
        self.all_img_mask_save_to_name_list = [all_img_mask_save_to_name_list[i] for i in rand_select_index]
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ori_img = torch.tensor(cv2.imread(self.all_img_name_list[index], cv2.IMREAD_UNCHANGED)).to(self.device)

        # cla_ori_img from rgba img trans to rgb img
        cla_ori_img_size = ori_img.size()
        cla_ori_img_alpha = ori_img[:, :, 3].unsqueeze(-1).broadcast_to([cla_ori_img_size[0],
                                                                         cla_ori_img_size[1], 3])
        cla_ori_img_rgb = ori_img[:, :, :3]
        tensor_255 = torch.ones_like(cla_ori_img_rgb) * 255
        ori_img_3channel = torch.where(cla_ori_img_alpha > 0, cla_ori_img_rgb, tensor_255)

        img_save_to_name = self.all_img_save_to_name_list[index]
        img_mask_save_to_name = self.all_img_mask_save_to_name_list[index]
        return index, ori_img_3channel, img_save_to_name, img_mask_save_to_name

class gauss_weight_dataset(Dataset):
    def __init__(self, all_index_and_dist_name_list, all_gauss_weight_name_list, device):
        self.length = len(all_index_and_dist_name_list)
        self.all_index_and_dist_name_list = all_index_and_dist_name_list
        self.all_gauss_weight_name_list = all_gauss_weight_name_list
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_index_and_dist = torch.load(self.all_index_and_dist_name_list[index], map_location=self.device)
        gauss_weight_name = self.all_gauss_weight_name_list[index]
        return img_index_and_dist, gauss_weight_name