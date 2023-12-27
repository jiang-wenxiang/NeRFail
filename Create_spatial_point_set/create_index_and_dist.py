import os
import shutil
import time
from pathlib import Path

import configargparse
import cv2
import numpy as np
import torch
import tqdm
from scipy.spatial import distance


def get_coord_min_and_max(tensor1):
    set_len, height_coord, width_coord, coord = tensor1.size()
    tensor1_reshape = torch.reshape(tensor1, [-1, coord])
    tensor1_coord_min = torch.min(tensor1_reshape, dim=0)
    tensor1_coord_max = torch.max(tensor1_reshape, dim=0)
    return tensor1_coord_min[0], tensor1_coord_max[0]


def create_index_and_dist(label, epochs, mask_list):
    # basic_data_dir = "./data/nerf_synthetic/"+label+"/"
    # log_data_dir = "./logs/blender_paper_"+label+"/"

    basic_data_dir = "../data/nerf_synthetic/" + label + "/"
    log_data_dir = "./logs/blender_paper_" + label + "/"

    coord_save_npy_dir_name = "index_and_dist/"
    device = "cuda:0"

    test_coord_npy_dir = log_data_dir + "renderonly_test_"+str(epochs)+"/"
    train_coord_npy_dir = log_data_dir + "renderonly_train_"+str(epochs)+"/"
    val_coord_npy_dir = log_data_dir + "renderonly_val_"+str(epochs)+"/"

    coord_save_npy_dir = log_data_dir + coord_save_npy_dir_name

    test_save_coord_npy_dir = coord_save_npy_dir + "test/"
    train_save_coord_npy_dir = coord_save_npy_dir + "train/"
    val_save_coord_npy_dir = coord_save_npy_dir + "val/"

    train_img_num = 100
    val_img_num = 100
    test_img_num = 200
    split_parts = 1600
    top_number = 8

    zero_s = ["000", "00", "0", ""]
    number_to_npy_filename_4 = lambda x: zero_s[len(str(x))] + str(x) + ".npy"

    tensor_train_coord = None
    tensor_val_coord = None
    tensor_test_coord = None

    # change_target_img_index_list = [0, 15, 25]
    change_target_img_index_list = mask_list
    test_index_coord_file_name_list = [os.path.join(test_coord_npy_dir, zero_s[len(str(i))] + str(i) + ".npy") for i in change_target_img_index_list]

    test_index_change_list = [torch.from_numpy(np.load(file)).to(device) for file in test_index_coord_file_name_list]

    test_index_change_tensor = torch.reshape(torch.stack(test_index_change_list), (-1, 3))

    for train_img_i in range(train_img_num):
        train_npy_file_name = number_to_npy_filename_4(train_img_i)
        train_npy_path = train_coord_npy_dir + train_npy_file_name
        train_coord_np = np.load(train_npy_path)

        if tensor_train_coord is None:
            tensor_train_coord = torch.from_numpy(train_coord_np)
            tensor_train_coord = tensor_train_coord.to(device)
            tensor_train_coord = tensor_train_coord.unsqueeze(0)
        else:
            tensor_train_coord_temp = torch.from_numpy(train_coord_np)
            tensor_train_coord_temp = tensor_train_coord_temp.to(device)
            tensor_train_coord_temp = tensor_train_coord_temp.unsqueeze(0)
            tensor_train_coord = torch.cat([tensor_train_coord, tensor_train_coord_temp], dim=0)

    for val_img_i in range(val_img_num):
        val_npy_file_name = number_to_npy_filename_4(val_img_i)
        val_npy_path = val_coord_npy_dir + val_npy_file_name
        val_coord_np = np.load(val_npy_path)

        if tensor_val_coord is None:
            tensor_val_coord = torch.from_numpy(val_coord_np)
            tensor_val_coord = tensor_val_coord.to(device)
            tensor_val_coord = tensor_val_coord.unsqueeze(0)
        else:
            tensor_val_coord_temp = torch.from_numpy(val_coord_np)
            tensor_val_coord_temp = tensor_val_coord_temp.to(device)
            tensor_val_coord_temp = tensor_val_coord_temp.unsqueeze(0)
            tensor_val_coord = torch.cat([tensor_val_coord, tensor_val_coord_temp], dim=0)

    for test_img_i in range(test_img_num):
        test_npy_file_name = number_to_npy_filename_4(test_img_i)
        test_npy_path = test_coord_npy_dir + test_npy_file_name
        test_coord_np = np.load(test_npy_path)

        if tensor_test_coord is None:
            tensor_test_coord = torch.from_numpy(test_coord_np)
            tensor_test_coord = tensor_test_coord.to(device)
            tensor_test_coord = tensor_test_coord.unsqueeze(0)
        else:
            tensor_test_coord_temp = torch.from_numpy(test_coord_np)
            tensor_test_coord_temp = tensor_test_coord_temp.to(device)
            tensor_test_coord_temp = tensor_test_coord_temp.unsqueeze(0)
            tensor_test_coord = torch.cat([tensor_test_coord, tensor_test_coord_temp], dim=0)

    # for tab in ["train"]:
    # for tab in ["test", "train"]:
    for tab in ["test", "train", "val"]:
        if tab == "train":
            set_len, height_coord, width_coord, coords = tensor_train_coord.size()
        elif tab == "val":
            set_len, height_coord, width_coord, coords = tensor_val_coord.size()
        else:
            set_len, height_coord, width_coord, coords = tensor_test_coord.size()

        for img_i in range(set_len):
            if tab == "train":
                tensor_coord = tensor_train_coord[img_i, :, :, :]
            elif tab == "val":
                tensor_coord = tensor_val_coord[img_i, :, :, :]
            else:
                tensor_coord = tensor_test_coord[img_i, :, :, :]

            test_index_change_tensor_list = test_index_change_tensor.chunk(split_parts, dim=0)

            before_length = 0

            near_eight_idx_tensor_list = []
            near_eight_dist_tensor_list = []
            for chunk_test_index_change_tensor in tqdm.tqdm(test_index_change_tensor_list):
                tensor_tab_coord_dist = torch.cdist(tensor_coord, chunk_test_index_change_tensor)
                values, idx = torch.sort(tensor_tab_coord_dist, dim=-1)
                near_eight_idx_tensor_list.append(idx[:, :, :top_number] + before_length)
                near_eight_dist_tensor_list.append(values[:, :, :top_number])
                before_length += chunk_test_index_change_tensor.size()[0]

                near_eight_idx_tensor = torch.cat(near_eight_idx_tensor_list, dim=-1)
                near_eight_dist_tensor = torch.cat(near_eight_dist_tensor_list, dim=-1)

                values, idx = torch.sort(near_eight_dist_tensor, dim=-1)

                near_eight_idx_tensor_list = [near_eight_idx_tensor.gather(index=idx[:, :, :top_number], dim=-1)]
                near_eight_dist_tensor_list = [values[:, :, :top_number]]

            assert len(near_eight_idx_tensor_list) == 1 and len(near_eight_dist_tensor_list) == 1, "ERROR List Length!"
            coord_near_eight_global_index = [
                near_eight_dist_tensor_list[0].unsqueeze(0),
                near_eight_idx_tensor_list[0].unsqueeze(0)
            ]

            coord_near_eight_global = torch.cat(coord_near_eight_global_index, dim=0)

            if tab == "train":
                os.makedirs(train_save_coord_npy_dir, exist_ok=True)
                torch.save(coord_near_eight_global, train_save_coord_npy_dir + str(img_i) + ".pth")
            elif tab == "val":
                os.makedirs(val_save_coord_npy_dir, exist_ok=True)
                torch.save(coord_near_eight_global, val_save_coord_npy_dir + str(img_i) + ".pth")
            else:
                os.makedirs(test_save_coord_npy_dir, exist_ok=True)
                torch.save(coord_near_eight_global, test_save_coord_npy_dir + str(img_i) + ".pth")

            if tab == "train":
                print(tab + " [" + str(img_i + 1) + "/" + str(train_img_num) + "]")
            elif tab == "val":
                print(tab + " [" + str(img_i + 1) + "/" + str(val_img_num) + "]")
            else:
                print(tab + " [" + str(img_i + 1) + "/" + str(test_img_num) + "]")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--label', type=str, default="lego", help='object name')
    parser.add_argument('--epochs', type=int, default=199999, help='the number of nerf model retrain')
    mask_list = [50, 75, 125]
    args = parser.parse_args()
    create_index_and_dist(label=args.label, epochs=args.epochs, mask_list=mask_list)
