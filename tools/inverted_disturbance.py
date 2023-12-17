import cv2
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# disturbance_file_root = "../output/inception/attack/lego/IGSM_3P_100_to_n_e_32_a_2/attack_masks/base_masks/"
# disturbance_file_name_list = ["50.png", "75.png", "125.png"]

disturbance_file_root = "../output/inception/attack/lego/IGSM_3P_100_to_n_e_32_a_2/attack_masks/test/"
disturbance_file_name_list = ["r_23.png"]

for disturbance_file_name in disturbance_file_name_list:

    r_img_numpy = cv2.imread(disturbance_file_root + disturbance_file_name, cv2.IMREAD_UNCHANGED)

    r_img = torch.tensor(r_img_numpy).to(device)
    r_img_alpha = r_img[:, :, 3].unsqueeze(-1)
    r_img_rgb = r_img[:, :, :3]

    inv_r_img_rgb = 255 - r_img_rgb
    inv_r_img_alpha = (r_img_rgb[:, :, 0] + r_img_rgb[:, :, 1] + r_img_rgb[:, :, 2]).unsqueeze(-1)
    inv_r_img_alpha = r_img_alpha * inv_r_img_alpha

    inv_r_img_alpha_0 = torch.zeros_like(inv_r_img_alpha) * 255
    inv_r_img_alpha_255 = torch.ones_like(inv_r_img_alpha) * 255
    inv_r_img_alpha = torch.where(inv_r_img_alpha > 0, inv_r_img_alpha_255, inv_r_img_alpha_0)

    inv_r_img = torch.cat([inv_r_img_rgb, inv_r_img_alpha], -1)

    cv2.imwrite(disturbance_file_root + "inv_" + disturbance_file_name, inv_r_img.cpu().detach().numpy())

