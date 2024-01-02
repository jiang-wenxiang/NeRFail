import os.path
import shutil

import configargparse


def get_test_dir_after_nerf_attack(scene_name = "lego", model_name = "vgg16", now_class_idx = 10, attack_epochs = 100,
                       print_a = 2, print_e = 32, target_class_idx = "n", base_mask_image_number = 3, m1=0, m2=0,
                       method_name = "NeRFail", step=0, something_need_log: dict = None):

    if step == 0:
        step_name = "attack"
    elif step == 1:
        step_name = "nerf"
    elif step == 2:
        step_name = "defense"
    elif step == 3:
        step_name = "nerf_defense"

    if method_name is None:
        pass
    else:
        if method_name == "IGSM":
            method_name = "NeRFail_S"
        elif method_name == "Universal":
            method_name = "NeRFail"

    test_dir = None

    if method_name is None:
        test_dir = None

    if method_name == "NeRFail":
        test_dir = './output/'+model_name+'/'+step_name+'/' + scene_name + \
                   '/'+method_name+'_'+str(base_mask_image_number)+'P_'+str(attack_epochs) + \
                   '_to_'+str(target_class_idx)+'_e_'+str(print_e)+'_m_'+str(m1)+'_'+str(m2)

    elif method_name == "NeRFail_S":
        test_dir = './output/'+model_name+'/'+step_name+'/' + scene_name + \
                   '/'+method_name+'_'+str(base_mask_image_number)+'P_'+str(attack_epochs)+'_to_' + \
                   str(target_class_idx)+'_e_'+str(print_e)+'_a_'+str(print_a)

    elif method_name == "IGSM_2D":
        test_dir = './output/'+model_name+'/'+step_name+'/' + scene_name + \
                   '/'+method_name+'_'+str(attack_epochs)+'_to_'+str(target_class_idx) + \
                   '_e_'+str(print_e)+'_a_'+str(print_a)

    elif method_name == "No_attack" and step == 0:
        test_dir = None

    elif method_name == "No_attack":
        test_dir = './output/'+model_name+'/'+step_name+'/' + scene_name + '/no_attack'

    elif method_name == "Universal_2D":
        test_dir = './output/'+model_name+'/'+step_name+'/' + scene_name + \
                   '/'+'Universal_2D_'+str(attack_epochs)+'_to_'+str(target_class_idx)+'_e_' + \
                   str(print_e)+'_m_'+str(m1)+'_'+str(m2)

    return test_dir


def move_files(from_dir, to_dir):
    if to_dir is not None:
        if os.path.isdir(from_dir):
            if not os.path.exists(to_dir):
                os.makedirs(to_dir)
            files = os.listdir(from_dir)
            for f in files:
                if os.path.isdir(from_dir + "/" + f):
                    move_files(from_dir + "/" + f, to_dir + "/" + f)
                else:
                    shutil.move(from_dir + "/" + f, to_dir + "/" + f)
        else:
            shutil.move(from_dir, to_dir)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--from_dir', type=str)

    parser.add_argument('--e', type=int, default=32)
    parser.add_argument('--a', type=int, default=2)
    parser.add_argument('--m1', type=int, default=8)
    parser.add_argument('--m2', type=int, default=100)
    parser.add_argument('--base_mask_image_number', type=int, default=3)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--method_name', type=str, default=None)
    parser.add_argument('--label', type=str, default='lego')
    parser.add_argument('--model_name', type=str, default='inception')
    parser.add_argument('--target_class_idx', default='n')

    args = parser.parse_args()

    copy_from_list = ['renderonly_train_199999', 'renderonly_test_199999', 'renderonly_val_199999',
                      '200000.tar', 'args.txt', 'config.txt']
    copy_to_list = ['train', 'test', 'val', '200000.tar', 'args.txt', 'config.txt']

    copy_to_dir_path = get_test_dir_after_nerf_attack(print_e=args.e, print_a=args.a, model_name=args.model_name,
                       base_mask_image_number=args.base_mask_image_number, m1=args.m1,
                       m2=args.m2, step=args.step,
                       method_name=args.method_name, target_class_idx=args.target_class_idx,
                       scene_name=args.label)

    for i in range(len(copy_from_list)):
        move_files(args.from_dir + '/' + copy_from_list[i], copy_to_dir_path + '/' + copy_to_list[i])
