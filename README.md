# NeRFail: Neural Radiance Fields-based multiview adversarial attack

Wenxiang Jiang <sup> \* </sup>, Hanwei Zhang <sup> \* \† </sup>, Xi Wang, Zhongwen Guo <sup> \† </sup>, Hao Wang<br>
(<sup> \* </sup> Equal Contribution.)<br>
(<sup> \† </sup> Corresponding Author.)

Abstract: *Adversarial attacks, _i.e._ generating adversarial perturbations with a small magnitude to deceive deep neural networks, are important for investigating and improving model trustworthiness. Traditionally, the topic was scoped within 2D images or a full dataset without considering multiview. Benefiting from Neural Radiance Fields (NeRF), one can easily reconstruct a 3D scene with a Multi-Layer Perceptron (MLP) from given 2D views and synthesize photo-realistic renderings from novel vantages. This opens up a door to discussing the possibility of undertaking to attack multiview NeRF network with downstream tasks taking input information from different rendering angles, which we denote _**Ne**ural **R**adiance **F**ields-based multiview **a**dversar**i**a**l** Attack (NeRFail)_. The goal is, given one scene and a subset of views, to deceive the recognition tasks of different unknown view angles as well as given views. To do so, we propose a transformation to mapping from pixels to 3D points so that our attack generates multiview adversarial perturbations by attacking a subset of images with different views against the downstream classification model to images rendered by NeRF from other views. Experiments show that our multiview adversarial perturbations successfully obfuscate the downstream classification at both known and unknown views. Notably, when retraining another NeRF on the training data with generated multiview perturbations, we show that the perturbation can be inherited and reproduced from the attacking agnostic model.*

**This paper has been accepted for the AAAI 2024 conference and is expected to be able to be queried after the conference.**

![NeRFail attack image](assets/NeRFail.png)



## Funding and Acknowledgments
This work received support from the National Key Research and Development Program of China (No. 2020YFB1707701) and the National Natural Science Foundation of China (Grant No. 61827810). <br>
This work also received support from DFG under grant No. 389792660 as part of TRR 248 <sup> \* </sup> and VolkswagenStiftung as part of Grant AZ 98514 <sup> \† </sup>. <br>
Wenxiang Jiang was funded by the China Scholarship Council (CSC).

(<sup> \* </sup> CPEC:https://perspicuous-computing.science)<br>
(<sup> \† </sup> EIS:https://explainable-intelligent.systems)

## Overview

The codebase has 4 main components that correspond to the 4 steps of the NeRF-based attack:

- A pytorch implementation of the original NeRF model. Thanks to [YenChen, Lin's implementation](https://github.com/yenchenlin/nerf-pytorch), this part we changed very little on his version. For pre-training of the scene and reconstruction of the scene after attack.
- The scripts is used to create an index and weight file for each image base pre-training nerf scene. This creation file includes the 8 spatial nearest point index and Gaussian weights from the selected image points set.
- Attack scripts. NeRFail and NeRFail-S proposed attack methods in the paper, and two baseline comparison methods, IGSM and UAP.
- Classifier model. Weights for the model, training and testing scripts.

They have been tested on Ubuntu 20.04.

## Environment
The computer we used to complete our experiments was an Ubuntu 20.04 system with 6 NVIDIA GeForce RTX 3090 graphics cards.<br>
Some of the important packages and their versions are listed below.

| package     | version      |
|:------------|:-------------|
| CUDA        | 12.0         |
| python      | 3.8          |
| pytorch     | 2.0.1+cu118  |
| torchvision | 0.15.2+cu118 |

In addition, this project requires NeRF training and rendering, and we need to install the packages mentioned in [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).<br>
We may use some other packages, if you are prompted at runtime that a package cannot be found, please try to use pip / pip3 to install them, we think it will be better for running successfully.<br>
For further reference, we may also be using packages that are already present in the [anaconda](https://www.anaconda.com/) or [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/) environments, and following the steps to [install Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) may help solve the problem.

## Running

### Training NeRF model
The first step is to train the NeRF model to reconstruct the scene that you want to attack.

To the *NeRFail/Create_spatial_point_set/nerf_pytorch/* directory.

```shell
cd Create_spatial_point_set
cd nerf_pytorch
```

Reconstruct a scene by NeRF training using the scene config files (the example given is the lego scene reconstruct).

```shell
python run_nerf.py --config ../configs/lego.txt
```

### Create the Gaussian weight Files
To the parent directory, *i.e.* the *NeRFail/Create_spatial_point_set/* directory.

```shell
cd ..
```

Output the spatial coordinate position of each pixel using the trained NeRF model.

```shell
python nerf_to_coord.py --config ./configs/lego_coord.txt 
```

From the set of 3D points created from the selected *p* images (default *p=3*), find the 8 nearest points and spatial distances for every pixel in every image.

```shell
python create_index_and_dist.py --label lego --epochs 199999
```

Switch to the tools directory (*NeRFail/tools/*).

```shell
cd ../tools
```

Transform the distances of the neighbouring 8 points into Gaussian weights.

```shell
python dist_to_weight.py --label lego
```

### Training and testing classification model
To the parent directory, *i.e.* the *NeRFail/*.

```shell
cd ..
```

Training for classification model using original train image set.

```shell
python model_train.py --model_name inception --num_classes 8 --data_dir ./data/nerf_synthetic
```

Testing for classification model using original test image set.

```shell
python model_test.py --model_name inception
```

### Attack method

The NeRFail, NeRFail_S, IGSM and UAP attack methods can be used when there are *index_and_weight* folder in the *NeRFail/Create_spatial_point_set/logs/blender_paper_**lego*** directory and ***inception**_8_best.pth* files in the *NeRFail/model/weights* directory.<br>
In these running paths, the ***lego*** can be replaced with other scene objects such as mic, chair, drums, etc (the above steps need to be completed again for the new scene). Also the ***inception*** can be replaced with other classification models such as alexnet, vgg16, resnet50, etc.<br>
If using scene datasets and classifiers not mentioned in our paper, you need to modify the program by yourself.
We may improve this in the future to make it more general.
#### parameters
The following are some of the optional parameters.<br>
**--e:** Limit the maximum change value per pixel (L<sub>&infin;</sub> norm) in the range 0-255.<br>
**--label:** Name of the scene object to attack.<br>
**--model_name:** Classification model name.<br>
**--attack_target_label_int:** Index of the targeted object to attack. If there are untargeted attack, this index shold be the index of the ground truth object.<br>
In general, the scene object corresponds to the index value as follows.<br>
0 -- chair<br>
1 -- drums<br>
2 -- ficus<br>
3 -- hotdog<br>
4 -- lego<br>
5 -- materials<br>
6 -- mic<br>
7 -- ship<br>
**--m1:** In UAP or NeRFail attacks, this parameter is used to improve the confidence of misclassification during the attack, but it may make the attack progress very slowly.<br>
**--m2:** In UAP or NeRFail attack methods, this parameter is used to speed up, but may lead to attack results falling into a local optimum solution that reduces the attack's success rate or the confidence of misclassification.<br>
The best way is to keep the balance of m1 and m2 parameters. After the limited number of ablation experiments (see the paper for details), we choose the default values that m1=8 and m2=100. We cannot rule out that for these two parameters, there may be more effective value.<br>

#### NeRFail
```shell
python attack_NeRFail.py --label lego --model_name inception --e 32 --attack_target_label_int 4 --m1 8 --m2 100
```
#### NeRFail-S
```shell
python attack_NeRFail_S.py --label lego --model_name inception --e 32 --attack_target_label_int 4
```
#### UAP (NeRFail baseline)
```shell
python attack_UAP_2D.py --label lego --model_name inception --e 32 --attack_target_label_int 4 --m1 8 --m2 100
```
#### IGSM (NeRFail-S baseline)
```shell
python attack_IGSM_2D.py --label lego --model_name inception --e 32 --attack_target_label_int 4
```

### Retraining NeRF using the attacked dataset

***Note:*** Must delete the *.tar* files in the *logs/blender_paper_lego* folder before using the attacked dataset to train NeRF, or the NeRF model will load the previously trained weights as the initial weight values.<br>
Switch to the NeRF model path, to the *NeRFail/Create_spatial_point_set/nerf_pytorch/* directory.

```shell
cd Create_spatial_point_set
cd nerf_pytorch
```

The NeRF is trained using the attacked training set, where the config file is the same as in normal training, and the location where the replacement training set is located is specified via the **--train_dir** parameter.

Example 1: Training NeRF using FeRFail-S attacked result.

```shell
python run_nerf.py --config ../configs/lego.txt --train_dir ../../output/inception/attack/lego/NeRFail_S_3P_100_to_n_e_32_a_2/train
```

Example 2: Training NeRF using FeRFail attacked result.

```shell
python run_nerf.py --config ../configs/lego.txt --train_dir ../../output/inception/attack/lego/NeRFail_3P_100_to_n_e_32_m_8_100/train
```

***Note:*** The Nerf training process has a certain degree of randomness. When the PSNR remains low (PSNR<15), it may be due to getting stuck in a local optimal solution. In this case, you could try terminating the training and restarting.<br>
To the parent directory, *i.e.* the *NeRFail/Create_spatial_point_set/*.
```shell
cd ..
```

Rendering images on the train, test and validation set viewpoint of the NeRF model trained on the attacked training set.

```shell
python nerf_render_only.py --config ./configs/lego_coord.txt
```

### Testing the NeRF scene training by the attack dataset 

To the parent directory, *i.e.* the *NeRFail/*.
```shell
cd ..
```

Run the script to transfer the rendered dataset to the test path, or you can copy it manually. This makes it easy to test the results with the same parameters again if needed in the future.

#### parameters

The following are some of the optional parameters.<br>
**--from_dir:** Root directory of the transfer source.<br>
**--e:** Limit the maximum change value per pixel (L<sub>&infin;</sub> norm) in the range 0-255.<br>
**--label:** Name of the scene object to attack.<br>
**--model_name:** Classification model name.<br>
**--attack_target_label_int:** Index of the targeted object to attack. If there are untargeted attack, this index shold be the index of the ground truth object.<br>
**--m1:** In UAP or NeRFail attacks, this parameter is used to improve the confidence of misclassification during the attack.<br>
**--m2:** In UAP or NeRFail attack methods, this parameter is used to speed up.<br>
**--method_name:** Attack method name.<br>
**--base_mask_image_number:** Number of images selected as 3D point sets.<br>

```shell
python transfer_files.py --from_dir ./Create_spatial_point_set/logs/blender_paper_lego --model_name inception --label lego --method_name NeRFail_S --target_class_idx n --e 32 --step 1 --base_mask_image_number 3
```

Test the NeRF rendering results using the appropriate experimental parameters, before doing so you should make sure that the rendering results have been located under the correct path.

Example 1: Testing test set images rendered by NeRF training using NeRFail-S attacked result.

```shell
python model_test.py --model_name inception --label lego --method_name NeRFail_S --target_class_idx n --e 32 --step 1 --base_mask_image_number 3 --setname test
```

Example 2: Testing val set images rendered by NeRF training using NeRFail-S attacked result.

```shell
python model_test.py --model_name inception --label lego --method_name NeRFail_S --target_class_idx n --e 32 --step 1 --base_mask_image_number 3 --setname val
```

Example 3: Testing test set images rendered by NeRF training using NeRFail attacked result.

```shell
python model_test.py --model_name inception --label lego --method_name NeRFail --target_class_idx n --e 32 --step 1 --base_mask_image_number 3 --setname test --m1 8 --m2 100
```

Example 4: Testing val set images rendered by NeRF training using NeRFail attacked result.

```shell
python model_test.py --model_name inception --label lego --method_name NeRFail --target_class_idx n --e 32 --step 1 --base_mask_image_number 3 --setname val --m1 8 --m2 100
```

### Randomness
Because the program was optimized and modified many times during the experimental process, and the randomness of the NeRF and classification models during training and attacking was not fully qualified, you may not get exactly the same results as we did, but the trends should match.<br>
It is worth noting that when NeRF training starts, it may happen that the PSNR stays very small and does not improve. At this time, you can try to restart the NeRF training process 1 or 2 times.<br>
There are other parameters that affect the resulting attack success rate, such as the output resolution at which the NeRF model is trained. The ASR decreases significantly when half of the resolution settings are turned on. The ablation experiment of these other parameters deserves to be researched in the future.
