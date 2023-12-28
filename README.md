# NeRFail: Neural Radiance Fields-based multiview adversarial attack

Wenxiang Jiang*, Hanwei Zhang*†, Xi Wang, Zhongwen Guo†, Hao Wang<br>
(* Equal Contribution.)(† Corresponding Author.)

Abstract: *Adversarial attacks, \ie generating adversarial perturbations with a small magnitude to deceive deep neural networks, are important for investigating and improving model trustworthiness. Traditionally, the topic was scoped within 2D images or a full dataset without considering multiview. Benefiting from Neural Radiance Fields (NeRF), one can easily reconstruct a 3D scene with a Multi-Layer Perceptron (MLP) from given 2D views and synthesize photo-realistic renderings from novel vantages. This opens up a door to discussing the possibility of undertaking to attack multiview NeRF network with downstream tasks taking input information from different rendering angles, which we denote <font color=blue>**Ne**ural **R**adiance **F**ields-based multiview **a**dversar**i**a**l** Attack (NeRFail)</font>. The goal is, given one scene and a subset of views, to deceive the recognition tasks of different unknown view angles as well as given views. To do so, we propose a transformation to mapping from pixels to 3D points so that our attack generates multiview adversarial perturbations by attacking a subset of images with different views against the downstream classification model to images rendered by NeRF from other views. Experiments show that our multiview adversarial perturbations successfully obfuscate the downstream classification at both known and unknown views. Notably, when retraining another NeRF on the training data with generated multiview perturbations, we show that the perturbation can be inherited and reproduced from the attacking agnostic model.*

![NeRFail attack image](assets/NeRFail.png)

```shell
cd Create_spatial_point_set
cd nerf_pytorch
```

```shell
python run_nerf.py --config ../configs/lego.txt
```

```shell
cd ..
```

```shell
python nerf_to_coord.py --config ./configs/lego_coord.txt 
```

```shell
python create_index_and_dist.py --label lego --epochs 199999
```

```shell
cd ../tools
```

```shell
python dist_to_weight.py --label lego
```

```shell
cd ..
```


```shell
python model_train.py --model_name inception --num_classes 8 --data_dir ./data/nerf_synthetic
```
testing for classification model using original test image set
```shell
python model_test.py --model_name inception
```

```shell
python attack_IGSM_2D.py --label lego --model_name inception --e 32 --attack_target_label_int 4
```

```shell
python attack_NeRFail_S.py --label lego --model_name inception --e 32 --attack_target_label_int 4
```

```shell
python attack_NeRFail.py --label lego --model_name inception --e 32 --attack_target_label_int 4 --m1 8 --m2 100
```

```shell
python attack_UAP_2D.py --label lego --model_name inception --e 32 --attack_target_label_int 4 --m1 8 --m2 100
```

Clear logs/blender_paper_lego folder before using the attacked dataset to train nerf, or the nerf model will load the previously trained weights as the initial weight values.

```shell
cd Create_spatial_point_set
cd nerf_pytorch
```

```shell
python run_nerf.py --config ../configs/lego.txt --train_dir ../../output/inception/attack/lego/NeRFail_S_3P_100_to_n_e_32_a_2/train
```

The Nerf training process has a certain degree of randomness. When the PSNR remains low (PSNR<15), it may be due to getting stuck in a local optimal solution. In this case, you could try terminating the training and restarting.
