<!-- TODO: - Upload Conda files and Evaluation code -->


# Towards Texture- And Shape-Independent 3D Keypoint Estimation in Birds



## Description
This repository contains the code for the paper: ["Towards Texture- And Shape-Independent 3D Keypoint Estimation in Birds"](https://arxiv.org/abs/2505.16633). 
We offer code and scripts for training and inference.

**Abstract:**\
In this paper, we present a texture-independent approach to estimate and track 3D joint positions of multiple pigeons. For this purpose, we build upon the existing 3D-MuPPET framework, which estimates and tracks the 3D poses of up to 10 pigeons using a multi-view camera setup. We extend this framework by using a segmentation method that generates silhouettes of the individuals, which are then used to estimate 2D keypoints. Following 3D-MuPPET, these 2D keypoints are triangulated to infer 3D poses, and identities are matched in the first frame and tracked in 2D across subsequent frames. Our proposed texture-independent approach achieves comparable accuracy to the original texture-dependent 3D-MuPPET framework. Additionally, we explore our approach's applicability to other bird species. To do that, we infer the 2D joint positions of four bird species without additional fine-tuning the model trained on pigeons and obtain preliminary promising results. Thus, we think that our approach serves as a solid foundation and inspires the development of more robust and accurate texture-independent pose estimation frameworks.

## Prerequisites

The presented framework is a texture independent adaption of [3D-MuPPET](https://alexhang212.github.io/3D-MuPPET/). The majority of the code is therefore based on the origial 3D-MuPPET.

Before starting, clone the following repositories into `Repositories/`: 
- [sort](https://github.com/abewley/sort) 
- [3DPOP](https://github.com/alexhang212/Dataset-3DPOP)
- [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) 
- [i-muppet](https://github.com/urs-waldmann/i-muppet/#i-muppet-interactive-multi-pigeon-pose-estimation-and-tracking) 


## Dataset and Weights
The results were optained on the following datsets:
- [3D-POP](https://doi.org/10.17617/3.HPBBC7)
- [Wild MuPPET](https://doi.org/10.17617/3.ENDMTI)
- [Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom/)

All new weights are in this repository, weights used in the paper can be found here:
- Yolo_Barn.pt from [3D-MuPPET](https://alexhang212.github.io/3D-MuPPET/), which can be downloaded [here](https://zenodo.org/records/10453890)
- sam_vit_h_4b8939.pth, sam_vit_b_01ec64.pth the weights for [SAM](https://segment-anything.com/), which can be downloaded [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)



## Inference on 3DPOP

To perform inference on the 3D-POP dataset: 
1. Download
the [3D-POP dataset](https://github.com/alexhang212/Dataset-3DPOP). 
2. Download the pretrained weights, mentioned above and place them in the `Weights/` directory.

### Inference Scripts
If you run the scripts with the following commands, this will use the pre-trained weights from 3D-MuPPET. If you want to specify custom weights/ checkpoints, you can input it as argument. Use ` -h ` to find out what argument names to use.

1.  **DLCSAM**:
> Note: The DeepLabCut weight parameter is the directory for the exported model.
```bash
# 2D Inference
python Inference/YOLODLCSAM_2DInference.py --input [input_video] 
# 3D Inference
python Inference/YOLODLCSAM_3DInference.py --dataset [3dpop_path] --seq [3dpop_sequence]
```

2. **DLCISO***:
> Note: The DeepLabCut weight parameter is the directory for the exported model.
```bash
# 2D Inference
python Inference/YOLODLCISO_2DInference.py --input [input_video] 
# 3D Inference
python Inference/YOLODLCISO_3DInference.py --dataset [3dpop_path] --seq [3dpop_sequence]
```


# Training
We also provide scripts for training all models in the paper. To download the dataset used in the manuscript, please download the "N6000" folder from the [3D-POP](https://doi.org/10.17617/3.HPBBC7) repository, and paste it in `TrainingData/` directory, which is the default directory.


## 1. Create Training Dataset
The DeepLabCut models were trained by segmenting the DLC training dataset provided in [3D-POP](https://doi.org/10.17617/3.HPBBC7). Our training dataset can be created by running this script.
1.  **DLCSAM**:
```bash
python Training/createDLCSAMdataset.py 
```
2.  **DLCISO**:
```bash
python Training/createDLCISOdataset.py 
```

## 2. Training DeepLabCut
For training run the following, with the path to DeepLabCut project folder. The config file can be found within the DeepLabCut folder structure. For more info and help, please refer to [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/README.html#). By default the training dataset is created in  `TrainingData/DLCSAM` or `TrainingData/DLCISO`.

```bash
python Training/DLC_Train.py --path [dataset_path]
```
