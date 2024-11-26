# AAAI2023-AICL
The official implementation of "Actionness Inconsistency-guided Contrastive Learning for Weakly-supervised Temporal Action Localization"(AAAI2023)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/actionness-inconsistency-guided-contrastive/weakly-supervised-action-localization-on-2)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on-2?p=actionness-inconsistency-guided-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/actionness-inconsistency-guided-contrastive/weakly-supervised-action-localization-on-1)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on-1?p=actionness-inconsistency-guided-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/actionness-inconsistency-guided-contrastive/weakly-supervised-action-localization-on)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on?p=actionness-inconsistency-guided-contrastive)

## Abstract
Weakly-supervised temporal action localization (WTAL) aims to detect action instances given only video-level labels. To address the challenge, recent methods commonly employ a two-branch framework, consisting of a class-aware branch and a class-agnostic branch. In principle, the two branches are supposed to produce the same actionness activation. However, we observe that there are actually many inconsistent activation regions. These inconsistent regions usually contain some challenging segments whose semantic information (action or background) is ambiguous. In this work, we propose a novel Actionness Inconsistency-guided Contrastive Learning (AICL) method which utilizes the consistent segments to boost the representation learning of the inconsistent segments. Specifically, we first define the consistent and inconsistent segments by comparing the predictions of two branches and then construct positive and negative pairs between consistent segments and inconsistent segments for contrastive learning. In addition, to avoid the trivial case where there is no consistent sample, we introduce an action consistency constraint to control the difference between the two branches. We conduct extensive experiments on THUMOS14, ActivityNet v1.2, and ActivityNet v1.3 datasets, and the results show the effectiveness of AICL with state-of-the-art performance.

## Results
|  Dataset         | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7| AVG(0.1:0.5) | AVG(0.1:0.7) |
| -----------      | --- | --- | ----| ----| ----| ---| -- | ---- | -----|
| THUMOS14         | 73.1| 67.8| 58.2| 48.7| 36.9|25.3| 14.9| 56.9| 46.4|

|  Dataset         | 0.5 | 0.75 | 0.95 | AVG(0.5:0.95) |
| -----------      | --- | --- | ----| ----|
| ActivityNet 1.2  | 49.6| 29.1| 5.9| 29.9|
| ActivityNet 1.3  | 44.2| 27.4| 5.8| 27.6|

## Preparation
CUDA Version: 11.7

Pytorch-gpu: 1.9.0

Numpy: 1.23.1 

Python: 3.9.7

GPU: NVIDIA 1080Ti

Dataset: Download the two-stream I3D features for THUMOS'14 to "DATA_PATH". You can download them from [Google Drive](https://drive.google.com/file/d/1paAv3FsqHtNsDO6M78mj7J3WqVf_CgSG/view?usp=sharing).

Update the data_path in "./scripts/train.sh" and "./scripts/inference.sh".

You can download our trained model from [here(Extract code:AICL)](https://pan.baidu.com/s/1L7ayrAEQ8frjnYY6VDd3tw).

## Training
```
    bash ./scripts/train.sh
```

## Inference
```
    bash ./scripts/inference.sh
```
## Implementation Details
Implementation details are shown in the supplementary material(supplement.pdf).


## Acknowledgement
This repository was based on the [ASL](https://github.com/layer6ai-labs/ASL) repo found.
The download link to the dataset is provided by [CoLA](https://github.com/zhang-can/CoLA).

## Citation
If this work is helpful for your research, please consider citing our works.
```
@inproceedings{li2023actionness,
  title={Actionness inconsistency-guided contrastive learning for weakly-supervised temporal action localization},
  author={Li, Zhilin and Wang, Zilei and Liu, Qinying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={1513--1521},
  year={2023}
}
```
