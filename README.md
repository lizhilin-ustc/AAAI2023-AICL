# AAAI2023-AICL
The official implementation of "Actionness Inconsistency-guided Contrastive Learning for Weakly-supervised Temporal Action Localization"(AAAI2023)

## Abstract
Weakly-supervised temporal action localization (WTAL) aims to detect action instances given only video-level labels. To address the challenge, recent methods commonly employ a two-branch framework, consisting of a class-aware branch and a class-agnostic branch. In principle, the two branches are supposed to produce the same actionness activation. However, we observe that there are actually many inconsistent activation regions. These inconsistent regions usually contain some challenging segments whose semantic information (action or background) is ambiguous. In this work, we propose a novel Actionness Inconsistency-guided Contrastive Learning (AICL) method which utilizes the consistent segments to boost the representation learning of the inconsistent segments. Specifically, we first define the consistent and inconsistent segments by comparing the predictions of two branches and then construct positive and negative pairs between consistent segments and inconsistent segments for contrastive learning. In addition, to avoid the trivial case where there is no consistent sample, we introduce an action consistency constraint to control the difference between the two branches. We conduct extensive experiments on THUMOS14, ActivityNet v1.2, and ActivityNet v1.3 datasets, and the results show the effectiveness of AICL with state-of-the-art performance.

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

## Acknowledgement
This repository was based on the [ASL](https://github.com/layer6ai-labs/ASL) repo found.

The download link to the dataset is provided by [CoLA](https://github.com/zhang-can/CoLA).
