# Open-Vocabulary Segmentation with Semantic-Assisted Calibration [CVPR 2024]
Yong Liu*, Sule Bai*, Guanbin Li, Yitong Wang, Yansong Tang
(*equal contribution)

The repository contains the official implementation of "Open-Vocabulary Segmentation with Semantic-Assisted Calibration"

[Paper](https://arxiv.org/abs/2312.04089)

<a href='https://arxiv.org/abs/2312.04089'><img src='https://img.shields.io/badge/ArXiv-2312.04089-red'></a> 



---
## 📖 Pipeline & Results
<p align="center">
 <img src="imgs/pipeline.png" width="88%">
 <img src="imgs/visual.png" width="50%">
 <img src="imgs/results.png" width="37.5%">
</p>






### Tab of Content
- [Installation](#1)
- [Data Preparation](#2)
- [Usage](#3)
  - [Training](#5)
  - [Evaluation](#4)
- [Cite](#6)

<span id="1"></span>


If you find any bugs due to carelessness on our part in organizing the code, feel free to contact us and point that!

### Installation
Please see [installation guide](./INSTALL.md).
   

<span id="2"></span>

### Data Preparation
Please follow the instruction of [ov-seg](https://github.com/facebookresearch/ov-seg) to prepare the training and test data. The data should be organized like:
```
$DETECTRON2_DATASETS/
  coco/                 # COCOStuff-171
  ADEChallengeData2016/ # ADE20K-150
  ADE20K_2021_17_01/    # ADE20K-847
  VOCdevkit/
    VOC2012/            # PASCALVOC-20
    VOC2010/            # PASCALContext-59, PASCALContext-459
```


<span id="3"></span>

### Usage

- #### Pretrained Weight
  We have provided the pretrained SCAN-VitL weights and the finetuned Contextual-shifted CLIP weights. Please download them from [here](https://drive.google.com/drive/folders/1obgHGQngtQms0u5YUJRnwd4y1IzME-c8?usp=drive_link).



#### Evaluation 

  <span id="4"></span>
  ```
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH>
  ```
  - Here is an example:
  ```
  python train_net.py --num-gpu 8 --eval-only --config-file configs/scan_vitL.yaml MODEL.WEIGHTS ./SCAN.pth DATASETS.TEST \(\"ade20k_sem_seg_val\",\) MODEL.CLIP_ADAPTER.REPLACE_RATIO 0.05 MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.75 MODEL.CLIP_ADAPTER.MASK_THR 0.55
  ```

<span id="5"></span>
#### Training
1. Train the segmentation model:
  ```
  python train_net.py  --config-file <CONFIG_FILE> --num-gpus <NUM_GPU>
  ```

  - Here is an example:

  ```
  python train_net.py  --num-gpu 8 --config-file configs/scan_vitL.yaml
  ```

2. Fuse segmentation model with finetuned CLIP.

  We have provided the [finetuned CLIP weights](https://drive.google.com/drive/folders/1obgHGQngtQms0u5YUJRnwd4y1IzME-c8?usp=drive_link). You can directly fuse the pretrained weights with the segmentation model to get the final model. The fuse command is:
  ```
  cd tools
  python replace_clip.py
  ```
  You need to specify the "clip_ckpt" and "ovseg_model" in the file according to your CLIP path and segmentation model path.


  (Optional) If you want to finetune the CLIP model from scratch, please follow  [ov-seg](https://github.com/facebookresearch/ov-seg) to prepare the corresponding data. The finetued command is:

  ```
  cd open_clip_training
  cd src
  bash scripts/finetune_VitL_with_mask.sh
  ```



<span id="6"></span>
### Cite 

If you find our work helpful, we'd appreciate it if you could cite our paper in your work.
```
@article{liu2023open,
  title={Open-Vocabulary Segmentation with Semantic-Assisted Calibration},
  author={Liu, Yong and Bai, Sule and Li, Guanbin and Wang, Yitong and Tang, Yansong},
  journal={arXiv preprint arXiv:2312.04089},
  year={2023}
}
```
