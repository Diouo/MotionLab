# MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm

<p align="left">
    <a href='https://www.youtube.com/watch?v=X5yFMSJLNcE'><img src='https://img.shields.io/badge/video-video-purple' alt='youtube video'></a>
    <a href='https://motionlab-anonymous.github.io/motionlab.github.io/'><img src='https://img.shields.io/badge/project-project-blue' alt='project page'></a>
</p>

An anonymous preliminary code of MotionLab, whose core is in ./rfmotion/models/modeltype/rfmotion.py.

## News
- [2025/01/23] release demo code
- [2025/01/23] release training code
- [2025/01/23] release evaluating code
- [2025/02/01] release codes of specialist models
- [2025/02/03] release checkpoints
## Folder Structure
```
├── checkpoints
│   ├── motionflow
│   │   ├── motionflow.ckpt
│   ├── clip-vit-large-patch14
│   ├── glove
│   ├── mdm-ldm
│   │   ├── motion_encoder.ckpt
│   │   ├── motionclip.pth.tar
│   ├── smpl
│   │   ├── SMPL_NEUTRAL.pkl
│   ├── smplh
│   │   ├── SMPLH_NEUTRAL.npz
│   ├── t2m
│   │   ├── Comp_v6_KLD01
├── datasets
│   ├── all
│   │   ├── new_joint_vecs
│   │   │   ├── 000000.npy
│   │   │   ├── 040000.npy
│   │   ├── new_joints
│   │   │   ├── 000000.npy
│   │   │   ├── 040000.npy
│   │   ├── texts
│   │   │   ├── 000000.txt
│   │   │   ├── 040000.txt
│   │   ├── train_humanml.txt
│   │   ├── train_motionfix.txt
│   │   ├── val_humanml.txt
│   │   ├── val_motionfix.txt
│   │   ├── test_humanml.txt
│   │   ├── test_motionfix.txt
├── experiments
│   ├── rfmotion
│   │   ├── SPECIFIED NAME OF EXPERIMENTS
│   │   │   ├── checkpoints
```


### 1. Setup Conda:
python: 3.9.20; torch: 2.1.1; pytorch-lightning: 1.9.4; cuda: 11.8.0;

```
conda create python=3.9 --name rfmotion
conda activate rfmotion
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download Dependencies:
The results should be placed as shown in Folder Structure, including glove,t2m, smpl and clip.
```
bash prepare/download_smpl_model.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_clip.sh
```

### 3.Prepare Datasets:
Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and [MotionFix](https://github.com/atnikos/motionfix) dataset.

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to process the AMASS data into HumanML3D format, then copy the results into "all" as shown in Folder Structure.

Follow the instructions in [MotionFix-Retarget](https://github.com/MotionLab-Anonymous/MotionFix-Retarget) to process the MotionFix data into HumanML3D format, then copy the results into "all" as shown in Folder Structure.

### 4. Download Checkpoint:
The results should be placed as shown in Folder Structure, including [motion_encoder.ckpt, motionclip.pth.tar, motionflow.ckpt](https://drive.google.com/drive/folders/1ph3otOlYqINvwFuvrt92nvzypZDs4haj?usp=drive_link).

## Demo the MotionLab
FFMPEG is necessary for exporting videos, otherwise only SMPL mesh can be exported.

You should first check the configure in ./configs/config_rfmotion.yam, to assign the checkpoint and task:

      DEMO:
        TYPE: "text" # for text-based motion generation; alongside "hint", "text_hint", "inbetween", "text_inbetween", "style", "source_text", "source_hint", "source_text_hint"
        CHECKPOINTS: "./checkpoints/motionflow/motionflow.ckpt"  # Pretrained model path
        
```
cd ./script
bash demo.sh
```

Notably, rendering the video directly here may result in poor export results, which may cause the video clarity to decrease and the lighting to be unclear. It is recommended to export the mesh and then render the video in professional 3D software like Blender.

## Train the MotionLab
You should first check the configure in ./configs/config_rfmotion.yaml
```
cd ./script
bash train_rfmotion.sh
```

## Evaluate the MotionLab
You should first check the configure in ./configs/config_rfmotion.yam, to assign the checkpoint and task:

      TEST:
            CHECKPOINTS: "./checkpoints/motionflow/motionflow.ckpt"  # Pretrained model path
              
      METRIC:
            TYPE: ["MaskedMetrics", "TM2TMetrics", "SourceTextMetrics", "SourceHintMetrics", "SourceTextHintMetrics", "InbetweenMetrics", "TextInbetweenMetrics","TextHintMetrics", "HintMetrics", "StyleMetrics", ]
```
cd ./script
bash test_rfmotion.sh
```

## Specialist Models
If you are intrested in the specialist models focousing on specific task, you can replace ./config/config_rfmotion.yaml with ./config/config_rfmotion_TASK.yaml. And the corresponding core code is the ./rfmotion/models/modeltype/rfmotion_seperate.py.

## Acknowledgements

Some codes are borrowed from [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [MotionFix](https://github.com/atnikos/motionfix), [MCM-LDM](https://github.com/XingliangJin/MCM-LDM), [diffusers](https://github.com/huggingface/diffusers).

## Citation
If you find MotionLab useful for your work please cite:
```
@article{       ,
  author    = {},
  title     = {MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm},
  journal   = {},
  year      = {2025},
}
```


