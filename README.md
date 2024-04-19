
## Installation

```bash
conda env create -f environment.yml
```

## Dataset Preparation

### CLEVR Dataset with Masks

Download the clevr dataset from [here](https://drive.google.com/uc?export=download&id=15FhXv-1x8T68ZFohOLyohyZgpGfMKmEO)

And put the downloaded clevr_with_masks.h5 file into CV703_Project/data directory.

### MS COCO 2017

Download COCO dataset (`2017 Train images`,`2017 Val images`,`2017 Train/Val annotations`) from [here](https://cocodataset.org/#download) and place them following this structure into CV703_Project/data directory:

```
COCO2017
   ├── annotations
   │    ├── instances_train2017.json
   │    ├── instances_val2017.json
   │    └── ...
   ├── train2017
   │    ├── 000000000009.jpg
   │    ├── ...
   │    └── 000000581929.jpg
   └── val2017
        ├── 000000000139.jpg
        ├── ...
        └── 000000581781.jpg
```

### Pretrained Checkpoint
Download the pretrained checkpoint from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/jiantong_zhao_mbzuai_ac_ae/EufPAWetphdOh3cnevfhGjkBEOZfw9fUvEOsk5w95c3fcQ?e=aVijHB) and put the files into CV703_Project/checkpoints directory.

## Evaluation

### Evaluate the stage 2 Slot Attention model on CLEVR dataset:

```bash
python eval_clevr.py
```

### Evaluate the stage 2 Slot Attention model on MS COCO 2017 dataset:

```bash
python eval_coco.py
```

## Training

### Train Stage 1 Slot Attention model on CLEVR dataset:

```bash
python train_clevr.py
```

### Train Stage 1 Slot Attention model on MS COCO 2017 dataset:

```bash
python train_coco.py
```

### Train Masked Autoencoder model

```bash
python train_MAE.py
```

### Train Stage 2 Slot Attention model on CLEVR dataset:

```bash
python train_clevr_stage_2.py
```

### Train Stage 2 Slot Attention model on MS COCO 2017 dataset:

```bash
python train_coco_stage_2.py
```

