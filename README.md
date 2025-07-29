# HSCM
Hierarchical Sequential Context Modelling for  High-Fidelity Image Inpainting
**Pre-trained model**
----------------------
We released the pre-trained model 
[Google Drive](https://drive.google.com/drive/folders/13RVnOWo7iI8JMlxzWxhq89F2de62_A5V?usp=drive_link)

**Test masks**
----------------------
[Google Drive](https://drive.google.com/drive/folders/1zbqWaMd5hyZOhfppLZF_OGNJl1yFSo1h?usp=drive_link)

**Inpainting results**
----------------------
[Google Drive](https://drive.google.com/drive/folders/1g7jixSdx-06WmqTIPGDhtTSxCm3JGrQm?usp=drive_link)

**Getting Started**
----------------------
[Download pre-trained model]
Download the pre-trained model to `./checkpoints`

[Data Preparation]
Download the Datasets.
Set `config.yaml` with the corresponding paths at 'TEST_INPAINT_IMAGE_FLIST', 'TRAIN_INPAINT_IMAGE_FLIST' and 'TEST_MASK_FLIST'. Set the `--MAKS 1` for training,  and  `--MAKS 3` for testing.

For training, in `config.yml`, set the `--MAKS 1`, then run:
```
python train.py
```
For testing, in `config.yml`, set the `--MAKS 3` for the external fixed mask index, then run:
```
python test.py
```



