# HSCM

**Pre-trained model**
----------------------
We released the pre-trained model 
[Google Drive]()

**Test masks**
----------------------
[Google Drive]()

**Inpainting results**
----------------------
[Google Drive]()

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



