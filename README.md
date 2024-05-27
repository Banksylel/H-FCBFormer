
# H-FCBFormer: Hierarchical Fully Convolutional Branch Transformer for Occlusal Contact Segmentation with Articulating Paper

Anonymous code for: H-FCBFormer: Hierarchical Fully Convolutional Branch Transformer for Occlusal Contact Segmentation with Articulating Paper

Authors: ANON


## Overview

### Abstract

Occlusal contact identification is a commonly used tool in the field of orthodontics and prosthodontics to identify temporomandibular alignment, malocclusion, and assess the quality of prosthetics. However, articulating paper methods show significant false positive indications of the contact area, leaving the identification of true occlusal indications to clinicians. To address this, we propose a multiclass Vision Transformer (ViT) and Fully Convolutional Network (FCN) ensemble semantic segmentation model with a combination hierarchical loss function, which we name as Hierarchical Fully Convolutional Branch Transformer (H-FCBFormer). We facilitate model training by generating semantic segmentation masks derived from expert annotated object-wise articulating paper masks of contacts and gold standard masks. The proposed method outperforms other machine learning baseline methods evaluated on our dataset and performs better than an expert in terms of accurately identifying object-wise and pixel-wise occlusal contact areas while taking significantly less time to identify them.


### Quantitative Results

<p align="center">
	<img width=900, src="git_images/quant-results.PNG"> <br />
	<em>
		Figure 1: Table comparing the mean segmentation metrics and (mean standard deviation) of our H-FCBFormer, Multiclass U-Net and Multiclass FCBFormer for relevant classes, Medically True Positive (MTP) (true contact within AP ink area), Medically False Positive (MFP) (true contact within AP ink area) and FULL Contact (union of MTP and MFP) detection tasks.
	</em>
</p>


### Qualitative Results

<p align="center">
	<img width=900, src="git_images/qual-results.PNG"> <br />
	<em>
		 Figure 2: Images containing the digital images, predicted masks and target masks, for patient 01 retest AP 100μm Passive and patient 01 retest AP 012μm Active. Each method is displayed for the FULL Contact class, MTP Contact class, and MFP Contact class. Green indicates the target mask (FN), red indicates the Predicted mask (FP) and yellow indicates the overlap/agreed location (TP)
	</em>
</p>


## Usage

### Preparation

+ Create and activate virtual environment:

```
python3 -m venv ~/H-FCBFormer-env
source ~/H-FCBFormer-env/bin/activate
```

+ Clone the repository and navigate to new directory:

```
git clone https://github.com/anon-account123/H-FCBFormer
cd ./H-FCBFormer
```

+ Install the requirements:

```
pip install -r requirements.txt
```

+ Install the latest version of pytorch:

[Pytorch](https://pytorch.org/get-started/locally/)

+ Download the [PVTv2-B3](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) VIT backbone weights to `./`

+ Download the [Pretrained](https://drive.google.com/file/d/1_6MzRjm3fp0x_ec9QHTDdBmKXVs68-r-/view) weights to `./`


### Data setup

+ Contact me via this github to reuest access to the data.

File Layout:

```
[PATH TO DATASET]/
	- train/
		- images/
	 		- img1.png
			- img2.png
			...
		- masks/
			- img1.png
			- img2.png
			...
	- val/
		- images/
		- masks/
```

Class Setup:

+ Train and val datasets should be randomly split (patient-wise) during preprocessing. 

+ 'images' folder contains the digital images, 'masks' contain the multiclass mask for the corresponding patient in the 'images' folder. 

+ Masks are stored as grey-scale images of the same file extenson and name of the corresponding digital image

+ Within the mask pixel values for the the 'Background' class is pixel val 0, Class 1 is pixel val 255, Class 2 is pixel val 254, ect. 
	- This was done so you can view the quality of the mask by opening the image and does not require a config file to map pixel ranges to classes.

+ We recommend storing images and masks as .PNG as it prevents lossy compression from changing the class values for some pixels in the masks.

+ Please change class_tree.txt to discribe your class hierarchy, where an indent indicates a child-class of the above class in the previous indent, where the above class is a undetected superclass (not detected by the model, is the sum of the child classes). eg.:

```
Background
FullContact
	TrueContact
	FalseContact
```

+ classes should be layed out with the first primary-class (leaf class) as the first class (255) in the masks to predict. E.g Background = 0, TrueContact = 255, FalseContact = 254

### Train

Train FCBFormer on the train split of a dataset:

```
python train.py ----save-path="[FULL PATH TO FOLDER]/Trained models/[EXPERIMENT NAME]" --model-weights="[FULL PATH TO FOLDER]/pvt_v2_b3.pth" --model-weights="[FULL PATH TO FOLDER]/FCBFormer_CVC.pt" --data-root="[FULL PATH TO TRAIN DATASET]" --val-dataset="[FULL PATH TO VAL DATASET]" --tree-root="[FULL PATH TO FOLDER]/class_tree.txt" --batch-size=4 --val-batch=25 --img-size=352 --test-remove=True --epochs=400 --num-classes=3 --num-workers=0 --calc-super=True --save-images-batch=True --save-images-batch-num=10 --include-background=True --no-ph-weights=True --hierarchical-loss=True
```

+ Replace `[FULL PATH TO FOLDER]` with the full file path to the repo on your device.

+ Replace `[EXPERIMENT NAME]` with the name of your experiment.

+ Replace `[FULL PATH TO TRAIN DATASET]` with the full file path to the dataset on your device. e.g. "C:/user/Dataset/train"

+ Replace `[FULL PATH TO VAL DATASET]` with the full file path to the dataset on your device. e.g. "C:/user/Dataset/val"


### Prediction and Evaluation

Predict and evaluate FCBFormer on the val split of a dataset:

```
python predictEval.py --train-dataset="[EXPERIMENT NAME]" --data-root="[FULL PATH TO VAL DATASET]" --full-ds=False --pre-split-val=True --model-weights="[FULL PATH TO FOLDER]/Trained models/[EXPERIMENT NAME]/best.pt" --pretrain-weights="[FULL PATH TO FOLDER]/pvt_v2_b3.pth" --img-size=352 --num-classes=3 --num-workers=0 --calc-super=True --include-background=True --include-std-div=True
```

+ Replace `[FULL PATH TO FOLDER]` with the full file path to the repo on your device.

+ Replace `[EXPERIMENT NAME]` with the name of your experiment.

+ Replace `[FULL PATH TO VAL DATASET]` with the full file path to the dataset on your device. e.g. "C:/user/Dataset/val"


### Agreement Evaluation

+ (if you have multiple folds) Move all predicted masks you want to evaluate into a new folder [FULL PATH TO PREDICTED MASKS FOLDER].

+ Agreement is calculated using the OFR masks of specified sensitivity for the same patient as the predicted AP masks. The system automatically finds the same patient for pred AP and GT OFR in their respective folders. So names have to include OFR and AP for the separate images. E.g. 'Z01Rd-AP012A' and 'Z01Rd-OFR200'

+ You need a separate folder of ground truth OFR (of the sensitivities you want to evaluate) images in a separate folder to compare your predicted masks to the OFR masks.

+ Only contains the OFR images you want to evaluate against the predicted masks. Only include OFR masks of the same patients in your predicted set of masks.

```
./dataset/agreement_comparison_masks/[OFR MASKS].png
```


Evaluate Predicted FCBFormer against ground truth OFR images:

```
python predictEval.py --mask-root="[FULL PATH TO PREDICTED MASKS FOLDER]" --invert-mask=False --ofr-gt-masks="[FULL PATH TO GT OFR MASKS]" --list-of-ofr=[LIST OF STRINGS OF OFR SENSITIVITEIS YOU WANT TO TEST] --conf-intervals=False
```

+ Replace `[FULL PATH TO PREDICTED MASKS FOLDER]` with the full file path to the predicted masks by the model.

+ Replace `[FULL PATH TO GT OFR MASKS]` with the full file path to the ground truth OFR masks of the same patients as the predicted masks. e.g. "C:/user/H-FCBFormer/dataset/agreement_comparison_masks"

+ Replace `[LIST OF STRINGS OF OFR SENSITIVITEIS YOU WANT TO TEST]` with a list of strings of the sensitites you want to include in the evaluation. e.g. ['40','50','100','200']
