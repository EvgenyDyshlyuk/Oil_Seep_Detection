# Oil and Gas Seep Detection Segmentation Project
### This is a personal project on multiclass image segmentation problem for oil and gas see seepages identification.

## Data
- Images from SAR (Synthetic-aperture radar https://en.wikipedia.org/wiki/Synthetic-aperture_radar) are provided for training together with annotated masks images.
- 790 256X256 images (Tiff I:16 unsigned integer 16 bit) and masks (L-mode = 8 bit grayscale) are provided.
- 7 classes of seepages are annotated.
- Example of an image and corresponding mask are shown below.

![**Image and Mask](https://github.com/EvgenyDyshlyuk/Oil_Seep_Detection/blob/master/figures/image_and_mask.png)

## Architechture
Famous 2015 (https://arxiv.org/abs/1505.04597) fully convolutional U-Net architechture was selected for this task: 
![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/Unet.png)

- There are other approaches for Image Segmentation, but fully-convolutional nets are still very efficient/popular.
- Some improved approaches for UNet architecture exist as well, including such as when UNet is used on top of e.g. ResNet (pretrained or not), or other famous solutions for better feature extraction.
- Here I used simple UNet for simplicity at the start of the work on the problem.

Fully-convolutional implies that it doesn't contain fully-connected layers, but only convolutional, max-pooling, and batch normalization layers all of which are invariant to the size of an image. This allows the network to be able to accept images of any size (practically upsampling should be done on the image of even diminsion - so for UNET this implies image size multiple of 2^3 = 8, and for e.g. ResNetUnet image size mulitple 2^5 = 32). 
Convolutional neural nets are not scale-invariant. For example, if one trains on the cats of the same size in pixels on images of a fixed resolution, the net would fail on images of smaller or larger sizes of cats. In order to overcome this problem, there are at least two methods:
- multi-scale training of images of different sizes in fully-convolutional nets in order to make the model more robust to changes in scale. Random resizing (Zoom in/out) image augmentation can be used to overcome this problem.
- having multi-scale architecture (not used here) sourse: (https://ai.stackexchange.com/questions/6274/how-can-i-deal-with-images-of-variable-dimensions-when-doing-image-segmentation)
- In this particular project, I don't think image augmentation by random resizing can help (would be worth checking, but I didn't) because:
- The obtained images are most definitly satelite images (satelites are quite far and normally on the same orbit/distance - so no difference in distance/zoom). The physics involved includes waves on the sea (needs some medium speed winds) - so size of waves matters - zoom is probably not good because of this as well. The image contrast is mainly due to different density and surface tension on the oil-air and gas-air interface vs normal water-air interface. Due to different surface tension the shape of waves changes and the contrast arize. So only rotation (180 deg) and horizontal flip (no need for vertical because horizontal and rotation by 180 cover it) was used for augmentation. Other possibility is to pad images and use random crop, but should not make much difference because fully convolutional CNNs should be invariant to feature location on the image.
- train images are iaugmented by: random rotation (180 deg), and random horizontal flip
- test images are not augmented

## Loss
- The problem is highly imbalanced because the masks are quite sparse in this dataset (masks/background ~= 1/100). 
- High imbalance on learning classes (see below):

![**Class Imbalance](https://github.com/EvgenyDyshlyuk/Oil_Seep_Detection/blob/master/figures/class_imbalance.png)

- Good short description of image segmentation losses (https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5)
- Good implementation (with extra unused parameters - so removed here) (https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)
- Different Losses exist (see below):

![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/loss.png)

- BCELoss takes into account false positives (FP) - so not a good metric per se for segmentation tasks with a lot of background (as here)
- Dice (doesn't take into accout FP) is a good metric for imbalanced datasets
- Combo loss DiceBCELoss (Dice + weighted BCE) with small bce weight is a good way to go forward - exploits good convergence from BCE and concentrates on Dice
- Even better way is to use Focal Tversky loss - which helps to make attention on small segmentation masks and helps fighting imbalance problem



- Here I used DiceBCELoss and FocalTversky.
- I used BCELoss and DiceLoss as an additional metric to monitor training - these simple metrics help to understand what's going on in trainig.

2018 article A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (https://arxiv.org/abs/1810.07842)



## Training
- Training Done on EC2 GPU (p2.xlarge)
