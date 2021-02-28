# Oil and Gas Seep Detection Segmentation Project
### This is a personal project on multiclass image segmentation problem for oil and gas see seepages identification.

## Data
- images from SAR (Synthetic-aperture radar https://en.wikipedia.org/wiki/Synthetic-aperture_radar) are provided for training together with annotated masks images.
- 790 256X256 images (Tiff I:16 unsigned integer 16 bit) and masks (L-mode = 8 bit grayscale) are provided.
- 7 classes of seepages are annotated.

![**Image and Mask](https://github.com/EvgenyDyshlyuk/Oil_Seep_Detection/blob/master/figures/image_and_mask.png)

## Architechture
Famouse 2015 (https://arxiv.org/abs/1505.04597) fully convolutional U-Net architechture was selected for this task: 
![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/Unet.png)

- There are other approaches for Image Segmentation, but full convolutional nets are still very efficient/popular.
- Some improved approaches for UNet architecture exist as well, including such as when UNet is used on top of e.g. ResNet (pretrained or not), or other famous solutions for better feature extraction.
- Here I used simple UNet for simplicity at the start of the work on the problem.

Fully-convolutional implies that it doesn't contain fully-connected layers, but only convolutional, max-pooling, and batch normalization layers all of which are invariant to the size of an image. This allows the network to be able to accept images of any size (practically upsampling should be done on the image of even diminsion - so for UNET this implies image size multiple of 2^3 = 8, and for e.g. ResNetUnet image size mulitple 2^5 = 32).
Convolutional neural nets are not scale-invariant. For example, if one trains on the cats of the same size in pixels on images of a fixed resolution, the net would fail on images of smaller or larger sizes of cats. In order to overcome this problem, there are at least two methods:
- multi-scale training of images of different sizes in fully-convolutional nets in order to make the model more robust to changes in scale (used here with image augmentation by random resizing). Zoom in/out image augmentation can be used.
- having multi-scale architecture (not used here) sourse: (https://ai.stackexchange.com/questions/6274/how-can-i-deal-with-images-of-variable-dimensions-when-doing-image-segmentation)
- In this particular project, I don't think scale should be a problem, because:
- The obtained images are most definitly satelite images (satelites are quite far and normally on the same orbit/distance - so no difference in distance/zoom). The physics involved includes waves on the sea (needs some medium speed winds) - so size of waves matters - zoom is probably not good because of this as well. The image contrast is mainly due to different density and surface tension on the oil-air and gas-air interface vs normal water-air interface. Due to different surface tension the shape of waves changes and the contrast arize. So only rotation (180 deg) and horizontal flip (no need for vertical because horizontal and rotation by 180 cover it) was used for augmentation. Other possibility is to pad images and use random crop, but should not make much difference because fully convolutional CNNs should be invariant to feature location on the image.
- train images are iaugmented: rotation, horizontal flip
- test images not augmented

## Loss
![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/loss.png)


## Training
- Training Done on EC2 GPU (p2.xlarge)
