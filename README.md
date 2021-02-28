# Oil and Gas Seep Detection Segmentation Project
### This is a personal project on multiclass image segmentation problem of oil and gas see seepages identification.

## Data
790 256X256 images (Tiff I:16 unsigned integer 16 bit) and masks (L-mode = 8 bit grayscale) are provided.
![**Example of Image and Mask](https://github.com/EvgenyDyshlyuk/Oil_Seep_Detection/blob/master/figures/image_and_mask.png)

## Architechture
Fully convolutional U-Net architechture was selected for this task. 
![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/Unet.png)

Fully-convolutional implies that it doesn't contain fully-connected layers, but only convolutional, max-pooling, and batch normalization layers all of which are invariant to the size of an image. This allows the network to be able to accept images of any size (practically upsampling should be done on the image of even diminsion - so for UNET this implies image size multiple of 2^3 = 8, and for ResNetUnet image size mulitple 2^5 = 32).
Convolutional neural nets are not scale-invariant. For example, if one trains on the cats of the same size in pixels on images of a fixed resolution, the net would fail on images of smaller or larger sizes of cats. In order to overcome this problem, there are at least two methods )might be more in the literature):
- multi-scale training of images of different sizes in fully-convolutional nets in order to make the model more robust to changes in scale (used here with image augmentation by random resizing)
- having multi-scale architecture (not used here) sourse: (https://ai.stackexchange.com/questions/6274/how-can-i-deal-with-images-of-variable-dimensions-when-doing-image-segmentation)

- train images are iaugmented: rotation, horizontal flip
- test images not augmented

## Loss
![**U-Net](https://github.com/EvgenyDyshlyuk/Image_Segmentation_Capstone_Project/blob/master/figures/loss.png)


## Training
- Training Done on EC2 GPU (p2.xlarge)
