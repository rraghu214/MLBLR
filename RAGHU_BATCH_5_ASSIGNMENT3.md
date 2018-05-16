# RAGHU_BATCH_5_ASSIGNMENT3
16/05/2018
## Dilated Convolution

In the classic convolution example, we considered a 3 X 3 kernel convolving on a 5 X 5 image (stride = 1). In this process apart from the edges the other pixels are read twice atleast. This leads to checkerboard issue where the same information is picked up more than once. We use dilated convolution to avoid this to some extent. Here the receptive field is increased, which gives better context of the image. As shown in the below image the pixels are spread apart and the gaps are padded with zeros. 

![Dilated Convolution](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/dilation.gif)

![Strides](https://raw.githubusercontent.com/hassony2/inria/master/wiki-images/dilated-convolution.png)

D is the gaps between the pixels (+1)

Following are the key features of this type:

1. Used for edge detection
2. Image segmentation if receptive features are broader
3.  Used for separating objects
4.  This is used in healthcare to determine tumours and cancer cells.
5.  Autonomous drone navigator


## Depthwise Convolution

This type of convolution is cost effective way of handling convolutions. 

A regular 3x3 convolution over 16 input channels and 32 output channels does the following: every single of the 16 channels is traversed by 32 3x3 kernels resulting in a total of 4608 (16x32x3x3) parameters. we now have 32 different feature maps for each of the 16 channels. now we take one feature map out of every 16 input channels and add them together. since we can do that 32 times, we get the 32 output channels we wanted.

For depthwise convolutions on the same setup we traverse each of the 16 channels with 1 3x3 kernel resulting in 16 feature maps. each of these feature maps in then traversed by 32 1x1 convolutions resulting in 512 (16x32) feature maps. now we take 1 feature map out of each of the 16-input channel and add them up. Since we can do that 32 times, we get the 32 output channels we wanted. the total number of parameters can be calculated by 16x3x3 + 16x32x1x1 = 656 parameters. 

Depthwise convolution followed by 1 X 1 convolution is known as Depthwise Separable Convolution.


# Data Augmentation 

Data augmentation is the process of generating more data sets for training using the existing samples itself. This can be useful when we don't have sufficient training data. Augmentation also generates a variety of data using various techniques like rotation, flipping, tuning the colours, contrast etc. This would avoid our model from being an overfit one. 

Consider the below image, had the model not been fed with the other varieties, then it would learn to detect that it is a watermelon only when placed on a horizontal plane as shown. 
 Here the data is generated based on rotation angles which enables the system to generalize.

![1](https://cdn-images-1.medium.com/max/1000/1*1FMKI3BuS-ZQFvF4FElxIQ.png)