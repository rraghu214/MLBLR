# RAGHU_BATCH_5_ASSIGNMENT3

## Dilated Convolution

In the classic convolution example, consider a 3 X 3 kernel convolving on a 5 X 5 image (stride = 1). In this process apart from the edges the other pixels are read twice atleast. This leads to checkerboard issue where the same information is picked up more than once. We use dilated convolution to avoid this to some extent. Here the receptive field is increased, which gives better context of the image. As shown in the below image the pixels are spread apart and the gaps are padded with zeros. 

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

