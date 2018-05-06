# Raghu_Batch5_Assignment1
05/05/2018

## Convolution

In layman terms convolution means combination. It refers to combining 2 matrices to provide an output value or matrix.  In neural networks, image recognition and classification are carried out using Convolution technique.

Consider a grayscale (size 28 X 28 X 1) image showing a single digit number. Our aim is to identify the number. Single digit can be from 0 to 9. Hence, we have 10 possible results. 
Every pixel's intensity in the image is represented by a value. 

**Note**: In 28 X 28 X1, 1 is the number of channels. It is 1 for grayscale and 3 for RGB images.

![](https://chrisjmccormick.files.wordpress.com/2015/01/layer_1.png)

By rule of thumb, we consider a random 3X3 matrix to slide on this image and extract the features. This results to a layer of 26 X 26. When convoluted again it results in 24 X 24, then 22 X 22 and so on till it reaches 10 X 10 where a well learnt neural network gives the correct output.

- 3 X 3 matrix used here is termed as _'filter'_.
- In the process of convolution when a filter is applied on the input, the result is nothing but sum of dot products of the 2 matrices.


------------------------------------------------------------------------------------------------
## Filters / Kernels

Filters are matrices that convolve on an image, used for feature extraction. Consider a single digit number as follows:
 __
 __|
 __|

 We consider 2 **"filters"**  to pull out the features from this image. Horizonal dashes "-" and the vertical ones "|".  Convolution results in:
 
 Feature Map 1:
  __                        
  __     
  __


Feature Map 2:

|
|

Number of filters is equal to the number of feature maps obtain post convolution. 

In CNN, at some point to compress the number of channels we reduce the number of filters and apply point convolution (1 X 1 filter) so as to optimize the model.


------------------------------------------------------------------------------------------------
### How to create an account on GitHub and upload a sample project

Git is a distributed version control system for source code maintenance. Following are the steps to go about..

Create an account in  [Git Hub](https://github.com/)

Install [Git SCM](https://git-scm.com/downloads)

    Initial settings..

![0.bh9snlp5q3](\:storage\0.bh9snlp5q3.png)

    Create a new repo


![0.lo25v5pomud](\:storage\0.lo25v5pomud.png)

    Once created, the next steps are intutive..


![0.qcboctw8fo](\:storage\0.qcboctw8fo.png)


    Navigate to the project folder and promote it to a Git folder by "git init"

![0.r3olhtpyvk7](\:storage\0.r3olhtpyvk7.png)

    Add the files to staging area, which is like a local versioning system.

![0.iwfi8nha2r](\:storage\0.iwfi8nha2r.png)

    Commit the changes to Staging area.
![0.szn82tq02pk](\:storage\0.szn82tq02pk.png)

    Set the remote repo URL for the code to be pushed to the Git server.

![0.xg95h49pfah](\:storage\0.xg95h49pfah.png)
  
     Push the changes
  
  ![0.ccbg3a1mdb](\:storage\0.ccbg3a1mdb.png)
  
     Verifying the changes..
  ![0.pllarprggb](\:storage\0.pllarprggb.png)
  
  Since this is a public repo users can initialize this remote repository and checkout the code to local by using **"git checkout <*branch_name*>"** and commit the same way!
