CS231n
======

[CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/)

### Data Preprocessing

- Mean Subtraction
- Normalization
  - In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.
- PCA
  - There are two ways to use PCA in Image.
  - First, use 'eigenvector' directly
    - We treat image as a single *point* of 784 dimensions in MNIST case.
    - When it is visualized, we visualize eigen vectors(dimension of *784*), not a transformed image.(BE CAREFUL!)
    - for korean explanation, recommend this article, [[선형대수학 #6] 주성분분석(PCA)의 이해와 활용 :: 다크 프로그래머](http://darkpgmr.tistory.com/m/post/110)
  - second, use PCA as dimension reduction(truncating)
- **whitening**
  - divide every dimension by the eigenvalue

> In practice. We mention PCA/Whitening in these notes for completeness, but these transformations are not used with Convolutional Networks. However, it is very important to zero-center the data, and it is common to see normalization of every pixel as well. (**real?**)

- **the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).**


```
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
U,S,V = np.linalg.svd(cov)
Xwhite = Xrot / np.sqrt(S + 1e-5)
```

### Learning

- SGD
  - IF we use 10 batches, learning rate \alpha should be 1/10 * (learning rate of whole batch)

Visualizing and Understanding Convolutional Networks.pdf
=====================================

- imagenet 2013 winning team for classification
- It became known as the ZF Net (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers.

OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
======================================

- imagenet 2013 winning team of object localization task
- see bounding box approach(...)

Very Deep Convolutional Networks for Large-Scale Visual Recognition
======================================

- VGGNet imagenet 2014 winning team of object localization
