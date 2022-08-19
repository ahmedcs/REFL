# Plain CIFAR10 Dataset (Not Preprocessed for Oort/FedScale)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

## Organization

The extracted `cifar10.tar.gz` file is splited into training and testing set folders. No IDs are assigned to clients and it is done while loading the dataset where samples are assigned to clients. 

# References
This dataset is covered in more detail at [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).