# SimCLR
Pytorch implementation of the paper
[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

* ADAM optimizer
* ExponentialLR schedular. No warmup or other exotics
* Batchsize of 256 via gradient accumulation

## Feature model
* [Resnet50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), where the first convolutional layer has a filter size of 3 instead of 7.  
* h() feature dimensionality: 2048
* z() learning head output dimensionality: 128

## Classifier model
* Simple 1 layer Neural network from 2048 to num_classes

## Classification Results

| Epochs | 100 | 200 | 
| ------ |-----| ------| 
| Paper | 83.9| 89.2 | 
| This repo |87.49 | 88.16 |

## Run
Train the feature extracting model (resnet). Note CIFAR10C inherits from datasets.CIFAR and provides the augmented image pairs. 

    python train_features.py --batch-size=64 --accumulation-steps=4 --tau=0.5 
                              --feature-size=128 --dataset-name=CIFAR10C --data-dir=path/to/your/data
    
Train the classifier model. Needs a saved feature model to extract features from images. 

    python train_classifier.py --load-model=models/modelname_timestamp.pt --dataset-name=CIFAR10 
                              --data-dir=path/to/your/data
