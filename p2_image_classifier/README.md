# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Code:

* [Image Classifier Project.ipynb](Image%20Classifier%20Project.ipynb) - Jupyter notebook with model training and testing
* [train.py](train.py) - Train a new network on a data set with train.py
* [predict.py](predict.py) - Predict flower name from an image with predict.py along with the probability of
that name.

## Command line output:

```console
root@725e72d845e3:/home/workspace/ImageClassifier$ python train.py --epochs 10 --learning_rate 0.001
Loading pretrained model...................................
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 553433881/553433881 [00:04<00:00, 112966943.22it/s]
Using  cuda:0 ...........................................
Training model..............................................
Epoch 1/10.. Train loss: 4.610.. Validation loss: 0.183.. Accuracy: 0.025
Epoch 1/10.. Train loss: 4.543.. Validation loss: 0.177.. Accuracy: 0.044
Epoch 1/10.. Train loss: 4.418.. Validation loss: 0.181.. Accuracy: 0.066
Epoch 1/10.. Train loss: 4.344.. Validation loss: 0.177.. Accuracy: 0.060

[...snip...]

Epoch 10/10.. Train loss: 0.682.. Validation loss: 0.005.. Accuracy: 0.872
Epoch 10/10.. Train loss: 0.690.. Validation loss: 0.005.. Accuracy: 0.879
Epoch 10/10.. Train loss: 0.635.. Validation loss: 0.007.. Accuracy: 0.878
Validating on test set..........................................
Test accuracy: 85%
Saving model checkpoint.........................................

root@725e72d845e3:/home/workspace/ImageClassifier$ python predict.py --img flowers/test/39/image_07035.jpg --checkpoint model_checkpoint.pth
Predictions for  flowers/test/39/image_07035.jpg :
47.32 % --  columbine
13.51 % --  fire lily
11.76 % --  siam tulip
7.83 % --  sweet pea
6.16 % --  monkshood
root@725e72d845e3:/home/workspace/ImageClassifier$ python predict.py --img flowers/test/49/image_06213.jpg --checkpoint model_checkpoint.pth
Predictions for  flowers/test/49/image_06213.jpg :
100.00 % --  oxeye daisy
0.00 % --  english marigold
0.00 % --  osteospermum
0.00 % --  barbeton daisy
0.00 % --  spring crocus
root@725e72d845e3:/home/workspace/ImageClassifier$ python predict.py --img flowers/test/20/image_04912.jpg --checkpoint model_checkpoint.pth
Predictions for  flowers/test/20/image_04912.jpg :
74.85 % --  giant white arum lily
7.24 % --  anthurium
5.03 % --  moon orchid
4.47 % --  cyclamen
2.63 % --  sword lily
```

## Verifying that AlexNet works:

(The accuracy is not great because I only tested with 1 epoch, just wanted to make sure it ran through end-to-end)

```console
root@430cfaed1069:/home/workspace/ImageClassifier# python train.py --epochs 1 --arch alexnet --learning_rate 0.001
Downloading: "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" to /root/.torch/models/alexnet-owt-4df8aa71.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 244418560/244418560 [00:02<00:00, 87608880.78it/s]
Loading alexnet pretrained model...........................
Using cuda:0 ...........................................
Training model..............................................
Epoch 1/1.. Train loss: 4.639.. Validation loss: 0.177.. Accuracy: 0.004
Epoch 1/1.. Train loss: 4.559.. Validation loss: 0.178.. Accuracy: 0.031
Epoch 1/1.. Train loss: 4.456.. Validation loss: 0.177.. Accuracy: 0.058

[...snip...]

Epoch 1/1.. Train loss: 2.425.. Validation loss: 0.080.. Accuracy: 0.499
Epoch 1/1.. Train loss: 2.259.. Validation loss: 0.091.. Accuracy: 0.500
Epoch 1/1.. Train loss: 2.322.. Validation loss: 0.098.. Accuracy: 0.507
Validating on test set..........................................
Test accuracy: 45%
Saving model checkpoint.........................................

root@430cfaed1069:/home/workspace/ImageClassifier# python predict.py --img flowers/test/49/image_06213.jpg --checkpoint model_checkpoint.pth
Predictions for  flowers/test/49/image_06213.jpg :
20.27 % --  black-eyed susan
10.95 % --  oxeye daisy
8.93 % --  barbeton daisy
7.29 % --  english marigold
6.80 % --  gazania
```
