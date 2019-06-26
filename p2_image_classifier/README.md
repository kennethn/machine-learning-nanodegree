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
Epoch 1/10.. Train loss: 4.261.. Validation loss: 0.180.. Accuracy: 0.094
Epoch 1/10.. Train loss: 4.059.. Validation loss: 0.183.. Accuracy: 0.112
Epoch 1/10.. Train loss: 4.095.. Validation loss: 0.196.. Accuracy: 0.161
Epoch 1/10.. Train loss: 3.928.. Validation loss: 0.201.. Accuracy: 0.210
Epoch 1/10.. Train loss: 3.650.. Validation loss: 0.189.. Accuracy: 0.228
Epoch 1/10.. Train loss: 3.498.. Validation loss: 0.181.. Accuracy: 0.288
Epoch 1/10.. Train loss: 3.360.. Validation loss: 0.197.. Accuracy: 0.298
Epoch 1/10.. Train loss: 3.331.. Validation loss: 0.149.. Accuracy: 0.366
Epoch 1/10.. Train loss: 3.178.. Validation loss: 0.134.. Accuracy: 0.411
Epoch 1/10.. Train loss: 2.862.. Validation loss: 0.095.. Accuracy: 0.415
Epoch 1/10.. Train loss: 2.664.. Validation loss: 0.117.. Accuracy: 0.471
Epoch 1/10.. Train loss: 2.429.. Validation loss: 0.110.. Accuracy: 0.502
Epoch 1/10.. Train loss: 2.571.. Validation loss: 0.099.. Accuracy: 0.513
Epoch 1/10.. Train loss: 2.373.. Validation loss: 0.080.. Accuracy: 0.525
Epoch 1/10.. Train loss: 2.223.. Validation loss: 0.090.. Accuracy: 0.529
Epoch 1/10.. Train loss: 2.081.. Validation loss: 0.073.. Accuracy: 0.568
Epoch 2/10.. Train loss: 2.116.. Validation loss: 0.068.. Accuracy: 0.547
Epoch 2/10.. Train loss: 1.984.. Validation loss: 0.068.. Accuracy: 0.591
Epoch 2/10.. Train loss: 2.047.. Validation loss: 0.087.. Accuracy: 0.564
Epoch 2/10.. Train loss: 1.880.. Validation loss: 0.080.. Accuracy: 0.587
Epoch 2/10.. Train loss: 1.795.. Validation loss: 0.076.. Accuracy: 0.635
Epoch 2/10.. Train loss: 1.796.. Validation loss: 0.067.. Accuracy: 0.607
Epoch 2/10.. Train loss: 1.692.. Validation loss: 0.047.. Accuracy: 0.651
Epoch 2/10.. Train loss: 1.614.. Validation loss: 0.077.. Accuracy: 0.627
Epoch 2/10.. Train loss: 1.624.. Validation loss: 0.068.. Accuracy: 0.613
Epoch 2/10.. Train loss: 1.595.. Validation loss: 0.053.. Accuracy: 0.643
Epoch 2/10.. Train loss: 1.698.. Validation loss: 0.053.. Accuracy: 0.658
Epoch 2/10.. Train loss: 1.408.. Validation loss: 0.038.. Accuracy: 0.673
Epoch 2/10.. Train loss: 1.610.. Validation loss: 0.052.. Accuracy: 0.679
Epoch 2/10.. Train loss: 1.407.. Validation loss: 0.065.. Accuracy: 0.695
Epoch 2/10.. Train loss: 1.621.. Validation loss: 0.060.. Accuracy: 0.696
Epoch 2/10.. Train loss: 1.457.. Validation loss: 0.038.. Accuracy: 0.683
Epoch 2/10.. Train loss: 1.476.. Validation loss: 0.042.. Accuracy: 0.718
Epoch 2/10.. Train loss: 1.371.. Validation loss: 0.044.. Accuracy: 0.734
Epoch 2/10.. Train loss: 1.472.. Validation loss: 0.042.. Accuracy: 0.707
Epoch 2/10.. Train loss: 1.518.. Validation loss: 0.045.. Accuracy: 0.705
Epoch 2/10.. Train loss: 1.325.. Validation loss: 0.030.. Accuracy: 0.727
Epoch 3/10.. Train loss: 1.416.. Validation loss: 0.036.. Accuracy: 0.724
Epoch 3/10.. Train loss: 1.019.. Validation loss: 0.034.. Accuracy: 0.726
Epoch 3/10.. Train loss: 1.273.. Validation loss: 0.026.. Accuracy: 0.742
Epoch 3/10.. Train loss: 1.144.. Validation loss: 0.023.. Accuracy: 0.741
Epoch 3/10.. Train loss: 1.070.. Validation loss: 0.026.. Accuracy: 0.770
Epoch 3/10.. Train loss: 1.182.. Validation loss: 0.039.. Accuracy: 0.765
Epoch 3/10.. Train loss: 1.246.. Validation loss: 0.034.. Accuracy: 0.763
Epoch 3/10.. Train loss: 1.223.. Validation loss: 0.024.. Accuracy: 0.765
Epoch 3/10.. Train loss: 1.202.. Validation loss: 0.031.. Accuracy: 0.765
Epoch 3/10.. Train loss: 1.385.. Validation loss: 0.041.. Accuracy: 0.736
Epoch 3/10.. Train loss: 1.213.. Validation loss: 0.022.. Accuracy: 0.766
Epoch 3/10.. Train loss: 1.170.. Validation loss: 0.022.. Accuracy: 0.736
Epoch 3/10.. Train loss: 1.294.. Validation loss: 0.029.. Accuracy: 0.759
Epoch 3/10.. Train loss: 1.045.. Validation loss: 0.026.. Accuracy: 0.756
Epoch 3/10.. Train loss: 1.030.. Validation loss: 0.031.. Accuracy: 0.788
Epoch 3/10.. Train loss: 1.228.. Validation loss: 0.025.. Accuracy: 0.806
Epoch 3/10.. Train loss: 0.994.. Validation loss: 0.016.. Accuracy: 0.788
Epoch 3/10.. Train loss: 1.098.. Validation loss: 0.025.. Accuracy: 0.768
Epoch 3/10.. Train loss: 1.139.. Validation loss: 0.026.. Accuracy: 0.781
Epoch 3/10.. Train loss: 1.104.. Validation loss: 0.018.. Accuracy: 0.783
Epoch 4/10.. Train loss: 1.234.. Validation loss: 0.026.. Accuracy: 0.802
Epoch 4/10.. Train loss: 1.110.. Validation loss: 0.028.. Accuracy: 0.792
Epoch 4/10.. Train loss: 1.030.. Validation loss: 0.017.. Accuracy: 0.798
Epoch 4/10.. Train loss: 1.024.. Validation loss: 0.014.. Accuracy: 0.786
Epoch 4/10.. Train loss: 0.988.. Validation loss: 0.015.. Accuracy: 0.830
Epoch 4/10.. Train loss: 1.051.. Validation loss: 0.017.. Accuracy: 0.795
Epoch 4/10.. Train loss: 0.916.. Validation loss: 0.019.. Accuracy: 0.814
Epoch 4/10.. Train loss: 0.882.. Validation loss: 0.022.. Accuracy: 0.803
Epoch 4/10.. Train loss: 1.047.. Validation loss: 0.024.. Accuracy: 0.829
Epoch 4/10.. Train loss: 0.895.. Validation loss: 0.023.. Accuracy: 0.820
Epoch 4/10.. Train loss: 0.946.. Validation loss: 0.027.. Accuracy: 0.808
Epoch 4/10.. Train loss: 1.013.. Validation loss: 0.017.. Accuracy: 0.813
Epoch 4/10.. Train loss: 0.987.. Validation loss: 0.016.. Accuracy: 0.805
Epoch 4/10.. Train loss: 1.041.. Validation loss: 0.019.. Accuracy: 0.800
Epoch 4/10.. Train loss: 0.988.. Validation loss: 0.018.. Accuracy: 0.806
Epoch 4/10.. Train loss: 1.124.. Validation loss: 0.026.. Accuracy: 0.827
Epoch 4/10.. Train loss: 1.001.. Validation loss: 0.024.. Accuracy: 0.813
Epoch 4/10.. Train loss: 0.854.. Validation loss: 0.020.. Accuracy: 0.786
Epoch 4/10.. Train loss: 0.861.. Validation loss: 0.020.. Accuracy: 0.827
Epoch 4/10.. Train loss: 0.786.. Validation loss: 0.018.. Accuracy: 0.824
Epoch 4/10.. Train loss: 0.884.. Validation loss: 0.011.. Accuracy: 0.819
Epoch 5/10.. Train loss: 0.761.. Validation loss: 0.010.. Accuracy: 0.805
Epoch 5/10.. Train loss: 0.862.. Validation loss: 0.016.. Accuracy: 0.804
Epoch 5/10.. Train loss: 0.870.. Validation loss: 0.017.. Accuracy: 0.817
Epoch 5/10.. Train loss: 0.791.. Validation loss: 0.015.. Accuracy: 0.827
Epoch 5/10.. Train loss: 0.862.. Validation loss: 0.010.. Accuracy: 0.841
Epoch 5/10.. Train loss: 0.993.. Validation loss: 0.012.. Accuracy: 0.833
Epoch 5/10.. Train loss: 0.878.. Validation loss: 0.016.. Accuracy: 0.843
Epoch 5/10.. Train loss: 0.847.. Validation loss: 0.008.. Accuracy: 0.825
Epoch 5/10.. Train loss: 0.920.. Validation loss: 0.023.. Accuracy: 0.821
Epoch 5/10.. Train loss: 0.966.. Validation loss: 0.018.. Accuracy: 0.842
Epoch 5/10.. Train loss: 0.870.. Validation loss: 0.011.. Accuracy: 0.830
Epoch 5/10.. Train loss: 0.690.. Validation loss: 0.010.. Accuracy: 0.857
Epoch 5/10.. Train loss: 0.828.. Validation loss: 0.014.. Accuracy: 0.835
Epoch 5/10.. Train loss: 0.962.. Validation loss: 0.012.. Accuracy: 0.835
Epoch 5/10.. Train loss: 0.821.. Validation loss: 0.008.. Accuracy: 0.820
Epoch 5/10.. Train loss: 0.940.. Validation loss: 0.006.. Accuracy: 0.825
Epoch 5/10.. Train loss: 0.937.. Validation loss: 0.019.. Accuracy: 0.834
Epoch 5/10.. Train loss: 0.939.. Validation loss: 0.022.. Accuracy: 0.822
Epoch 5/10.. Train loss: 0.777.. Validation loss: 0.014.. Accuracy: 0.837
Epoch 5/10.. Train loss: 0.769.. Validation loss: 0.021.. Accuracy: 0.840
Epoch 5/10.. Train loss: 0.861.. Validation loss: 0.013.. Accuracy: 0.855
Epoch 6/10.. Train loss: 0.823.. Validation loss: 0.009.. Accuracy: 0.850
Epoch 6/10.. Train loss: 0.793.. Validation loss: 0.011.. Accuracy: 0.839
Epoch 6/10.. Train loss: 0.848.. Validation loss: 0.017.. Accuracy: 0.859
Epoch 6/10.. Train loss: 0.750.. Validation loss: 0.017.. Accuracy: 0.860
Epoch 6/10.. Train loss: 0.868.. Validation loss: 0.024.. Accuracy: 0.839
Epoch 6/10.. Train loss: 0.809.. Validation loss: 0.015.. Accuracy: 0.851
Epoch 6/10.. Train loss: 0.876.. Validation loss: 0.017.. Accuracy: 0.833
Epoch 6/10.. Train loss: 0.744.. Validation loss: 0.023.. Accuracy: 0.836
Epoch 6/10.. Train loss: 0.842.. Validation loss: 0.014.. Accuracy: 0.854
Epoch 6/10.. Train loss: 0.824.. Validation loss: 0.013.. Accuracy: 0.850
Epoch 6/10.. Train loss: 0.752.. Validation loss: 0.014.. Accuracy: 0.861
Epoch 6/10.. Train loss: 0.686.. Validation loss: 0.009.. Accuracy: 0.854
Epoch 6/10.. Train loss: 0.788.. Validation loss: 0.012.. Accuracy: 0.858
Epoch 6/10.. Train loss: 0.779.. Validation loss: 0.023.. Accuracy: 0.851
Epoch 6/10.. Train loss: 0.779.. Validation loss: 0.012.. Accuracy: 0.864
Epoch 6/10.. Train loss: 0.840.. Validation loss: 0.009.. Accuracy: 0.843
Epoch 6/10.. Train loss: 0.738.. Validation loss: 0.014.. Accuracy: 0.835
Epoch 6/10.. Train loss: 0.924.. Validation loss: 0.016.. Accuracy: 0.830
Epoch 6/10.. Train loss: 0.836.. Validation loss: 0.014.. Accuracy: 0.860
Epoch 6/10.. Train loss: 0.692.. Validation loss: 0.010.. Accuracy: 0.862
Epoch 7/10.. Train loss: 0.976.. Validation loss: 0.010.. Accuracy: 0.856
Epoch 7/10.. Train loss: 0.691.. Validation loss: 0.013.. Accuracy: 0.862
Epoch 7/10.. Train loss: 0.778.. Validation loss: 0.018.. Accuracy: 0.855
Epoch 7/10.. Train loss: 0.663.. Validation loss: 0.015.. Accuracy: 0.868
Epoch 7/10.. Train loss: 0.698.. Validation loss: 0.015.. Accuracy: 0.854
Epoch 7/10.. Train loss: 0.720.. Validation loss: 0.014.. Accuracy: 0.851
Epoch 7/10.. Train loss: 0.678.. Validation loss: 0.014.. Accuracy: 0.853
Epoch 7/10.. Train loss: 0.789.. Validation loss: 0.016.. Accuracy: 0.865
Epoch 7/10.. Train loss: 0.608.. Validation loss: 0.031.. Accuracy: 0.848
Epoch 7/10.. Train loss: 0.836.. Validation loss: 0.029.. Accuracy: 0.860
Epoch 7/10.. Train loss: 0.617.. Validation loss: 0.028.. Accuracy: 0.867
Epoch 7/10.. Train loss: 0.727.. Validation loss: 0.024.. Accuracy: 0.858
Epoch 7/10.. Train loss: 0.576.. Validation loss: 0.016.. Accuracy: 0.850
Epoch 7/10.. Train loss: 0.755.. Validation loss: 0.006.. Accuracy: 0.858
Epoch 7/10.. Train loss: 0.822.. Validation loss: 0.004.. Accuracy: 0.880
Epoch 7/10.. Train loss: 0.770.. Validation loss: 0.005.. Accuracy: 0.879
Epoch 7/10.. Train loss: 0.744.. Validation loss: 0.005.. Accuracy: 0.857
Epoch 7/10.. Train loss: 0.845.. Validation loss: 0.005.. Accuracy: 0.858
Epoch 7/10.. Train loss: 0.910.. Validation loss: 0.008.. Accuracy: 0.865
Epoch 7/10.. Train loss: 0.662.. Validation loss: 0.010.. Accuracy: 0.861
Epoch 7/10.. Train loss: 0.745.. Validation loss: 0.008.. Accuracy: 0.866
Epoch 8/10.. Train loss: 0.806.. Validation loss: 0.007.. Accuracy: 0.876
Epoch 8/10.. Train loss: 0.692.. Validation loss: 0.005.. Accuracy: 0.867
Epoch 8/10.. Train loss: 0.552.. Validation loss: 0.006.. Accuracy: 0.846
Epoch 8/10.. Train loss: 0.710.. Validation loss: 0.005.. Accuracy: 0.863
Epoch 8/10.. Train loss: 0.633.. Validation loss: 0.003.. Accuracy: 0.867
Epoch 8/10.. Train loss: 0.792.. Validation loss: 0.003.. Accuracy: 0.875
Epoch 8/10.. Train loss: 0.759.. Validation loss: 0.003.. Accuracy: 0.873
Epoch 8/10.. Train loss: 0.671.. Validation loss: 0.005.. Accuracy: 0.877
Epoch 8/10.. Train loss: 0.603.. Validation loss: 0.010.. Accuracy: 0.865
Epoch 8/10.. Train loss: 0.702.. Validation loss: 0.016.. Accuracy: 0.864
Epoch 8/10.. Train loss: 0.846.. Validation loss: 0.013.. Accuracy: 0.858
Epoch 8/10.. Train loss: 0.706.. Validation loss: 0.010.. Accuracy: 0.866
Epoch 8/10.. Train loss: 0.597.. Validation loss: 0.009.. Accuracy: 0.866
Epoch 8/10.. Train loss: 0.733.. Validation loss: 0.007.. Accuracy: 0.880
Epoch 8/10.. Train loss: 0.607.. Validation loss: 0.007.. Accuracy: 0.874
Epoch 8/10.. Train loss: 0.681.. Validation loss: 0.010.. Accuracy: 0.883
Epoch 8/10.. Train loss: 0.617.. Validation loss: 0.015.. Accuracy: 0.877
Epoch 8/10.. Train loss: 0.705.. Validation loss: 0.007.. Accuracy: 0.864
Epoch 8/10.. Train loss: 0.656.. Validation loss: 0.007.. Accuracy: 0.866
Epoch 8/10.. Train loss: 0.612.. Validation loss: 0.005.. Accuracy: 0.864
Epoch 9/10.. Train loss: 0.760.. Validation loss: 0.008.. Accuracy: 0.866
Epoch 9/10.. Train loss: 0.652.. Validation loss: 0.017.. Accuracy: 0.868
Epoch 9/10.. Train loss: 0.575.. Validation loss: 0.023.. Accuracy: 0.862
Epoch 9/10.. Train loss: 0.530.. Validation loss: 0.023.. Accuracy: 0.853
Epoch 9/10.. Train loss: 0.754.. Validation loss: 0.016.. Accuracy: 0.858
Epoch 9/10.. Train loss: 0.837.. Validation loss: 0.007.. Accuracy: 0.887
Epoch 9/10.. Train loss: 0.676.. Validation loss: 0.007.. Accuracy: 0.860
Epoch 9/10.. Train loss: 0.756.. Validation loss: 0.007.. Accuracy: 0.873
Epoch 9/10.. Train loss: 0.614.. Validation loss: 0.013.. Accuracy: 0.825
Epoch 9/10.. Train loss: 0.732.. Validation loss: 0.010.. Accuracy: 0.844
Epoch 9/10.. Train loss: 0.826.. Validation loss: 0.007.. Accuracy: 0.851
Epoch 9/10.. Train loss: 0.677.. Validation loss: 0.006.. Accuracy: 0.868
Epoch 9/10.. Train loss: 0.707.. Validation loss: 0.005.. Accuracy: 0.862
Epoch 9/10.. Train loss: 0.706.. Validation loss: 0.007.. Accuracy: 0.866
Epoch 9/10.. Train loss: 0.622.. Validation loss: 0.012.. Accuracy: 0.860
Epoch 9/10.. Train loss: 0.722.. Validation loss: 0.014.. Accuracy: 0.858
Epoch 9/10.. Train loss: 0.693.. Validation loss: 0.012.. Accuracy: 0.848
Epoch 9/10.. Train loss: 0.532.. Validation loss: 0.008.. Accuracy: 0.861
Epoch 9/10.. Train loss: 0.674.. Validation loss: 0.007.. Accuracy: 0.883
Epoch 9/10.. Train loss: 0.556.. Validation loss: 0.004.. Accuracy: 0.881
Epoch 9/10.. Train loss: 0.637.. Validation loss: 0.007.. Accuracy: 0.870
Epoch 10/10.. Train loss: 0.579.. Validation loss: 0.013.. Accuracy: 0.838
Epoch 10/10.. Train loss: 0.745.. Validation loss: 0.019.. Accuracy: 0.841
Epoch 10/10.. Train loss: 0.519.. Validation loss: 0.017.. Accuracy: 0.851
Epoch 10/10.. Train loss: 0.625.. Validation loss: 0.015.. Accuracy: 0.866
Epoch 10/10.. Train loss: 0.570.. Validation loss: 0.008.. Accuracy: 0.874
Epoch 10/10.. Train loss: 0.583.. Validation loss: 0.005.. Accuracy: 0.876
Epoch 10/10.. Train loss: 0.690.. Validation loss: 0.005.. Accuracy: 0.859
Epoch 10/10.. Train loss: 0.571.. Validation loss: 0.007.. Accuracy: 0.873
Epoch 10/10.. Train loss: 0.752.. Validation loss: 0.007.. Accuracy: 0.882
Epoch 10/10.. Train loss: 0.664.. Validation loss: 0.009.. Accuracy: 0.887
Epoch 10/10.. Train loss: 0.543.. Validation loss: 0.008.. Accuracy: 0.876
Epoch 10/10.. Train loss: 0.506.. Validation loss: 0.015.. Accuracy: 0.890
Epoch 10/10.. Train loss: 0.685.. Validation loss: 0.026.. Accuracy: 0.865
Epoch 10/10.. Train loss: 0.721.. Validation loss: 0.014.. Accuracy: 0.890
Epoch 10/10.. Train loss: 0.633.. Validation loss: 0.003.. Accuracy: 0.885
Epoch 10/10.. Train loss: 0.640.. Validation loss: 0.007.. Accuracy: 0.873
Epoch 10/10.. Train loss: 0.641.. Validation loss: 0.008.. Accuracy: 0.864
Epoch 10/10.. Train loss: 0.610.. Validation loss: 0.006.. Accuracy: 0.862
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
