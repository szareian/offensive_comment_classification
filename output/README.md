__Important Notice__: Our model has been run and tested on google colab (The only extra required package is the "nlpaug") which can be installed using the following command:

__!pip3 install nlpaug__

The following files are included in the project/output directory:

__model_final.h5__: This file contains the trained model used to achieve our best accuracy score. It is trained on GPU ((using keras function CuDNNLSTM so it cannot be loaded/run on CPU).

__submission_final.csv__: Our model outputs this file. It can be submitted to the "Late submission" page on "Toxic Comment Classification Challenge" on [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/submit) in order to measure the accuracy on test set. All the accuracy measurements are done using Kaggle’s mean column-wise ROC AUC metric. (We had to go with this type of evaluation at the end becuase kaggle does not realease the labels for test set)