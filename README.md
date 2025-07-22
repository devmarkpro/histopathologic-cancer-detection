# Histopathologic Cancer Detection - Kaggle Competition

This repository contains code and resources for the Kaggle competition on Histopathologic Cancer Detection. The goal of the competition is to develop a model that can accurately classify images of histopathologic samples as either cancerous or non-cancerous.

## Dataset


In this dataset, you are provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. You are predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.

You can find the dataset on Kaggle: [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data).


## Install GraphViz

To visualize the model architecture, you need to install GraphViz. You can do this using the following command:

```bash
brew install graphviz
```
