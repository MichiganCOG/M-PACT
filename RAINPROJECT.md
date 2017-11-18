#RAte INvariance (RAIN) Layer Project Page

This project contains the code and results used for the rate invariance layer project. The main problem being tackled is the normalization of changes to temporal frequency of the input sections.

Our Goal: Submit ASAP

## Table of Contents

* [Features to add to framework](#feattoadd)
* [Current Progress](#currentprogress)
* [Introduction](#intro)
* [Baseline Models](#baselinemodels)
* [Experiment 1 - Models trained using original datasets](#expt1)
* [Experiment 2 - Models trained using rate-modified datasets](#expt2)
* [Ideas for RAIN Layer](#ideas)

<a name="feattoadd"/>
## Features to add to Framework
* Extraction of only layer weights after training and save them as data dictionaries.
* Option to load from data dictionaries (trained models using the framework) or initialize models.
* Adding TF records to speed up pipeline and make it consistent.
* Confirm LRCN compatibility to TF records.
* Adding TSN model and confirm performance.
* Adding optical flow models and confirming their performance.

<a name="currentprogress"/>
## Current Progress

     EXPERIMENT 1: Trained using original datasets 
| Experiments      |  Coding in Progress  | Executing |  Debug  | Complete  (HMDB51)| Complete  (UCF101)|    
|:----------------:|:--------------------:|:---------:|:-------:|:-----------------:|:-----------------:|
| VGG16            |         &#9745;      |  &#9745;  | &#9745; |     &#9745;       |     &#9745;       |
| ResNet-50        |         &#9745;      |  &#9745;  | &#9745; |     &#9745;       |     &#9745;       |
| ResNet-50 + RAIN |         &#9974;      |  &#9974;  | &#9974; |     &#9974;       |     &#9974;       |

     EXPERIMENT 2: Trained using rate-modified datasets 
| Experiments      |  Coding in Progress  | Executing |  Debug  | Complete (HMDB51) | Complete (UCF101) |
|:----------------:|:--------------------:|:---------:|:-------:|:-----------------:|:-----------------:|
| VGG16            |         &#9974;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |
| ResNet-50        |         &#9745;      |  &#9745;  | &#9745; |       &#9745;     |       &#9745;     |
| ResNet-50 + RAIN |         &#9974;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |

#### LEGEND
Complete   -  &#9745; 
Incomplete -  &#9974;


<a name="intro"/>
## Introduction
**Problem**: Parameterzing temporal activity and extracting rate invariant features.

**Model**: end-to-end gradient based learning models (Neural Network), supervision using only action labels.

**Assumptions**

* Action: A set of body poses confined to a temporal order.
* Action Video: Single or multiple instances of an action within a video.
* Atmost, an action can take up the entire video (a complete single cycle over the entire set of datapoints).
* Frames per sec is consistent throughout all the videos in a dataset.
* Model revolves around the template formed by Spatial Transformer Networks(https://arxiv.org/abs/1506.02025).
* Assigning `soft` structure to extract necessary parameters, no intermediate or external supervision provided.


**Innovations**

* Differentiable, plug-n-play module which can be placed anywhere within a network where 1 independent dimension of the input is temporal.
* `Rate-normalized` outcome of the layer is fixed in size. Hence overcomes LSTM cell **memory issues** and allows **large videos to be applied to standard deep architectures in normal GPUs**.
* Custom layer definition for parameter extraction.
* Methodical demonstration of necessity to mathematically model rate-invariant characteristics in ANNs.

**Training data**: vanilla HMDB51 and UCF101 along with their rate modified verisons, with action labels.

**Testing data**: vanilla HMDB51 and UCF101 along with their rate modified versions.

**Applications**

* Improving action recognition, first person video etc.
* A plug-and-play layer to obtain normalized video representation.
* Application of the normalized representation to action segmentation, detection, etc.

<a name="baselinemodels"/>
## Baseline Models
* [ResNet50 + LSTM]:(https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) 
* [LRCN]           :()     
* [VGG16    + LSTM]:(https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

<a name="expt1"/>
## Experiment 1:  Models trained using original datasets

<a name="expt2"/>
## Experiment 2:  Models trained using rate-modified datasets

<a name="ideas"/>
## Ideas for RAIN Layer 
