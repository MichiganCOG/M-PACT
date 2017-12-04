# RAte INvariance (RAIN) Layer Project Page

This project contains the code and results used for the rate invariance layer project. The main problem being tackled is the normalization of changes to temporal frequency of the input sections.

Our Goal: Submit ASAP

## Table of Contents

* [Features to add to framework](#feattoadd)
* [Current Progress](#currentprogress)
* [Introduction](#intro)
* [Baseline Models](#baselinemodels)
* [RAIN Layers](#rainlayers)
    * [RAIN v1.0](#rainv1) 
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
| Experiments        |  Coding in Progress  | Executing |  Debug  | Complete  (HMDB51)| Complete  (UCF101)|    
|:------------------:|:--------------------:|:---------:|:-------:|:-----------------:|:-----------------:|
| VGG16              |         &#9745;      |  &#9974;  | &#9745; |     &#9974;       |     &#9974;       |
| ResNet-50          |         &#9745;      |  &#9745;  | &#9745; |     &#9745;       |     &#9974;       |
| ResNet-50 + RAINv1 |         &#9745;      |  &#9745;  | &#9745; |     &#9745;       |     &#9974;       |
| ResNet-50 + RAINv2 |         &#9745;      |  &#9745;  | &#9745; |     &#9745;       |     &#9974;       |
| ResNet-50 + RAINv3 |         &#9745;      |  &#9745;  | &#9745; |     &#9974;       |     &#9974;       |

     EXPERIMENT 2: Trained using rate-modified datasets 
| Experiments        |  Coding in Progress  | Executing |  Debug  | Complete (HMDB51) | Complete (UCF101) |
|:------------------:|:--------------------:|:---------:|:-------:|:-----------------:|:-----------------:|
| VGG16              |         &#9745;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |
| ResNet-50          |         &#9745;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |
| ResNet-50 + RAINv1 |         &#9745;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |
| ResNet-50 + RAINv2 |         &#9745;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |
| ResNet-50 + RAINv3 |         &#9745;      |  &#9974;  | &#9974; |       &#9974;     |       &#9974;     |

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

<a name="rainlayers"/>
## RAIN Layers
This section contains the implemented ideas for various versions of RAIN layers and their descriptions.

<a name="rainv1"/>
RAIN Layer v1.0
---------------
The primary concept utilized in the formulation of this version is: The use of sampling and phase offset akin to 1D signals should be sufficient to characterize a given action signal.

![RAINv1Paramnw] (/images/Paramnw.pdf)
![RAINv1extractlayer] (/images/extractlayer.pdf)

<a name="expt1"/>
Experiment 1:  Models trained using original datasets
========================================================
The first experiment lists the results of models trained using only the original datasets.

<a name="paramsexpt1"/>
Parameters 
----------
1. HMDB51 (For ResNet50 + LSTM and VGG16 + LSTM)
    * lr - 0.001   
    * momentum - 0.9   
    * wd - 0.001     
    * inputDims - 50 (250 for RAIN layers)
    * seqLength - 50    
    * outputDims - 51      
    * numVids - 3570    
    * size - 224        
    * nEpochs - 30    
    * saveFreq - 1      
    * valFreq - 3    
    * split - 1              
    * preprocessing:   
        * reduce framerate from 30 to 25 fps
        * resize to 256
        * random crop to 224
        * random flip
        * mean subtraction (R - 123.68, G - 116.778, B - 103.94)
        * extract 125 frame (loop if necessary) using offset
        * sample 25 from those 125
        * pad with 25 frames of zero
    
2. UCF 101 (For ResNet50 + LSTM and VGG16 + LSTM)
    * lr - 0.001   
    * momentum - 0.9   
    * wd - 0.001     
    * inputDims - 50     
    * seqLength - 50    
    * outputDims - 101      
    * numVids - 9537    
    * size - 224        
    * nEpochs - 30    
    * saveFreq - 1      
    * valFreq - 3    
    * split - 1              
    * preprocessing:   
        * resize to 256
        * random crop to 224
        * random flip
        * mean subtraction (R - 123.68, G - 116.778, B - 103.94)
        * extract 125 frame (loop if necessary) using offset
        * sample 25 from those 125
        * pad with 25 frames of zero

<a name="progressexpt1"/>
Progress
--------
    HMDB51 Baseline Experiments
|       Experiments        | Mean Recog. Accuracy(MRA) on Orig.  | Mean Recog. Accuracy(MRA) on Rate Modified  |
|:------------------------:|:----------------------------------: |:------------------------------------------: |
|   VGG16 + LSTM           |              --.--%                 |                  --.--%                     |
| ResNet50 + LSTM          |              43.46%                 |                  --.--%                     |

    HMDB51 Original RAIN Layer Experiments
|       Experiments        | Median of Extract Layer MRA  | Mean of Extract Layer MRA  | Max of Extract Layer MRA  |
|:------------------------:|:---------------------------: |:-------------------------: |:------------------------: |
| ResNet50 + RAINv1 + LSTM |             42.42%           |             33.53%         |          42.55%           |
| ResNet50 + RAINv2 + LSTM |             --.--%           |             43.53%         |          46.47%           |
| ResNet50 + RAINv3 + LSTM |             --.--%           |             --.--%         |          --.--%           |


    UCF101 Experiments
|       Experiments        | Mean Recog. Accuracy(MRA) on Orig.  | Mean Recog. Accuracy(MRA) on Rate Modified  |
|:------------------------:|:----------------------------------: |:------------------------------------------: |
|   VGG16 + LSTM           |              --.--%                 |                  --.--%                     |
| ResNet50 + LSTM          |              --.--%                 |                  --.--%                     |

    UCF101 Original RAIN Layer Experiments
|       Experiments        | Median of Extract Layer MRA  | Mean of Extract Layer MRA  | Max of Extract Layer MRA  |
|:------------------------:|:---------------------------: |:-------------------------: |:------------------------: |
| ResNet50 + RAINv1 + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv2 + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv3 + LSTM |             --.--%           |             --.--%         |          --.--%           |

### NOTES:

<a name="expt2"/>
Experiment 2:  Models trained using rate-modified datasets
========================================================
The second experiment lists the results of models trained using only the rate-modified datasets.

<a name="paramsexpt2"/>
Parameters 
----------
1. HMDB51 (For ResNet50 + LSTM and VGG16 + LSTM)
    * lr - 0.001  
    * momentum - 0.9 
    * wd - 0.001  
    * inputDims - 50    
    * seqLength - 50
    * outputDims - 51  
    * numVids - 35700 
    * size - 224        
    * nEpochs - 5 
    * saveFreq - 1  
    * valFreq - 3 
    * split - 1     
    * preprocessing:   
        * reduce framerate from 30 to 25 fps
        * resize to 256
        * random crop to 224
        * random flip, mean subtraction (R - 123.68, G - 116.778, B - 103.94)
        * extract 125 frame (loop if necessary) using offset
        * sample 25 from those 125
        * pad with 25 frames of zero
    
2. UCF 101 (For ResNet50 + LSTM and VGG16 + LSTM)
    * lr - 0.001   
    * momentum - 0.9  
    * wd - 0.001  
    * inputDims - 50     
    * seqLength - 50    
    * outputDims - 101
    * numVids - 95370 
    * size - 224    
    * nEpochs - 1
    * saveFreq - 1 
    * valFreq - 3 
    * split - 1
    * preprocessing:   
        * resize to 256
        * random crop to 224
        * random flip
        * mean subtraction (R - 123.68, G - 116.778, B - 103.94),  
        * extract 125 frame (loop if necessary) using offset
        * sample 25 from those 125
        * pad with 25 frames of zero

<a name="progressexpt2"/>
Progress
--------
    HMDB51 Experiments
|       Experiments        | Mean Recog. Accuracy(MRA) on Orig.  | Mean Recog. Accuracy(MRA) on Rate Modified  |
|:------------------------:|:----------------------------------: |:------------------------------------------: |
|   VGG16 + LSTM           |              --.--%                 |                  --.--%                     |
| ResNet50 + LSTM          |              --.--%                 |                  --.--%                     |
| ResNet50 + RAINv1 + LSTM |              --.--%                 |                  --.--%                     |

    UCF101 Experiments
|       Experiments        | Mean Recog. Accuracy(MRA) on Orig.  | Mean Recog. Accuracy(MRA) on Rate Modified  |
|:------------------------:|:----------------------------------: |:------------------------------------------: |
|   VGG16 + LSTM           |              --.--%                 |                  --.--%                     |
| ResNet50 + LSTM          |              --.--%                 |                  --.--%                     |
| ResNet50 + RAINv1 + LSTM |              --.--%                 |                  --.--%                     |


![Ex1:baselineSpeedup] (/images/resnet_UCF101Rate_vid23002_input.gif/)


### NOTES:

<a name="ideas"/>
Ideas for Future Versions of RAIN Layer 
===================
* V1:
    * Current implementation using offset and sampling parameters
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V2:
    * Idea: (alpha only) The immediate next idea we were thinking about was to understand if uniform sampling over the entire sequence is important. If retaining some information about the entire setup is necessary then why not remove the offset parameter and instead attempt to interpolate using sampling parameter only. In case the sampling parameter minimizes to uniform sampling then we have a strong case to retain it.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V3:
    * Idea: (phi only) Similarly to V2, the idea behind this version to remove the sampling parameter and retain the offset parameter. The sampling will be uniform. This will show the impact that splicing the input video based off of the offset parameter.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V4:
    * Idea: (direct indices) Another idea proposed was to extract indices directly from the parameterization network and use them to interpolate output of extract layer. An interesting case would be the ordering proposed by such a system, or rather the lack of ordering so to speak. Are we getting close to segmentation territory possibly ?
    * Non-Sorted output of Extract Layer
    * Sorted output of Extract Layer
* V5:
    * Idea: (phi start, phi end) Replace the sampling parameter with a second offset parameter indicating the end frame of the sampling region. L frames will then be uniformly sampled between the start offset parameter and end offset parameter.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V6:
    * Idea: (nonlinear rate transform) Allow the output of the extract layer to be parameters used to modify the input video by gradually slowing down the video from the default rate to some learned rate until it reaches the main action and then speeding it back up to the default rate. The beginning and end points of this sampling will still be the beginning and end of the input video.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V7:
    * Idea: (Increase L) When comparing the inputs to the baseline networks, we found that multiple quick cycles are provided as input to the baseline networks. Given this high amount of redundancy, a possible solution to our situation of capturing only a single cycle would be to increase L so that the network has a longer timeframe to look over and make a decision.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V8:
    * Idea: (Change activation of FC2 to ReLu) While observing the outcomes of V1 we found that the videos have a general tendency to get slowed down but rarely speed up. This enforces a scenario where the rates of RAIN output videos are not the same. This could possibly be due to shackling of alpha by sigmoid activation (although this still allows for a theoretic speed up). Instead we believe ReLu might be a possible solution to this issue.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V9:
    * Idea: (V7 + V8) The previous versions seeks to study the individual impacts of each suggested change while this version seeks to combine them.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
* V10:
    * Idea: (V2 + phi after sampling) Based off of v2 results, we found that it is benenficial to allow alpha to sample from the entire video. Only afterwards will we apply the offset to the clip that was sampled by alpha. We will then resample that output to the correct number of frames, L.
    * Mean of Extract Layer
    * Median of Extract Layer
    * Max of Extract Layer
    
* Alternate:
    * Pass the parameters through an LSTM before entering the RAIN layer.
