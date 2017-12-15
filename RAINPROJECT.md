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
    * [RAIN v2.0](#rainv2)
    * [RAIN v3.0](#rainv3)
    * [RAIN v4.0](#rainv4)
    * [RAIN v5.0](#rainv5)
    * [RAIN v6.0](#rainv6)
    * [RAIN v7.0](#rainv7)
    * [RAIN v8.0](#rainv8)
    * [RAIN v9.0](#rainv9)
    * [RAIN v10.0](#rainv10)
    * [RAIN v11.0](#rainv11)
    * [RAIN v12.0](#rainv12)
        * [RAIN v12.1](#rainv12.1)
    * [RAIN v14.0](#rainv14)
        * [RAIN v14.1](#rainv14.1)
        * [RAIN v14.2](#rainv14.2)
        * [RAIN v14.3](#rainv14.3)
    * [RAIN v15.0](#rainv15)
    * [RAIN v16.0](#rainv16)
    * [RAIN v17.0](#rainv17)
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
| ResNet-50 + RAINv1 |         &#9745;      |  &#9745;  | &#9745; |     &#9974;       |     &#9974;       |
| ResNet-50 + RAINv2 |         &#9745;      |  &#9974;  | &#9745; |     &#9974;       |     &#9974;       |
| ResNet-50 + RAINv3 |         &#9745;      |  &#9974;  | &#9745; |     &#9974;       |     &#9974;       |

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


<a name="rainv2"/>
RAIN Layer v2.0
---------------
The primary concept utilized in the formulation of this version is: The use of only a sampling parameter that will specify the rate at which to sample input videos starting at the first frame. This should give us a baseline to understand the preference of networks designed to handle rate to either stick with uniform sampling or choose an alternate path.

![RAINv2Paramnw] (/images/Paramnw.pdf)
![RAINv2extractlayer] (/images/extractlayer2.pdf)


<a name="rainv3"/>
RAIN Layer v3.0
---------------
The primary concept utilized in the formulation of this version is: The use of only a phase offset to select the start of the RAIN output clip then uniformly sample to the end of the input data. Once again, this model is used to study the impact of offset while holding the sampling parameter constant to understand the networks inner working.

![RAINv3Paramnw] (/images/Paramnw.pdf)
![RAINv3extractlayer] (/images/extractlayer3.pdf)


<a name="rainv4"/>
RAIN Layer v4.0
---------------
The primary concept utilized in the formulation of this version is: The use of directly learning indices of frames to sample from the input video from the output of the FC2 layer.  RAINv4 models will be trained with these indices sorted and unsorted. This model is an exploration into the importance of ordering for action videos.

![RAINv4Paramnw] (/images/Paramnw4.pdf)
![RAINv4extractlayer_sorted] (/images/extractlayer4_sorted.pdf)
![RAINv4extractlayer_unsorted] (/images/extractlayer4_unsorted.pdf)


<a name="rainv5"/>
RAIN Layer v5.0
---------------
The primary concept utilized in the formulation of this version is: The use of only a phase start offset and a phase end offset that directly specify the start and end frames of the output clip of the RAIN layer. Abandoning the original signal processing formulation, we instead seek to use offset anchors to determine the exact part of the video to extract without paying attention to rate control explicitly.

![RAINv5Paramnw] (/images/Paramnw.pdf)
![RAINv5extractlayer] (/images/extractlayer5.pdf)



<a name="rainv6"/>
RAIN Layer v6.0
---------------
The primary concept utilized in the formulation of this version is: Allow the output of the extract layer to be parameters used to modify the input video by gradually slowing down the video from the default rate to some learned rate until it reaches the main action and then speeding it back up to the default rate. The beginning and end points of this sampling will still be the beginning and end of the input video.  Phi represents the offset for the frame at which the maximum slow down will be reached.

![RAINv6Paramnw] (/images/Paramnw.pdf)
![RAINv6extractlayer] (/images/extractlayer6.pdf)


<a name="rainv7"/>
RAIN Layer v7.0
---------------
The primary concept utilized in the formulation of this version is: Increase the number of frames output by the RAIN layer. Models will be trained with 75 and 100 frames.
Given multiple cycles of input being observed in baseline models, this model is used to study the impact of allowing the network to see a single cycle (if v1 or previous models capture a single cycle) for an extended period of time.

![RAINv7Paramnw] (/images/Paramnw.pdf)
![RAINv7extractlayer] (/images/extractlayer7.pdf)



<a name="rainv8"/>
RAIN Layer v8.0
---------------
The primary concept utilized in the formulation of this version is: Change the activation of the FC2 layer from sigmoid to ReLu. In an effort to accomodate more open ended estimations of sampling and offset from v1, we chose to allow ReLU activations so that both slow down and speed up are afforded to the RAIN layer.

![RAINv8Paramnw] (/images/Paramnw.pdf)
![RAINv8extractlayer] (/images/extractlayer.pdf)


<a name="rainv9"/>
RAIN Layer v9.0
---------------
The primary concept utilized in the formulation of this version is: Change the activation of the FC2 layer from sigmoid to ReLu and increase the number of output frames to 75 and 100.

![RAINv9Paramnw] (/images/Paramnw.pdf)
![RAINv9extractlayer] (/images/extractlayer7.pdf)



<a name="rainv10"/>
RAIN Layer v10.0
---------------
The primary concept utilized in the formulation of this version is: Allow the sampling parameter to first sample frames from the beginning of the input data then apply the phase offset parameter to select the start frame of the output. Sample L frames from the output. This is built on the hypothesis that if sampling seeks to look at the entire video in a variable rate form then we can use offset to skim the portions that we explicitly desire. In this state, we are not guarenteed a single cycle.

![RAINv10Paramnw] (/images/Paramnw.pdf)
![RAINv10extractlayer] (/images/extractlayer10.pdf)


<a name="rainv11"/>
RAIN Layer v11.0
---------------
The primary concept utilized in the formulation of this version is: Seeing that v3 fluctuates the learned value of phi from zero to one, it may be possible to apply a relu activation to fc2 instead of the sigmoid which can bias the values to zero or one.

![RAINv11Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv11extractlayer Placeholder] (/images/extractlayer3.pdf)



<a name="rainv12"/>
RAIN Layer v12.0
---------------
The primary concept utilized in the formulation of this version is: Remove initialization in parameterization network of v3. Currently the convolutional layers in the parameterization network are initialized to pretrained resnet weights, this may be what causes the network to bias the values of 0 and 1 for phi.

![RAINv12Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv12extractlayer Placeholder] (/images/extractlayer3.pdf)

<a name="rainv12.1"/>
RAIN Layer v12.1
---------------
The primary concept utilized in the formulation of this version is: Add an ReLu activation after Conv1 in the parameterization network.

![RAINv12.1Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv12.1extractlayer Placeholder] (/images/extractlayer3.pdf)



<a name="rainv13"/>
RAIN Layer v13.0
---------------
The primary concept utilized in the formulation of this version is: To add an LSTM into the parameterization network of v3. The network may not be able to learn the temporal information of the input properly in its current state which could be improved upon with an LSTM.

![RAINv13Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv13extractlayer Placeholder] (/images/extractlayer3.pdf)



<a name="rainv14"/>
RAIN Layer v14.0
---------------
The primary concept utilized in the formulation of this version is: The v1 definition of the RAIN layer causes the output video to return the first frame of the input if alpha equals zero no matter the value of phi. This model avoid this issue by first calculating alpha and then adding the offset. 

![RAINv14Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv14extractlayer Placeholder] (/images/extractlayer.pdf)

<a name="rainv14.1"/>
RAIN Layer v14.1
---------------
The primary concept utilized in the formulation of this version is: Remove the ReLu after Conv1 to compare with v14.0 to determine the impact of the ReLu.

![RAINv14Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv14extractlayer Placeholder] (/images/extractlayer.pdf)

<a name="rainv14.2"/>
RAIN Layer v14.2
---------------
The primary concept utilized in the formulation of this version is: Remove the parameterization network initializations to determine their impact.

![RAINv14Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv14extractlayer Placeholder] (/images/extractlayer.pdf)

<a name="rainv14.3"/>
RAIN Layer v14.3
---------------
The primary concept utilized in the formulation of this version is: Remove the activation function on FC1 to determine its impact.

![RAINv14Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv14extractlayer Placeholder] (/images/extractlayer.pdf)


<a name="rainv15"/>
RAIN Layer v15.0
---------------
The primary concept utilized in the formulation of this version is: To train v14 alpha and phi using different activation functions for FC2. The value phi has a linear relationship with the chosen output indices while alphas relationship is nonlinear. Thus FC2 is split into FC2a for alpha with a sigmoid activation function and FC2b for phi with an ReLu activation function.
![RAINv15Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv15extractlayer Placeholder] (/images/extractlayer.pdf)

<a name="rainv16"/>
RAIN Layer v16.0
---------------
The primary concept utilized in the formulation of this version is: This model attempts to fix the issue that could arrise from combining dimension K and V immediately after the convolution layers by first passing dimension K through FC1 then reshaping the output and passing dimension V through FC2 before reducing the output to 2 dimensions in FC3. This model only modifies the parameterization network, the extraction layer remains the same as v14.
![RAINv16Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv16extractlayer Placeholder] (/images/extractlayer.pdf)

<a name="rainv17"/>
RAIN Layer v17.0
---------------
The primary concept utilized in the formulation of this version is: The filter blocks relating to each pixel may contain the information needed to learn alpha and phi.  Allow output of FC2 to be the filter dimension, V, x 2. This model only modifies the parameterization network, the extraction layer remains the same as v14.
![RAINv17Paramnw Placeholder] (/images/Paramnw.pdf)
![RAINv17extractlayer Placeholder] (/images/extractlayer.pdf)




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
| ResNet50 + LSTM          |              43.01%                 |                  41.33%                     |

    HMDB51 Original RAIN Layer Experiments
|       Experiments        | Median of Extract Layer MRA  | Mean of Extract Layer MRA  | Max of Extract Layer MRA  | 
|:------------------------:|:---------------------------: |:-------------------------: |:------------------------: |
| ResNet50 + RAINv1 + LSTM |             34.44%           |             **44.44**%         |          32.94%           |
| ResNet50 + RAINv2 + LSTM |             **48.50**%           |             34.25%         |          **43.99**%           |
| ResNet50 + RAINv3 + LSTM |             **44.51**%           |             42.48%         |          --.--%           |  
| ResNet50 + RAINv4 No Sort + LSTM |             40.00%*           |             --.--%         |          --.--%           |
| ResNet50 + RAINv11 + LSTM |             **45.10**%           |             **46.54**%         |          --.--%           | 
| ResNet50 + RAINv12 + LSTM |             42.22%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv12.1 + LSTM |             **44.05**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv14 + LSTM |             39.54%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv14.1 + LSTM |             36.34%*           |             --.--%         |          --.--%           |
| ResNet50 + RAINv14.2 + LSTM |             40.59%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv14.3 + LSTM |             **44.05**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv15 + LSTM |             34.77%*           |             --.--%         |          --.--%           |
| ResNet50 + RAINv16 + LSTM |             35.95%*           |             --.--%         |          --.--%           |
| ResNet50 + RAINv17 + LSTM |             33.92%*           |             --.--%         |          --.--%           |
| ResNet50 + RAINv18 + LSTM |             35.49%*           |             37.84%*         |          --.--%           |
| ResNet50 + RAINv19 + LSTM |             **43.86**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv20 + LSTM |             --.--%           |             --.--%         |          **44.58**%           |
| ResNet50 + RAINv21 + LSTM |             **43.14**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv22 + LSTM |             **43.33**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv23 + LSTM |             **44.64**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv23_LSTM + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv23.1 + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv24 + LSTM |             41.96%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv24_LSTM + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv25 + LSTM |             41.11%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv26 + LSTM |             **43.76**%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv27 + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv28 + LSTM |             --.--%           |             --.--%         |          --.--%           |
| ResNet50 + RAINv29 + LSTM |             --.--%           |             --.--%         |          --.--%           |
*-models were not trained to completion due to having already learned either one or zero for phi and alpha.


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

### Experiment 1 NOTES:

#### RAINv1

Median Output: phi ~ 1e^-2, alpha ~ 0.0 This model learned to supply a constant value of nearly 0 to alpha and a value of 0.02 to phi no matter the input video. Since alpha is zero, the output video calculation will consistently result in returning the first frame of the input video repeated to 50 frames regardless of the value of phi.  

Mean Output: phi = 1.0, alpha = 1.0  This model learned to supply a constant value of 1 to phi and alpha no matter the input video. This results in the output video to consist of only the last L frames of the input video. Phi is cutting away all but the last L frames of the video rendering alpha useless. However, this improves the classification accuracy by 1% over the baseline suggesting that it is valid to reduce the input video to contain only a few actions as opposed to sampling the entire input.  This could indicate an increase in performance if the RAIN layer were to be able to detect a single action.

Max Output: phi ~ 1e^-1, alpha ~ 0.0 This model learned to supply a constant value of nearly 0 to alpha and a value of 0.002 to phi no matter the input video. Since alpha is zero, the output video calculation will consistently result in returning the first frame of the input video repeated to 50 frames regardless of the value of phi.  

![ RAINv1 Input Median Mean Max](images/Combined_RAINv1.gif)

Input video into RAIN layer (HMDB51 video looped to reach 250 frames), RAINv1 Median output given input video (50 frames), RAINv1 Mean output given input video (50 frames), RAINv1 Max output given input video (50 frames)



#### RAINv2

Median Output: alpha = 1.0

Mean Output: alpha ~ 0.0

Max Output: alpha = 1.0

This model starts at the first frame of the input video and samples according to alpha. Thus when alpha is ~ 0.0, the output of the RAIN layer is only the first frame repeated L (50) times. When alpha = 1.0, the input video is uniformly sampled to L frames.

![ RAINv2 Input Median Mean Max](images/Combined_RAINv2.gif)

The input into RAINv2 and the output of v2 median, mean, and max respectively. For v2 median and max the output video consisted of 50 uniformly sampled frames from the input video.  For v2 mean, alpha = 0.0 so the output video consisted of the first frame of the input video repeated 50 times.

#### RAINv3

Median Output: phi ~ 0.0

Mean Output: phi = 1.0


This model starts at the phi frame of the input video and uniformly sampled to the end of the video. Thus when phi is ~ 0.0, the output of the RAIN layer is the input video uniformly sampled to L frames. When phi = 1.0, the output consists of the last L (50) frames of the input video.


![ RAINv3 Input Median Mean](images/Combined_RAINv3.gif)

The input into RAINv3 and the output of v3 median and mean respectively. The output video for v3 median consists of the input video uniformly sampled to 50 frames since phi = 0.0. Phi = 1.0 for v3 mean so the output video consists of the last 50 frames of the input video.


#### RAINv4 No Sort

Median Output: FC2 output ~ 0.999 or <1e^-4 indicating that the 50 frames being selected are either the first or the last frame of the input video. 

Instead of taking 2 parameters, alpha and phi, into the RAIN layer, this model recieves L parameters each directly indicating and index of the input video to append to the output video. The only values there were output were either nearly 0.0 or 1.0, thus the output video consisted of randomly alternating between the first and last frame of the input video.



#### RAINv11

Median Output: phi = 0.0

Mean Output: phi = 0.0

Since this is an alternate version of v3, phi = 0.0 represents selecting the first frame and then uniformly sampling to the end of the input video.




#### RAINv12

Median Output: phi = 1.0


RAINv12.1:

Median Output: phi = 1.0

Since these are alternate versions of v3, phi = 1.0 represents selecting the last L (50) frames of the input video.



#### RAINv14

Median Output: phi = 1.0, alpha = 1.0 




RAINv14.1:

Median Output: phi ~ 0.0, alpha ~ 0.0


RAINv14.2:

Median Output: phi = 1.0, alpha = 1.0




RAINv14.3:

Median Output: phi = 0.0, alpha = 1.0


These models have been updated such that alpha = 0.0 will not break down to selecting only the first frame of the input video. Instead alpha = 0.0 will now select a single frame located at phi and repeat it L (50) times. Phi = 0.0 will select the first frame of the input video and sample according to alpha, phi = 1.0 will select the last L (50) frames of the input video regardless of alpha.

![ RAINv14.0 Input and Median ](images/Combined_RAINv14.gif)

The input into RAINv14.0 and the output of v14.0 median. Since phi = 1.0, the output video was the last 50 frames of the input video.


#### RAINv15

Median Output: phi > 400, alpha = 1.0

This model is an alternate version of v14 where FC2 has been split into FC2a and FC2b with sigmoid and ReLu activations for alpha and phi respectively. Since phi is now unbounded, any value above N (250) frames will get reduced to N. Since all values of phi were above 250, the output of the RAIN layer was the last frame of the input video repeated L (50) times.


#### RAINv16

Median Output: phi ~ 1.0, alpha ~ 0.0

Since the extraction layer of this model is the same as v14, alpha = 0.0 will now select a single frame located at phi and repeat it L (50) times. Phi = 0.0 will select the first frame of the input video and sample according to alpha, phi = 1.0 will select the last L (50) frames of the input video.



#### RAINv17

Median Output: phi ~ 1e^-5, alpha ~ 1e^-4

Since the extraction layer of this model is the same as v14, alpha = 0.0 will now select a single frame located at phi and repeat it L (50) times. Phi = 0.0 will select the first frame of the input video and sample according to alpha, phi = 1.0 will select the last L (50) frames of the input video.


#### RAINv18

Median Output: phi = 0.0, alpha = 0.0

Mean Output: phi = 0.0, alpha = 0.0


#### RAINv19

Median Output: phi = 0.0, alpha = 0.0 


#### RAINv20

Max Output: phi = 1.0


#### RAINv21

Median Output: phi = 0.1104, alpha = 1.0 (-0.0303)


#### RAINv22

Median Output: phi = 1.0 (-0.00795), alpha = 0.9395


#### RAINv23

Median Output: alpha ~ 0.462

RAINv23 LSTM

Median Output: alpha = 1.0 (before negative exponent alpha = 0.0)

RAINv23.1

Median Output: alpha ~ IN PROGRESS (currently alpha = 1.0 (-0.0803))


#### RAINv24

Median Output: alpha = 1.0 (-0.0280)


RAINv24 LSTM

Median Output: alpha = 1.0 (before negative exponent alpha = 0.0)



#### RAINv25

Median Output: phi = 1.0 (-0.0272)


#### RAINv26

Median Output: phi ~ 0.6075


#### RAINv27

Median Output: phi = 1.0 (-0.104), alpha = 0.0491


#### RAINv28

Median Output: phi = 1.0 (-0.0132), alpha = 1.0 (-0.00335)


#### RAINv29

Median Output: phi = IN PROGRESS (currently phi = 0.646), alpha = IN PROGRESS (currently alpha = 1.0 (-0.0917)


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

* V11:
    * Idea: (V3 + ReLu) Seeing that v3 fluctuates the learned value of phi from zero to one, it may be possible to apply a relu activation to fc2 instead of the sigmoid which can bias the values to zero or one.
    * Median of Extract Layer

* V12:
    * Idea: (V3 + random initializations) Currently the convolutional layers in the parameterization network are initialized to pretrained resnet weights, this may be what causes the network to bias the values of 0 and 1 for phi.
    * Median of Extract Layer

* V12.1:
    * Idea: (V12 + ReLu after Conv1 in Parameterization Network) Add an ReLu activation after Conv1 in the parameterization network.
    * Median of Extract Layer

* V13:
    * Idea: (V3 + LSTM) The network may not be able to learn the temporal information of the input properly in its current state which could be improved upon with an LSTM.
    * Median of Extract Layer

* V14:
    * Idea: (Alternate V1) The v1 definition of the RAIN layer causes the output video to return the first frame of the input if alpha equals zero no matter the value of phi. This model avoid this issue by first calculating alpha and then adding the offset. 
    * Median of Extract Layer

* V14.1:
    * Idea: (V14 without ReLu after Conv1 in Parameteriazation Network) Remove the ReLu after Conv1 to compare with v14.0 to determine the impact of the ReLu.
    * Median of Extract Layer

* V14.2:
    * Idea: (V14.1 with random initializations) Remove the parameterization network initializations to determine their impact.
    * Median of Extract Layer

* V14.3:
    * Idea: (V14.2 with FC1 linear activation) Remove the activation function on FC1 to determine its impact.
    * Median of Extract Layer
    
* V15:
    * Idea: (V14 ReLu activation for phi, sigmoid activation for alpha, random initializations) The value phi has a linear relationship with the chosen output indices while alphas relationship is nonlinear. Thus FC2 is split into FC2a for alpha with a sigmoid activation function and FC2b for phi with an ReLu activation function.
    * Median of Extract Layer
     
* V16:
    * Idea: (Add FC3 layer with V dims)  This model attempts to fix the issue that could arrise from combining dimension K and V immediately after the convolution layers by first passing dimension K through FC1 then reshaping the output and passing dimension V through FC2 before reducing the output to 2 dimensions in FC3.
    * Median of Extract Layer
     
* V17:
    * Idea: (Pass K dims through FC1 then reshape to V dims through FC2) The filter blocks relating to each pixel may contain the information needed to learn alpha and phi.  Allow output of FC2 to be the filter dimension, V, x 2. 
    * Median of Extract Layer
     
* V18:
    * Idea: (Parameterization Network at the end, phi and alpha) Since the parameterization network has been located at the top of the model, the gradients may have had less of an impact by the time they reached the parameterization network, resulting in alpha and phi just falling to 0.0 or 1.0.  Moving the parameterization network to the end could allow the gradients to have a greater impact on alpha and phi values. 
    * Mean of Extract Layer
    * Median of Extract Layer
 
* V19:
    * Idea: (V18 + negative exponent) The tanh used in v18 to force the values of alpha and phi may have been to abrupt causing them to fall to 0.0 or 1.0.  A negative exponent has a more gradual curve which could allow alpha and phi to learn values in between 0 and 1.
    * Median of Extract Layer

* V20:
    * Idea: (V18 phi only, take max of block softmax) Instead of allowing phi to be any frames, only allow it to choose from the J blocks.  
    * Max of Extract Layer

* V21:
    * Idea: (alpha and phi as variables, based off of v14) Initialize alpha and phi as variables and let them learn based solely off of backpropagation. Inputs to extraction layer come from the beginning of the model, similar to v14.
    * Variables learned directly

* V22:
    * Idea: (alpha and phi as variables, based off of v18) Initialize alpha and phi as variables and let them learn based solely off of backpropagation. Inputs to extraction layer come from the end of the model, similar to v18.
    * Variables learned directly

* V23:
    * Idea: (alpha as a variable, based off of v2 + v21) Initialize alpha as a variable and let it learn based solely off of backpropagation. Inputs to extraction layer come from the beginning of the model, similar to v2.
    * Variables learned directly
     
* V23.1:
    * Idea: (v23 + tanh) ReLu with a negative exponential function causes any negative value to default to 1.0.  Switching the activation function to tanh will allow negative values to remain valid
    * Variables learned directly
* V24:
    * Idea: (alpha as a variable, based off of v2 + v22) Initialize alpha as a variable and let it learn based solely off of backpropagation. Inputs to extraction layer come from the end of the model, similar to v22.
    * Variables learned directly

* V25:
    * Idea: (phi as a variable, based off of v3 + v21) Initialize phi as a variable and let it learn based solely off of backpropagation. Inputs to extraction layer come from the beginning of the model, similar to v21.
    * Variables learned directly

* V26:
    * Idea: (phi as a variable, based off of v3 + v22) Initialize phi as a variable and let it learn based solely off of backpropagation. Inputs to extraction layer come from the end of the model, similar to v22.
    * Variables learned directly
     
* V27:
    * Idea: (2 Step - offset then alpha, beginning of model) Initialize phi and alpha and then let them learn solely off of backpropagation. First apply the offset phi, then sample alpha from the remaining frames.
    * Variables learned directly

* V28:
    * Idea: (2 Step - offset then alpha, end of model) Initialize phi and alpha and then let them learn solely off of backpropagation. First apply the offset phi, then sample alpha from the remaining frames.
    * Variables learned directly

* V29:
    * Idea: (Define alpha on top, then phi on bottom) Initialize alpha at the beginning of the model and sample the input video, then pass this through the network until the end where phi gets initialized and offsets the video and samples to L frames.
    * Variables learned directly

* Alternate:
    * Pass the parameters through an LSTM before entering the RAIN layer.
