# Tensorflow Activity Recognition Framework

This python framework provides modular access to common activity recognition LRCN models for the uses of baseline comparisons with current state of the art and custom models.

This README will walk you through the process of installing dependencies, downloading and formatting the required datasets, testing the install of the framework, and expanding the framework to your own models and datasets.


## Table of Contents


* [Requirements](#requirements)
* [Introduction and Setup](#introduction-and-setup)
	*  [Using the Framework](#using-the-framework)
	*  [Framework File Structure](#framework-file-structure)
	*  [Generate HDF5 Files](#generate-hdf5-files)
	*  [Examples of Common Uses](#examples-of-common-uses)
* [Add Custom Components](#add-custom-components)
	* [Adding a Model](#adding-a-model)
	* [Adding a Dataset](#adding-a-dataset-currently-unavailable)
* [Results](#expected-results)
* [Version History](#version-history)

## Requirements

#### Python Dependencies:
* Tensorflow 1.2.1
* Numpy
* OpenCV
* Cudnn
* Cuda
* gflags
* h5py


## Introduction and Setup


### Implemented Models:

* ResNet50
* LRCN
* VGG16


### Implemented Datasets:

* HMDB51
* UCF101

### Future Models:
* TSN
* C3D


### Future Datasets:
* Kinetics


### Using the framework

From the root directory, the training and testing is done through train_test_model.py

The parameters to train are:

```
python  train_test_model.py \

--model         The model archetecture to be used (ResNet, VGG16, LRCN)

--dataset       The dataset to use (UCF101, HMDB51)

--numGpus       Number of GPUs to train on (Not yet implemented)

--train         1 or 0 whether this is a training or testing run

--load          1 or 0 whether to use the current trained checkpoints with the same experiment_name or to train from random initialized weights

--size          Size of the input frame (227 for LRCN, 224 for ResNet and VGG16)

--inputDims     Input dimensions (number of frames to pass into model)

--outputDims    Output dimensions(number of classes in dataset)

--seqLength     Sequence length to input into lstm (16 for LRCN, 50 for ResNet and VGG16)

--expName       Experiment name

--numVids       Number of videos to train or test on within the split (Uses the first numVids videos in testlist/trainlist.txt)

--lr            Initial learning rate

--wd            Weight decay value, defaults to 0.0

--nEpochs       Number of epochs to train over

--split         Dataset split to use

--baseDataPath  The path to where all datasets are stored (Ex. This directory should then contain "UCF101HDF5RGB/Split1/trainlist_[0]_.hdf5" and "HMDB51HDF5RGB/Split2/testlist_[0]_.hdf5")

--fName			Which dataset list to use (trainlist, testlist, vallist)
```

Ex. Train ResNet on HMDB51 using 1 GPU

```
python train_test_model.py  --model resnet  --dataset HMDB51  --numGpus 1  --train 1  --load 1  --size 224  --inputDims 25  --outputDims 51  --seqLength 25  --expName example_1  --numVids 3570  --lr 0.001  --wd 0.0  --nEpochs 30  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist
```

The parameters to test are:

```
python  train_test_model.py \

--model         The model archetecture to be used (ResNet, VGG16, LRCN)

--dataset       The dataset to use (UCF101, HMDB51)

--numGpus       Number of GPUs to train on (Not yet implemented)

--train         1 or 0 whether this is a training or testing run

--size          Size of the input frame (227 for LRCN, 224 for ResNet and VGG16)

--inputDims     Input dimensions (number of frames to pass into model)

--outputDims    Output dimensions(number of classes in dataset)

--seqLength     Sequence length to input into lstm (16 for LRCN, 50 for ResNet and VGG16)

--expName       Experiment name

--numVids       Number of videos to train or test on within the split (Uses the first numVids videos in testlist/trainlist.txt)

--split         Dataset split to use

--baseDataPath  The path to where all datasets are stored (Ex. This directory should then contain "UCF101HDF5RGB/Split1/trainlist_[0]_.hdf5" and "HMDB51HDF5RGB/Split2/testlist_[0]_.hdf5")

--fName			Which dataset list to use (trainlist, testlist, vallist)
```

Ex. Test LRCN on UCF101 split 1

```
python train_test_model.py  --model lrcn  --dataset UCF101  --train 0  --size 227  --inputDims 160  --outputDims 101  --seqLength 16  --expName example_2  --numVids 3783  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```

### Framework File Structure
```
/TensorflowCV_Framework
   train_test_model.py  
   load_dataset.py  
   logger.py
   layers_utils.py
   utils.py
   genHDF5.py

    /datasets
	    /dataset
            /classInd.txt
            /testlist01.txt
            /trainlist01.txt
            /vallist01.txt (Optional)

   /models
        /model
            model_class.py
            model_preprocessing.py

   /results  
        /model
            /experiment_name
                tensorflow_checkpoints

    /Logs
        /model
            /dataset
                /experiment_name
                    tensorboard_log

    /scripts
        /experiment_name
            pbsScript.pbs

```


train_test_model.py - Main program for training and testing models
load_dataset.py - Loads specified dataset using a given models preprocessing
genHDF5.py - Generates hdf5 files for given dataset
trainlist, testlist, vallist - Lists of videos for training testing and validation splits

models - Includes the model class and preprocessing required for that class
results - Saved model weights and checkpoints
Logs - Tensorboard logs









### Generate HDF5 files

In order to run any of this code, the datasets will need to be downloaded and formatted correctly.  Datasets are not included and must be downloaded and converted to HDF5 format.

The datasets incorporated into this framework are UCF101 and HMDB51, however, they do not exist in this repository and must be downloaded.  These datasets have scripts associated with them to easily format them correctly.  Custom datasets can also be added as described in the next section.

The current genHDF5 program will make use of a pool of 30 processors to generate the files in parallel.

```
python scripts/dataset_formatting/genHDF5.py \

--vidsFile		File containing the paths to the videos
--baseDataPath	Original dataset directory
--baseDestPath	Destination directory
--chunk			Number of videos to aggregate together into a single HDF5 file (Ex. 100)
--fName			Name of HDF5 file, prefix (testlist, trainlist, vallist)
```



### Examples of Common Uses

#### Testing using existing models

Must download the saved checkpoints for the trained models with experiment names: resnet_test, vgg16_test, lrcn_test

Test ResNet on HMDB51:
```
python train_test_model.py  --model resnet  --dataset HMDB51  --train 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName resnet_test  --numVids 1530  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```

Test VGG16 on HMDB51:
```
python train_test_model.py  --model vgg16  --dataset HMDB51  --train 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName vgg16_test  --numVids 1530  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```

Test LRCN on UCF101:
```
python train_test_model.py  --model lrcn  --dataset UCF101  --train 0  --size 227  --inputDims 160  --outputDims 101  --seqLength 16  --expName lrcn_test  --numVids 3783  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```


#### Training models from scratch
Train ResNet on HMDB51:
```
python train_test_model.py  --model resnet  --dataset HMDB51  --train 1  --load 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName resnet_train  --numVids 3570  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist  --lr 0.001 --wd 0.0  --nEpochs 30
```

Train VGG16 on HMDB51:
```
python train_test_model.py  --model vgg16  --dataset HMDB51  --train 1  --load 0  --size 224  --inputDims 25  --outputDims 51  --seqLength 25  --expName vgg16_train  --numVids 3570  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist  --lr 0.001 --wd 0.0  --nEpochs 30
```

Train LRCN on UCF101:
```
python train_test_model.py  --model lrcn  --dataset UCF101  --train 1  --load 0  --size 227  --inputDims 160  --outputDims 101  --seqLength 16  --expName lrcn_train  --numVids 9537  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist  --lr 0.001 --wd 0.0  --nEpochs 30
```







## Add Custom Components

### Adding a model


##### Step 1: Create Model Directory Structure

Create the directories:
```
\models\modelName

\results\modelName
```

Add the empty file:
```
\models\modelName\__init__.py
```

##### Step 2: Add Model Class

Create the file model file:
```
\models\modelName\modelName_model.py
```

File Structure:
```
from model_preprocessing import preprocess


class ModelName():
	def __init__(self, verbose):
        self.verbose=verbose
        self.name = 'modelName'

    def inference(self, inputs, labels, isTraining, inputDims, outputDims, seqLength, scope, weight_decay=0.0, return_layer='logits', cpuId=0):
    	return

    def preprocess(self, index, data, labels, size, isTraining):
        return preprocess(index, data,labels, size, isTraining)

    def loss(self, logits, labels):
        return

```

#### Step 3: Add Model Preprocessing
Create the file model file:
```
\models\modelName\modelName_preprocessing.py
```

File Structure:
```
def preprocess(index, Data, labels, size, isTraining):
    ##Implementation
    return Data
```



#### Step 4: Import into train_test_model.py

At the top of the train_test_model.py file add:

```
from models.model.name_model import ModelClass

```

In the main of train_test_model.py add:

```
elif modelName == 'name_model':
    model = ModelClass()

```


### Adding a dataset
Adding a new dataset requires that the videos are structured in the expected directory structure and that there exist the necessary supplemental text files for the HDF5 file generation like the vidsFile




## Expected Results

### Accuracies of Models
The install of this framework can be tested by comparing the output with these expected testing results of the various models trained on the datasets.

|  Model Architecture  |      Dataset      |  Accepted Accuracy |  Framework Testing Accuracy |  
|----------|:----------:|:------:| :----:|
| LRCN |  UCF101 | 71.12% | 71.72% |
| ResNet |   HMDB51   |  43.90%  |  44.97% |
| VGG16 | HMDB51 |  --   |  28.10% |











## Version History


### Current Version: 1.0

#### Version 1.0
Initial release. Using pre generated HDF5 files, test LRCN model on UCF101 dataset and train ResNet and VGG16 models on HMDB51 dataset.  Tensorboard supported, single processor and single GPU implementation with the ability to cancel and resume training every 50 steps.  Documentation includes basic overview and example of training and testing commands.

Future features:

* Generate HDF5 files for new datasets
* Train and test ResNet and VGG16 on UCF101
* Implement parallel processing for data loading
* Support training on multiple GPUs
* Implement TSN and C3D models
* Implement Kinetics Database
* Expand documentation
