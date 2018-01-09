# Tensorflow Activity Recognition Framework

This python framework provides modular access to common activity recognition models for the use of baseline comparisons between the current state of the art and custom models.

This README will walk you through the process of installing dependencies, downloading and formatting the required datasets, testing the framework, and expanding the framework to your own models and datasets.


## Table of Contents


* [Requirements](#requirements)
* [Introduction and Setup](#introduction-and-setup)
	*  [Using the Framework](#using-the-framework)
	*  [Framework File Structure](#framework-file-structure)
	*  [Configuring Datasets](#configuring-datasets)
	*  [Examples of Common Uses](#examples-of-common-uses)
* [Add Custom Components](#add-custom-components)
	* [Adding a Model](#adding-a-model)
	* [Adding a Dataset](#adding-a-dataset)
* [Results](#expected-results)
* [Version History](#version-history)

## Requirements

#### Hardware and Software:
* Nvidia Graphics Card
* Ubuntu 16.04
* Python 2.7
* Cuda
* Cudnn
* gflags

#### Python Dependencies:
* Tensorflow 1.2.1
* Numpy
* OpenCV


## Introduction and Setup


### Implemented Models:

* ResNet50 + LSTM
* VGG16 + LSTM


### Implemented Datasets:

* HMDB51
* UCF101

### Future Models:
* TSN
* C3D
* ARTNet


### Future Datasets:
* Kinetics


### Using the framework

From the root directory, the training and testing is done through train_test_TFRecords_multigpu_model.py

The parameters to train are:

```
python  train_test_TFRecords_multigpu_model.py \

--model         The model archetecture to be used (ResNet, VGG16)

--dataset       The dataset to use for training (UCF101, HMDB51)

--numGpus       Number of GPUs to train on (Not yet implemented)

--train         1 or 0 whether this is a training or testing run

--load          1 or 0 whether to use the current trained checkpoints with the same experiment_name or to train from random initialized weights

--size          Size of the input frame (224 for ResNet and VGG16)

--inputDims     Input dimensions (number of frames to pass into model)

--outputDims    Output dimensions(number of classes in dataset)

--seqLength     Sequence length to input into lstm (50 for ResNet50 and VGG16)

--expName       Experiment name

--numVids       Number of videos to train or test on within the split (Uses the first numVids videos in testlist/trainlist.txt)

--valNumVids 	Number of videos to be used for validation

--lr            Initial learning rate

--wd            Weight decay value, defaults to 0.0

--nEpochs       Number of epochs to train over

--split         Dataset split to use

--baseDataPath  The path to where all datasets are stored (Ex. This directory should then contain tfrecords_HMDB51/Split1/trainlist/exampleVidName.tfrecords)

--fName			Which dataset list to use (trainlist, testlist, vallist)

--saveFreq		Frequency in epochs to save model checkpoints (default 1 aka every epoch)

--valFreq		Frequency in epochs to validate (default 3)

-returnLayer	List of strings indicating parameters within the model to be tracked during training (default ['logits'])
```

Ex. Train ResNet on HMDB51 using 4 GPUs

```
python train_test_TFRecords_multigpu_model.py  --model resnet  --dataset HMDB51  --numGpus 4  --train 1  --load 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName example_1  --numVids 3570  --lr 0.001  --wd 0.0  --nEpochs 30  --split 1  --baseDataPath /z/dat  --fName trainlist
```

The parameters to test are:

```
python  train_test_TFRecords_multigpu_model.py \

--model         The model archetecture to be used (ResNet50, VGG16)

--dataset       The dataset to use (UCF101, HMDB51)

--numGpus       Number of GPUs to train on (Not yet implemented)

--train         1 or 0 whether this is a training or testing run

--load          1 or 0 whether to use the current trained checkpoints with the same experiment_name or to train from random initialized weights

--size          Size of the input frame (224 for ResNet50 and VGG16)

--inputDims     Input dimensions (number of frames to pass into model)

--outputDims    Output dimensions(number of classes in dataset)

--seqLength     Sequence length to input into lstm (50 for ResNet50 and VGG16)

--expName       Experiment name

--numVids       Number of videos to train or test on within the split (Uses the first numVids videos in testlist/trainlist.txt)

--split         Dataset split to use

--baseDataPath  The path to where all datasets are stored (Ex. This directory should then contain tfrecords_HMDB51/Split1/trainlist/exampleVidName.tfrecords)

--fName			Which dataset list to use (trainlist, testlist, vallist)

--loadedDataset	Dataset that the model was trained on. This is to be used when testing a model on a different dataset than it was trained on.  
```

Ex. Test VGG16 on UCF101 split 1

```
python train_test_TFRecords_multigpu_model.py  --model vgg16  --dataset UCF101  --train 0  --load 1  --size 224  --inputDims 50  --outputDims 101  --seqLength 50  --expName example_2  --numVids 3783  --split 1  --baseDataPath /z/dat  --fName testlist
```

### Framework File Structure
```
/tf-activity-recognition-framework
   train_test_TFRecords_multigpu_model.py  
   load_dataset.py  
   logger.py
   layers_utils.py
   utils.py

    /datasets
	    /dataset_name
            /classInd.txt
            /testlist01.txt
            /trainlist01.txt
            /vallist01.txt (Optional)

   /models
        /model_name
            model_class.py
            model_preprocessing.py
            model_weights.npy (Optional)

   /results  
        /model_name
            /experiment_name
	            /checkpoints
	                checkpoint
	                checkpoint-100.npy
	                checkpoint-100.dat

    /Logs
        /model_name
            /dataset_name
                /experiment_name
                    tensorboard_log

    /scripts
        /PBS
	        /model_name
	            pbsScript.pbs

```


train_test_TFRecords_multigpu_model.py - Main program for training and testing models
load_dataset.py - Loads specified dataset using a given models preprocessing
trainlist, testlist, vallist - Lists of videos for training testing and validation splits

models - Includes the model class and video preprocessing required for that model
results - Saved model weights at specified checkpoints
Logs - Tensorboard logs









### Configuring Datasets


Currently integrated datasets: HMDB51, UCF101

In order to use this framework, the datasets will need to be downloaded and formatted correctly.  Datasets are not included and must be downloaded and converted to TFRecords format. Converting dataset videos into TFRecords binary files allows for optimized tensorflow data loading and processing.  

The currently methods to import and configure the datasets correctly are to either use the follow section [Adding a Dataset](#adding-a-dataset) to convert HMDB51 and UCF101 to tfrecords manually or to contact us directly for the datasets.





### Examples of Common Uses

#### Testing using existing models

Must download the saved checkpoints for the trained models with experiment names: tfrecords_resnet_HMDB51, tfrecords_vgg16_HMDB51

Test ResNet50 on HMDB51:
```
python train_test_TFRecords_multigpu_model.py  --model resnet  --dataset HMDB51  --train 0  --load 1  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName tfrecords_resnet_HMDB51  --numVids 1530  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```

Test VGG16 on HMDB51:
```
python train_test_TFRecords_multigpu_model.py  --model vgg16  --dataset HMDB51  --train 0  --load 1  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName tfrecords_vgg16_HMDB51  --numVids 1530  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName testlist
```



#### Training models from scratch
Train ResNet50 on HMDB51:
```
python train_test_TFRecords_multigpu_model.py  --model resnet  --dataset HMDB51  --numGpus 4  --train 1  --load 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName resnet_train  --numVids 3570  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist  --lr 0.001 --wd 0.0  --nEpochs 30
```

Train VGG16 on HMDB51:
```
python train_test_TFRecords_multigpu_model.py  --model vgg16  --dataset HMDB51  --numGpus 4  --train 1  --load 0  --size 224  --inputDims 50  --outputDims 51  --seqLength 50  --expName vgg16_train  --numVids 3570  --split 1  --baseDataPath /z/home/madantrg/Datasets  --fName trainlist  --lr 0.001 --wd 0.0  --nEpochs 30
```








## Add Custom Components

### Adding a model


##### Step 1: Create Model Directory Structure

Create the directory:
```
/models/modelName

```

Add the empty file:
```
/models/modelName/__init__.py
```

##### Step 2: Add Model Class

Create the model file:
```
/models/modelName/modelName_model.py
```

File Structure:
```
from layers_utils				   import *
from model_preprocessing_TFRecords import preprocess   as preprocess_tfrecords


class ModelName():
	def __init__(self, verbose):
        """
        Args:
            :verbose: Setting verbose command
        """
        self.verbose=verbose
        self.name = 'modelName'



	def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: String matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """
        
		# ADD MODEL HERE
		
    	return



    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)



    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        """
        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                    logits=logits)
        return cross_entropy_loss

```

#### Step 3: Add Model Preprocessing
Create the file model file:
```
/models/modelName/modelName_preprocessing.py
```

File Structure:
```
def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):

    # ADD PREPROCESSING HERE
    
    return input_data_tensor, labels_tensor
```



#### Step 4: Import into /models/\_\_init\_\_.py
At the top of the \_\_init\_\_.py file add:

```
from model.name_model import ModelClass

```

In the main of train_test_TFRecords_multigpu_model.py add:

```
elif modelName == 'name_model':
    model = ModelClass()

```


### Adding a dataset
Adding a new dataset requires that the videos converted to tfrecords and stored in a specific format. A tfrecord is simply a method of storing a video and information about the video in a binary file that is easily imported into tensorflow graphs.

Each tfrecord contains a dictionary with the following information from the original video:

* Label - Action class the video belongs to (type int64)
* Data - RGB or optical flow values for the entire video (type bytes)
* Frames - Total number of frames in the video (type int64)
* Height - Frame height in pixels (type int64)
* Width - Frame width in pixels (type int64)
* Channels - Number of channels (3 for RGB) (type int64)
* Name - Name of the video (type bytes)


The following code snipped is an example of how to convert a single video to tfrecords given the video data in the form of a numpy array.
```
def save_tfrecords(data, label, vidname, save_dir):
    filename = os.path.join(save_dir, vidname+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    features = {}
    features['Label'] = _int64(label)
    features['Data'] = _bytes(np.array(data).tostring())
    features['Frames'] = _int64(data.shape[0])
    features['Height'] = _int64(data.shape[1])
    features['Width'] = _int64(data.shape[2])
    features['Channels'] = _int64(data.shape[3])
    features['Name'] = _bytes(str(vidname))


    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    writer.close()

def _int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

```
A prerequisite to this is that the video must be passed in as a numpy or python array of floats/ints which can be done a number of ways. For example using OpenCV, matplotlib, or any other desired method.

The TFRecords for each dataset must be stored in a specific file structure, HMDB51 for example:
```
/tfrecords_HMDB51
	/Split1
		/trainlist
			vidName1.tfrecords
			vidName2.tfrecords
		/testlist
		/vallist
	/Split2
	/Split3
```


## Expected Results

### Accuracies of Models
The install of this framework can be tested by comparing the output with these expected testing results of the various models trained on the datasets.

|  Model Architecture  |      Dataset      |  Accepted Accuracy |  Framework Testing Accuracy at 30 Epochs |  Framework Testing Accuracy at 35 Epochs
|----------|:----------:|:------:| :----:| :----:|
| ResNet50 + LSTM |   HMDB51   |  43.90%  |  43.46% | 43.06% |
| VGG16 + LSTM | HMDB51 |  --   |  -- | -- |











## Version History


### Current Version: 2.0

#### Version 2.0
Implemented TFRecords based data loading to replace HDF5 files for increased performance.  Training has been updated to allow models to be trained on multiple GPUs concurrently.  Parallel data loading has been incorporated using TFRecords queues to allow maximized use of available GPUs.  The tensorflow saver checkpoints have been replaced with a custom version which reads and writes models weights directly to numpy arrays.  This will allow existing model weights from other sources to be more easily imported into this framework. Currently validation is not compatible with this tfrecords framework.

#### Version 1.0
Initial release. Using pre generated HDF5 files, test LRCN model on UCF101 dataset and train ResNet and VGG16 models on HMDB51 dataset.  Tensorboard supported, single processor and single GPU implementation with the ability to cancel and resume training every 50 steps.  Documentation includes basic overview and example of training and testing commands.

### Future features:

* Update validation to include tfrecords compatibility
* Implement TSN, C3D, and ARTNet
* Add training and testing on optical flow for the current datasets
* Implement Kinetics Database
* Expand documentation
