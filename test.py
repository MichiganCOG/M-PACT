# Basic imports
import os
import time
import argparse
import tensorflow      as tf
import numpy           as np
import multiprocessing as mp

# Tensorflow ops imports
from tensorflow.python.ops      import clip_ops
from tensorflow.python.ops      import init_ops
from tensorflow.python.ops      import control_flow_ops
from tensorflow.python.ops      import variable_scope as vs
from tensorflow.python.ops      import variables as vars_
from tensorflow.python.training import queue_runner_impl

# Custom imports
from models                       import *
from utils                        import initialize_from_dict, save_checkpoint, load_checkpoint, make_dir, Metrics
from Queue                        import Queue
from utils.logger                 import Logger
from random                       import shuffle
from utils.load_dataset_tfrecords import load_dataset


parser = argparse.ArgumentParser()

# Model parameters

parser.add_argument('--model', action= 'store', required=True,
        help= 'Model architecture (c3d, lrcn, tsn, vgg16, resnet)')

parser.add_argument('--inputDims', action='store', required=True, type=int,
        help = 'Input Dimensions (Number of frames to pass as input to the model)')

parser.add_argument('--outputDims', action='store', required=True, type=int,
        help = 'Output Dimensions (Number of classes in dataset)')

parser.add_argument('--seqLength', action='store', required=True, type=int,
        help = 'Number of output frames expected from model')

parser.add_argument('--modelAlpha', action='store', type=float, default=1.,
        help = 'Resampling factor for constant value resampling and alpha initialization')

parser.add_argument('--inputAlpha', action='store', type=float, default=1.,
        help = 'Resampling factor for constant value resampling of input video, used mainly for testing models.')

parser.add_argument('--dropoutRate', action='store', type=float, default=0.5,
        help = 'Value indicating proability of keeping inputs of the model\'s dropout layers.')

parser.add_argument('--freeze', action='store', type=int, default=0,
        help = 'Freeze weights during training of any layers within the model that have the option set. (default False)')

# Optimization parameters

parser.add_argument('--lr', action='store', type=float, default=0.001,
        help = 'Learning Rate')

parser.add_argument('--wd', action='store', type=float, default=0.0,
        help = 'Weight Decay')

parser.add_argument('--lossType', action='store', default='full_loss',
        help = 'String defining loss type associated with chosen model.')

parser.add_argument('--returnLayer', nargs='+',type=str, default=['logits'],
        help = 'Which model layers to be returned by the models\' inference during testing.')

parser.add_argument('--optChoice', action='store', default='default',
        help = 'String indicating optimizer choice')

parser.add_argument('--gradClipValue', action='store', type=float, default=5.0,
        help = 'Value of normalized gradient at which to clip.')

parser.add_argument('--lrboundary', nargs='+',type=int, default=[0],
        help = 'List of boundary epochs at which lr will be updated')

parser.add_argument('--lrvalues', nargs='+',type=float, default=[1.0],
        help = 'List of lr multiplier values, length of list must equal lrboundary')

# Experiment parameters

parser.add_argument('--dataset', action= 'store', required=True,
        help= 'Dataset (UCF101, HMDB51)')

parser.add_argument('--loadedDataset', action= 'store', default='HMDB51',
        help= 'Dataset (UCF101, HMDB51)')

parser.add_argument('--numGpus', action= 'store', type=int, default=1,
        help = 'Number of Gpus used for calculation')

parser.add_argument('--gpuList', nargs='+',type=str, default=[],
        help = 'List of GPU IDs to be used')

parser.add_argument('--train', action= 'store', type=int, default=0,
        help = 'Binary value to indicate training or evaluation instance')

parser.add_argument('--load', action='store', type=int, default=0,
        help = 'Whether you want to load a saved model to train from scratch.')

parser.add_argument('--loadedCheckpoint', action='store', type=int, default=-1,
        help = 'Specify the step of the saved model checkpoint that will be loaded for testing. Defaults to most recent checkpoint.')

parser.add_argument('--size', action='store', required=True, type=int,
        help = 'Input frame size')

parser.add_argument('--expName', action='store', required=True,
        help = 'Unique name of experiment being run')

parser.add_argument('--numVids', action='store', required=True, type=int,
        help = 'Number of videos to be used for training')

parser.add_argument('--nEpochs', action='store', type=int, default=1,
        help = 'Number of Epochs')

parser.add_argument('--split', action='store', type=int, default=1,
        help = 'Dataset split to use')

parser.add_argument('--baseDataPath', action='store', default='/z/dat',
        help = 'Path to datasets')

parser.add_argument('--fName', action='store', required=True,
        help = 'Which dataset list to use (trainlist, testlist, vallist)')

parser.add_argument('--saveFreq', action='store', type=int, default=1,
        help = 'Frequency in epochs to save model checkpoints')

parser.add_argument('--clipLength', action='store', type=int, default=-1,
        help = 'Length of clips to cut video into, -1 indicates using the entire video as one clip')

parser.add_argument('--videoOffset', action='store', default='none',
        help = '(none or random) indicating where to begin selecting video clips assuming clipOffset is none')

parser.add_argument('--clipOffset', action='store', default='none',
        help = '(none or random) indicating if clips are selected sequentially or randomly')

parser.add_argument('--clipOverlap', action='store', type=int, default=0,
        help = 'Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential')

parser.add_argument('--numClips', action='store', type=int, default=-1,
        help = 'Number of clips to break video into, -1 indicates breaking the video into the maximum number of clips based on clipLength, clipOverlap, and clipOffset')

parser.add_argument('--batchSize', action='store', type=int, default=1,
        help = 'Number of clips to load into the model each step.')

parser.add_argument('--metricsDir', action='store', type=str, default='default',
        help = 'Name of sub directory within experiment to store metrics. Unique directory names allow for parallel testing.')

parser.add_argument('--metricsMethod', action='store', default='avg_pooling',
        help = 'Which method to use to calculate accuracy metrics. (avg_pooling, last_frame, svm, svm_train or extract_features)')

parser.add_argument('--preprocMethod', action='store', default='default',
        help = 'Which preprocessing method to use (default, cvr, rr, sr are options for existing models)')

parser.add_argument('--randomInit', action='store', type=int, default=0,
        help = 'Randomly initialize model weights, not loading from any files (deafult False)')

parser.add_argument('--verbose', action='store', type=int, default=1,
        help = 'Boolean switch to display all print statements or not')


args = parser.parse_args()

if args.verbose:
    print "Setup of current experiments"
    print "\n############################"
    print args
    print "############################ \n"

# END IF

model_name = args.model

model = Models.create_model_object(modelName = model_name,
                                   inputAlpha = args.inputAlpha,
                                   modelAlpha = args.modelAlpha,
                                   clipLength = args.clipLength,
                                   numVids = args.numVids,
                                   numEpochs = args.nEpochs,
                                   batchSize = args.batchSize,
                                   numClips = args.numClips,
                                   numGpus = args.numGpus,
                                   train = args.train,
                                   expName = args.expName,
                                   outputDims = args.outputDims,
                                   inputDims = args.inputDims,
                                   preprocMethod = args.preprocMethod,
                                   dropoutRate = args.dropoutRate,
                                   freeze = args.freeze,
                                   verbose = args.verbose)


def test(model, input_dims, output_dims, seq_length, size, dataset, loaded_dataset, experiment_name, num_vids, split, base_data_path, f_name, load_model, return_layer, clip_length, video_offset, clip_offset, num_clips, clip_overlap, metrics_method, batch_size, metrics_dir, loaded_checkpoint, verbose, gpu_list, preproc_method, random_init):
    """
    Function used to test the performance and analyse a chosen model
    Args:
        :model:              tf-activity-recognition framework model object
        :input_dims:         Number of frames used in input
        :output_dims:        Integer number of classes in current dataset
        :seq_length:         Length of output sequence expected from LSTM
        :size:               List detailing height and width of frame
        :dataset:            Name of dataset being loaded
        :loaded_dataset:     Name of dataset which was used to train the current model
        :experiment_name:    Name of current experiment
        :num_vids:           Number of videos to be used for training
        :split:              Split of dataset being used
        :base_data_path:     Full path to root directory containing datasets
        :f_name:             Specific video directory within a chosen split of a dataset
        :load_model:         Boolean variable indicating whether to load from a checkpoint or not
        :return_layer:       Layer to return from the model, used to extract features
        :clip_length:        Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :video_offset:       String indicating where to begin selecting video clips (provided clipOffset is None)
        :clip_offset:        "none" or "random" indicating where to begin selecting video clips
        :num_clips:          Number of clips to break video into
        :clip_overlap:       Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential
        :metrics_method:     Which method to use to calculate accuracy metrics. ("default" or "svm")
        :batch_size:         Number of clips to load into the model each step.
        :metrics_dir:        Name of subdirectory within the experiment to store metrics. Unique directory names allow for parallel testing
        :loaded_checkpoint:  Specify the exact checkpoint of saved model to be loaded for further training/testing
        :verbose:            Boolean to indicate if all print statement should be procesed or not
        :gpu_list:           List of GPU IDs to be used
        :preproc_method:     The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing
        :random_init:        Randomly initialize model weights, not loading from any files (deafult False)

    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:

        # Initializers for checkpoint and global step variable
        ckpt    = None
        gs_init = 0

        ################################### Checkpoint loading block #######################################################

        # Load pre-trained/saved model
        if load_model:
            try:
                ckpt, gs_init, learning_rate_init = load_checkpoint(model.name, loaded_dataset, experiment_name, loaded_checkpoint)
                if verbose:
                    print 'A better checkpoint is found. The global_step value is: ' + str(gs_init)

            except:
                if verbose:
                    print "Failed loading checkpoint requested. Please check."
                exit()

            # END TRY

        else:
            ckpt = model.load_default_weights()

        # END IF

        ######################################################################################################################

        # Initialize model variables
        istraining       = False
        global_step      = tf.Variable(gs_init, name='global_step', trainable=False)
        number_of_videos = tf.Variable(num_vids, name='number_of_videos', trainable=False)
        video_step       = tf.Variable(1.0, name='video_step', trainable=False)

	# TF session setup
        config  = tf.ConfigProto(allow_soft_placement=True)
        sess    = tf.Session(config=config)
        init    = tf.global_variables_initializer()

        # Variables get randomly initialized into tf graph
        sess.run(init)

        data_path   = os.path.join(base_data_path, 'tfrecords_'+dataset, 'Split'+str(split), f_name)

        # Setting up tensors for models
        input_data_tensor, labels_tensor, names_tensor = load_dataset(model, 1, batch_size, output_dims, input_dims, seq_length, size, data_path, dataset, istraining, clip_length, video_offset, clip_offset, num_clips, clip_overlap, video_step, verbose)

        ######### GPU list check block ####################

        assert(len(gpu_list)<=1)

        if len(gpu_list) == 0:
            gpu_list = ['0'] # Default choice is ID = 0

        # END IF

        ###################################################

        ################################################## Setup TF graph block ######################################################

        # Model Inference
        with tf.device('/gpu:'+gpu_list[0]):
            logits = model.inference(input_data_tensor[0:batch_size,:,:,:,:],
                                     istraining,
                                     input_dims,
                                     output_dims,
                                     seq_length,
                                     scope,
                                     return_layer = return_layer)[0]

            # Logits
            softmax = tf.nn.softmax(logits)

        # END WITH

        ############################################################################################################################################


        ######################### Logger Setup block ######################################

        # Logger setup (Name format: Date, month, hour, minute and second, with a prefix of exp_test)
        log_name    = ("exp_test_%s_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                               dataset, experiment_name, metrics_method))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, metrics_dir, log_name))
        make_dir(os.path.join('results',model.name))
        make_dir(os.path.join('results',model.name, dataset))
        make_dir(os.path.join('results',model.name, dataset, experiment_name))
        make_dir(os.path.join('results',model.name, dataset, experiment_name, metrics_dir))

        ###################################################################################

        # TF session setup
        #sess    = tf.Session()
        init    = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord   = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        metrics = Metrics( output_dims, seq_length, curr_logger, metrics_method, istraining, model.name, experiment_name, dataset, metrics_dir, verbose=verbose)

        # Variables get randomly initialized into tf graph
        sess.run(init)

        # Check that weights were loaded or random initializations are requested
        if ((ckpt == None) or (random_init)):
            print "Caution: Model weights are not being loaded, using random initialization."

        else:
            # Model variables initialized from previous saved models
            initialize_from_dict(sess, ckpt, model.name)

        # END IF

        del ckpt

        acc               = 0
        count             = 0
        videos_loaded     = 0
        previous_vid_name = ''
        total_pred        = []

        if verbose:
            print "Begin Testing"

        # END IF

        ########################################## Testing loop block ################################################################

        while videos_loaded <= num_vids:
            output_predictions, labels, names = sess.run([softmax, labels_tensor, names_tensor])

                # END IF

            for batch_idx in range(len(names)):
                vid_name = names[batch_idx]
                if vid_name != previous_vid_name:
                    previous_vid_name = vid_name
                    videos_loaded += 1
                    if verbose:
                        print "Number of videos loaded: ", videos_loaded


                # Extract remaining clips from currently loaded video, once it finishes exit while loop
                if videos_loaded > num_vids:
                    break

                count += 1
                metrics.log_prediction(labels[batch_idx][0], output_predictions[batch_idx], vid_name, count)

            # END IF

        # END WHILE

        #########################################################################################################################################################

    # END WITH

    coord.request_stop()
    coord.join(threads)

    total_accuracy = metrics.total_classification()
    total_pred = metrics.get_predictions_array()

    if verbose:
        print "Total accuracy : ", total_accuracy
        print total_pred

    # Save results in numpy format
    np.save(os.path.join('results', model.name, dataset, preproc_method, experiment_name, metrics_dir, 'test_predictions_'+dataset+"_"+metrics_method+'.npy'), np.array(total_pred))


if __name__=="__main__":
    if not args.train:
        test(   model             = model,
                input_dims        = args.inputDims,
                output_dims       = args.outputDims,
                seq_length        = args.seqLength,
                size              = [args.size, args.size],
                dataset           = args.dataset,
                loaded_dataset    = args.loadedDataset,
                experiment_name   = args.expName,
                num_vids          = args.numVids,
                split             = args.split,
                base_data_path    = args.baseDataPath,
                f_name            = args.fName,
                load_model        = args.load,
                return_layer      = args.returnLayer,
                clip_length       = args.clipLength,
                video_offset      = args.videoOffset,
                clip_offset       = args.clipOffset,
                num_clips         = args.numClips,
                clip_overlap      = args.clipOverlap,
                metrics_method    = args.metricsMethod,
                batch_size        = args.batchSize,
                metrics_dir       = args.metricsDir,
                loaded_checkpoint = args.loadedCheckpoint,
                verbose           = args.verbose,
                gpu_list          = args.gpuList,
                preproc_method    = args.preprocMethod,
                random_init       = args.random_init)

    # END IF

    import pdb; pdb.set_trace()
