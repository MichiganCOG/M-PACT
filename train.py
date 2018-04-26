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
        help= 'Model architecture (c3d, tsn, i3d, resnet)')

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
        help = 'Freeze weights during training of any layers within the model that have the option set. (default 0)')

parser.add_argument('--loadWeights', action='store', type=str, default='default',
        help = 'String which can be used to specify the default weights to load.')

# Optimization parameters

parser.add_argument('--lr', action='store', type=float, default=0.001,
        help = 'Learning Rate')

parser.add_argument('--wd', action='store', type=float, default=0.0,
        help = 'Weight Decay')

parser.add_argument('--lossType', action='store', default='full_loss',
        help = 'String defining loss type associated with chosen model.')

parser.add_argument('--returnLayer', nargs='+',type=str, default=['logits'],
        help = 'Which model layers to be returned by the model\'s inference during testing.')

parser.add_argument('--optChoice', action='store', default='default',
        help = 'String indicating optimizer choice')

parser.add_argument('--gradClipValue', action='store', type=float, default=5.0,
        help = 'Value of normalized gradient at which to clip.')

# Experiment parameters

parser.add_argument('--dataset', action= 'store', required=True,
        help= 'Dataset (UCF101, HMDB51)')

parser.add_argument('--numGpus', action= 'store', type=int, default=1,
        help = 'Number of Gpus used for calculation')

parser.add_argument('--gpuList', nargs='+',type=str, default=[],
        help = 'List of GPU IDs to be used')

parser.add_argument('--train', action= 'store', type=int, default=1,
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

parser.add_argument('--clipStride', action='store', type=int, default=0,
        help = 'Number of frames that overlap between clips, 0 indicates no overlap and negative values indicate a gap of frames between clips')

parser.add_argument('--numClips', action='store', type=int, default=-1,
        help = 'Number of clips to break video into, -1 indicates breaking the video into the maximum number of clips based on clipLength, clipStride, and clipOffset')

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

parser.add_argument('--shuffleSeed', action='store', type=int, default=0,
        help = 'Seed integer for random shuffle of files in load_dataset function')

parser.add_argument('--preprocDebugging', action='store', type=int, default=0,
        help = 'Boolean indicating whether to load videos and clips in a queue or to load them directly for debugging (Default 0)')

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

model = models_import.create_model_object(modelName = model_name,
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
                                   loadWeights = args.loadWeights,
                                   verbose = args.verbose)



def _average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
                     is over individual gradients. The inner list is over the gradient
                     calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """

    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []

        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # END FOR

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    # END FOR
    return average_grads

def train(model, input_dims, output_dims, seq_length, size, num_gpus, dataset, experiment_name, load_model, num_vids, n_epochs, split, base_data_path, f_name, learning_rate_init, wd, save_freq, clip_length, video_offset, clip_offset, num_clips, clip_stride, batch_size, loss_type, metrics_dir, loaded_checkpoint, verbose, opt_choice, gpu_list, grad_clip_value, preproc_method, random_init, shuffle_seed, preproc_debugging):
    """
    Training function used to train or fine-tune a chosen model
    Args:
        :model:              tf-activity-recognition framework model object
        :input_dims:         Number of frames used in input
        :output_dims:        Integer number of classes in current dataset
        :seq_length:         Length of output sequence expected from LSTM
        :size:               List detailing height and width of frame
        :num_gpus:           Number of gpus to use when training
        :dataset:            Name of dataset being processed
        :experiment_name:    Name of current experiment
        :load_model:         Boolean variable indicating whether to load from a checkpoint or not
        :num_vids:           Number of videos to be used for training
        :n_epochs:           Total number of epochs to train
        :split:              Split of dataset being used
        :base_data_path:     Full path to root directory containing datasets
        :f_name:             Specific video directory within a chosen split of a dataset
        :learning_rate_init: Initializer for learning rate
        :wd:                 Weight decay
        :save_freq:          Frequency, in epochs, with which to save
        :clip_length:        Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :video_offset:       String indicating where to begin selecting video clips (provided clipOffset is None)
        :clip_offset:        "none" or "random" indicating where to begin selecting video clips
        :num_clips:          Number of clips to break video into
        :clip_stride:        Number of frames that overlap between clips, 0 indicates no overlap and negative values indicate a gap of frames between clips
        :batch_size:         Number of clips to load into the model each step.
        :loss_type:          String declaring loss type associated with a chosen model
        :metrics_dir:        Name of subdirectory within the experiment to store metrics. Unique directory names allow for parallel testing
        :loaded_checkpoint:  Specify the exact checkpoint of saved model to be loaded for further training/testing
        :verbose:            Boolean to indicate if all print statement should be procesed or not
        :opt_choice:         String indicating optimizer selected
        :gpu_list:           List of GPU IDs to be used
        :grad_clip_value:    Float value at which to clip normalized gradients
        :lr_boundaries:      List of epoch boundaries at which lr will be updated
        :lr_values:          List of lr multipliers to learning_rate_init at boundaries mentioned in lr_boundaries
        :preproc_method:     The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing
        :random_init:        Randomly initialize model weights, not loading from any files (deafult False)
        :preproc_debugging:  Boolean indicating whether to load videos and clips in a queue or to load them directly for debugging (Default 0)

    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:

        # Initializers for checkpoint and global step variable
        ckpt    = None
        gs_init = 0

        ################################### Checkpoint loading block #######################################################

        # Load pre-trained/saved model to continue training (or fine-tune)
        if load_model:
            try:
                ckpt, gs_init, learning_rate_init = load_checkpoint(model.name, dataset, experiment_name, loaded_checkpoint, preproc_method)
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
        global_step        = tf.Variable(gs_init, name='global_step', trainable=False)
        number_of_videos   = tf.Variable(num_vids, name='number_of_videos', trainable=False)
        number_of_epochs   = tf.Variable(n_epochs, name='number_of_epochs', trainable=False)
        video_step         = tf.Variable(1.0, name='video_step', trainable=False)
        istraining         = True
        reuse_variables    = None

        # TF session setup
        config  = tf.ConfigProto(allow_soft_placement=True)
        sess    = tf.Session(config=config)
        init    = tf.global_variables_initializer()

        # Variables get randomly initialized into tf graph
        sess.run(init)

        tower_losses       = []
        tower_grads        = []
        tower_slogits      = []

        data_path = os.path.join(base_data_path, 'tfrecords_'+dataset, 'Split'+str(split), f_name)

        # Setup tensors for models
        # input_data_tensor - [batchSize, inputDims, height, width, channels]
        input_data_tensor, labels_tensor, names_tensor = load_dataset(model, num_gpus, batch_size, output_dims, input_dims, seq_length, size, data_path, dataset, istraining, clip_length, video_offset, clip_offset, num_clips, clip_stride, video_step, preproc_debugging, shuffle_seed, verbose)

        ############### TO DO: FIX THIS ASAP ########################
        if ((batch_size == 1) and (num_clips==1)):
            sess.run(tf.assign_add(video_step, -2))

        else:
            sess.run(tf.assign_add(video_step, -1))

        # END IF
        ############################################################


        learning_rate = tf.Variable(learning_rate_init, name='learning_rate', trainable=False)

        # Define optimizer (Current selection is only momentum optimizer)
        if opt_choice == 'gd':
            optimizer = lambda lr: tf.train.GradientDescentOptimizer(lr)

        elif opt_choice == 'adam':
            optimizer = lambda lr: tf.train.AdamOptimizer(lr)

        else:
            optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        # END IF


        """ Multi-GPU setup: 1) Associate gpu device to specific model replica
                             2) Setup tower name scope for variables
        """

        ################# GPU list check block ####################

        assert((len(gpu_list) == num_gpus) or (len(gpu_list) == 0))

        if len(gpu_list) == 0:
            gpu_list = [str(x) for x in range(num_gpus)]

        # END IF

        ###########################################################


        ################################################## Setup TF graph block ######################################################
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:'+str(gpu_list[gpu_idx])):
                with tf.name_scope('%s_%d' % ('tower', int(gpu_list[gpu_idx]))) as scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
                        returned_layers = model.inference(input_data_tensor[gpu_idx*batch_size:gpu_idx*batch_size+batch_size,:,:,:,:],
                                                 istraining,
                                                 input_dims,
                                                 output_dims,
                                                 seq_length,
                                                 scope,
                                                 return_layer = ['logits'],
                                                 weight_decay = wd)

                        logits          = tf.cast(returned_layers[0], tf.float32)

                        # Calculating Softmax for probability outcomes : Can be modified, make function internal to model
                        slogits = tf.nn.softmax(logits)

                    # END WITH

                    reuse_variables = True

                    """ Within GPU mini-batch: 1) Calculate loss,
                                               2) Initialize optimizer with required learning rate and
                                               3) Compute gradients
                                               4) Aggregate losses, gradients and logits
                    """

                    total_loss = model.loss(logits, labels_tensor[gpu_idx*batch_size:gpu_idx*batch_size+batch_size, :], loss_type)
                    opt        = optimizer(learning_rate)
                    gradients  = opt.compute_gradients(total_loss, vars_.trainable_variables())

                    tower_losses.append(total_loss)
                    tower_grads.append(gradients)
                    tower_slogits.append(slogits)

                # END WITH

            # END WITH

        # END FOR


        """  After: 1) Computing gradients and losses need to be stored and averaged
                    2) Clip gradients by norm to required value
                    3) Apply mean gradient updates
        """

        gradients            = _average_gradients(tower_grads)
        gradients, variables = zip(*gradients)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, grad_clip_value)
        gradients            = list(zip(clipped_gradients, variables))
        grad_updates         = opt.apply_gradients(gradients, global_step=global_step, name="train")
        train_op             = grad_updates

        ############################################################################################################################################


        ######################### Logger Setup block ######################################

        # Logging setup initialization (Naming format: Date, month, hour, minute, second)
        log_name     = ("exp_train_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name))
        make_dir('results')
        make_dir(os.path.join('results',model.name))
        make_dir(os.path.join('results',model.name, dataset))
        make_dir(os.path.join('results',model.name, dataset, preproc_method))
        make_dir(os.path.join('results',model.name, dataset, preproc_method, experiment_name))
        make_dir(os.path.join('results',model.name, dataset, preproc_method, experiment_name, 'checkpoints'))
        curr_logger = Logger(os.path.join('logs', model.name, dataset, preproc_method, metrics_dir, log_name))

        ####################################################################################

        init    = tf.global_variables_initializer()
        coord   = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

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


        # Initialize tracking variables
        previous_vid_name = ""
        videos_loaded     = 0
        tot_count         = 0
        acc               = 0
        epoch_count       = 0
        tot_load_time     = 0.0
        tot_train_time    = 0.0
        last_loss         = None


        losses         = []
        total_pred     = []
        save_data      = []
        total_params   = []
        losses_tracker = []


        # Timing test setup
        time_init = time.time()

        batch_count = 0
        epoch_acc   = 0
        l_r         = learning_rate_init

        ########################################## Training loop block ################################################################

        # Loop epoch number of time over the training set
        while videos_loaded < n_epochs*num_vids:
            # Variable to update during epoch intervals
            if (epoch_count+1)*num_vids <= videos_loaded < (epoch_count+1)*num_vids + num_gpus*batch_size:
                batch_count = 0
                epoch_acc   = 0

                if epoch_count % save_freq == 0 and tot_count > 0:
                    if verbose:
                        print "Saving..."

                    save_checkpoint(sess, model.name, dataset, experiment_name, preproc_method, l_r, global_step.eval(session=sess))

                # END IF

                epoch_count += 1

            # END IF


            time_pre_train = time.time()

            ######################################### Running TF training session block ##################################  
            _, loss_train, predictions, gs, labels, vid_names, l_r, track_vars = sess.run([train_op, tower_losses,
                                                                       tower_slogits, global_step,
                                                                       labels_tensor, names_tensor,
                                                                       learning_rate, model.get_track_variables()])

            ################################################################################################################

            if verbose:
                print vid_names

            for name in vid_names:
                if name != previous_vid_name:
                    videos_loaded += 1
                    previous_vid_name = name
                tot_count += 1

            ######## Adaptive Learning Rate Control Block ############################

            losses_tracker.append(np.mean(loss_train))

            if videos_loaded % 10 == 0 and videos_loaded > 0:
                if last_loss is None:
                    last_loss = sum(losses_tracker)/10

                else:
                    difference_loss = last_loss - sum(losses_tracker)/10
                    last_loss = sum(losses_tracker)/10

                    if abs(difference_loss) < 0.001:
                        learning_rate/=10

                    # END IF

                # END IF

                if len(losses_tracker) == 10:
                    losses_tracker = []

                # END IF

            # END IF

            ###########################################################################

            # Transpose the extracted layers such that the mean is taken across the gpus and over any matrix with more than 1 dimension
            params_array = []
            for key in track_vars.keys():
                curr_params = np.array(track_vars[key])
                if len(curr_params.shape) > 1:
                    indices = np.arange(len(curr_params.shape)) + 1
                    indices[-1] = 0
                    curr_params = curr_params.transpose(indices)
                    params_array.append(np.mean(curr_params, axis=tuple(range(len(curr_params.shape))[1:])))

                else:
                    params_array.append([np.mean(curr_params)])

                # END IF

            # END FOR


            #################### Training accuracy computation block ###############

            # Compute training epoch accuracy
            for gpu_pred_idx in range(len(predictions)):
                for batch_idx in range(predictions[gpu_pred_idx].shape[0]):
                    pred = np.mean(predictions[gpu_pred_idx][batch_idx], 0).argmax()

                    if pred == labels[gpu_pred_idx*batch_size + batch_idx][0]:
                        epoch_acc +=1

                    # END IF

                    batch_count+=1

                # END FOR

            # END FOR

            ###################### Add variables to be tracked to logger #############

            time_post_train = time.time()
            tot_train_time += time_post_train - time_pre_train

            if verbose:
                print 'train_time: ', time_post_train-time_pre_train
                print 'step, loss: ', gs, loss_train

            # END IF

            curr_logger.add_scalar_value('train/train_time',time_post_train - time_pre_train, step=gs)
            curr_logger.add_scalar_value('train/loss',      float(np.mean(loss_train)), step=gs)
            curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(batch_count), step=gs)


            for layer in range(len(params_array)):
                for p in range(len(params_array[layer])):
                    curr_logger.add_scalar_value('tracked_training_variables/'+str(track_vars.keys()[layer]+'_'+str(p)), float(params_array[layer][p]), step=gs)

                # END FOR

            # END FOR

	    total_params.append(params_array)

            curr_logger.add_scalar_value('tracked_training_variables/learning_rate', float(l_r), step=gs)

        # END WHILE

        #########################################################################################################################################################

        if verbose:
            print "Saving..."

        # END IF

        save_checkpoint(sess, model.name, dataset, experiment_name, preproc_method, l_r, gs)
        coord.request_stop()
        coord.join(threads)

        if verbose:
            print "Tot train time: ", tot_train_time
            print "Tot time:       ", time.time()-time_init

    # END WITH

    # Save tracked parameterization variables as a numpy file
	if len(total_params) != 0:
	    total_params = np.array(total_params).flatten()
            make_dir(os.path.join('results',model.name, dataset, preproc_method, experiment_name, metrics_dir))

	    if os.path.isfile(os.path.join('results', model.name, dataset, preproc_method, experiment_name, metrics_dir, 'train_params_'+dataset+'.npy')):

	        loaded_params = np.load(os.path.join('results', model.name, dataset, preproc_method, experiment_name, metrics_dir, 'train_params_'+dataset+'.npy'))
		total_params = np.concatenate([loaded_params, total_params])

            # END IF

	    np.save(os.path.join('results', model.name, dataset, preproc_method, experiment_name, metrics_dir, 'train_params_'+dataset+'.npy'), total_params)

        # END IF

if __name__=="__main__":
    if args.train:
        train(  model               = model,
                input_dims          = args.inputDims,
                output_dims         = args.outputDims,
                seq_length          = args.seqLength,
                size                = [args.size, args.size],
                num_gpus            = args.numGpus,
                dataset             = args.dataset,
                experiment_name     = args.expName,
                load_model          = args.load,
                num_vids            = args.numVids,
                n_epochs            = args.nEpochs,
                split               = args.split,
                base_data_path      = args.baseDataPath,
                f_name              = args.fName,
                learning_rate_init  = args.lr,
                wd                  = args.wd,
                save_freq           = args.saveFreq,
                clip_length         = args.clipLength,
                video_offset        = args.videoOffset,
                clip_offset         = args.clipOffset,
                num_clips           = args.numClips,
                clip_stride         = args.clipStride,
                batch_size          = args.batchSize,
                loss_type           = args.lossType,
                metrics_dir         = args.metricsDir,
                loaded_checkpoint   = args.loadedCheckpoint,
                verbose             = args.verbose,
                opt_choice          = args.optChoice,
                gpu_list            = args.gpuList,
                grad_clip_value     = args.gradClipValue,
                preproc_method      = args.preprocMethod,
                random_init         = args.randomInit,
                shuffle_seed        = args.shuffleSeed,
                preproc_debugging   = args.preprocDebugging)

    # END IF
