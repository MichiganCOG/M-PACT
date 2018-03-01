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
from models                 import *
from utils                  import initialize_from_dict, save_checkpoint, load_checkpoint, make_dir, Metrics
from Queue                  import Queue
from logger                 import Logger
from random                 import shuffle
from load_dataset_tfrecords import load_dataset

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

def train(model, input_dims, output_dims, seq_length, size, num_gpus, dataset, experiment_name, load_model, num_vids, n_epochs, split, base_data_path, f_name, learning_rate_init, wd, save_freq, return_layer, clip_length, video_offset, clip_offset, num_clips, clip_overlap, batch_size, loss_type, metrics_dir, loaded_checkpoint, verbose, opt_choice):
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
        :return_layer:       Layers to be tracked during training
        :clip_length:        Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :video_offset:       String indicating where to begin selecting video clips (provided clipOffset is None)
        :clip_offset:        "none" or "random" indicating where to begin selecting video clips
        :num_clips:          Number of clips to break video into
        :clip_overlap:       Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential
        :batch_size:         Number of clips to load into the model each step.
        :loss_type:          String declaring loss type associated with a chosen model
        :metrics_dir:        Name of subdirectory within the experiment to store metrics. Unique directory names allow for parallel testing
        :loaded_checkpoint:  Specify the exact checkpoint of saved model to be loaded for further training/testing
        :verbose:            Boolean to indicate if all print statement should be procesed or not
        :opt_choice:         String indicating optimizer selected

    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:

        # Ensure that the default return layer is logits
        if len(return_layer) == 0:
            return_layer = ['logits']

        # Ensure first layer requested in return sequence is "logits" always
        if return_layer[0] != 'logits':
            return_layer.insert(0, 'logits')

        # END IF

        # Initializers for checkpoint and global step variable
        ckpt    = None
        gs_init = 0

        # Load pre-trained/saved model to continue training (or fine-tune)
        if load_model:
            try:
                ckpt, gs_init, learning_rate_init = load_checkpoint(model.name, dataset, experiment_name, loaded_checkpoint)
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

        model_params_array = []
        for rl in range(len(return_layer)-1):
            model_params_array.append([])

        # END FOR

        tower_losses       = []
        tower_grads        = []
        tower_slogits      = []

        data_path = os.path.join(base_data_path, 'tfrecords_'+dataset, 'Split'+str(split), f_name)

        # Setup tensors for models
        input_data_tensor, labels_tensor, names_tensor, video_step_update = load_dataset(model, num_gpus, batch_size, output_dims, input_dims, seq_length, size, data_path, dataset, istraining, clip_length, video_offset, clip_offset, num_clips, clip_overlap, video_step, verbose)

        if ((batch_size == 1) and (num_clips==1)):
            sess.run(tf.assign_add(video_step, -2))
        else:
            sess.run(tf.assign_add(video_step, -1))

        # Define optimizer (Current selection is only momentum optimizer)
        if opt_choice == 'gd':
            optimizer = lambda lr: tf.train.GradientDescentOptimizer(lr)

        else:
            optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        # END IF

        """ Multi-GPU setup: 1) Associate gpu device to specific model replica
                             2) Setup tower name scope for variables
        """
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:'+str(gpu_idx)):
                with tf.name_scope('%s_%d' % ('tower', gpu_idx)) as scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
                        returned_layers = model.inference(input_data_tensor[gpu_idx*batch_size:gpu_idx*batch_size+batch_size,:,:,:,:],
                                                 istraining,
                                                 input_dims,
                                                 output_dims,
                                                 seq_length,
                                                 scope,
                                                 return_layer = return_layer,
                                                 weight_decay = wd)

                        logits          = tf.cast(returned_layers[0], tf.float32)

                        for rl in range(len(returned_layers[1:])):
                            model_params_array[rl].append(returned_layers[1:][rl])

                        # END FOR

                        # Calculating Softmax for probability outcomes : Can be modified, make function internal to model
                        slogits = tf.nn.softmax(logits)

                        lr = vs.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(learning_rate_init))

                    # END WITH

                    reuse_variables = True

                    """ Within GPU mini-batch: 1) Calculate loss,
                                               2) Initialize optimizer with required learning rate and
                                               3) Compute gradients
                                               4) Aggregate losses, gradients and logits
                    """

                    total_loss = model.loss(logits, labels_tensor[gpu_idx*batch_size:gpu_idx*batch_size+batch_size, :], loss_type)
                    opt        = optimizer(lr)
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
        if 'tsn' in model.name:
            clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 10.0)
        else:
            clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 5.0)
        gradients            = list(zip(clipped_gradients, variables))
        grad_updates         = opt.apply_gradients(gradients, global_step=global_step, name="train")
        train_op             = grad_updates


        # Logging setup initialization (Naming format: Date, month, hour, minute, second)
        log_name     = ("exp_train_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name))
        make_dir('results')
        make_dir(os.path.join('results',model.name))
        make_dir(os.path.join('results',model.name, dataset))
        make_dir(os.path.join('results',model.name, dataset, experiment_name))
        make_dir(os.path.join('results',model.name, dataset, experiment_name, 'checkpoints'))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, metrics_dir, log_name))

        init    = tf.global_variables_initializer()
        coord   = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

        # Variables get randomly initialized into tf graph
        sess.run(init)

        # Model variables initialized from previous saved models
        initialize_from_dict(sess, ckpt, model.name)
        del ckpt

        # Initialize tracking variables
        previous_vid_name = ""
        videos_loaded     = 0
        tot_count         = 0
        acc               = 0
        epoch_count       = 0
        tot_load_time     = 0.0
        tot_train_time    = 0.0

        losses     = []
        total_pred = []
        save_data  = []

        lr            = learning_rate_init
        learning_rate = lr

        # Timing test setup
        time_init = time.time()

        batch_count = 0
        epoch_acc = 0
        # Loop epoch number of time over the training set
        while videos_loaded < n_epochs*num_vids:
            # Variable to update during epoch intervals
            if (epoch_count+1)*num_vids + 1 <= videos_loaded < (epoch_count+1)*num_vids + num_gpus*batch_size + 1:
                batch_count = 0
                epoch_acc   = 0

                if epoch_count % save_freq == 0 and tot_count > 0:
                    if verbose:
                        print "Saving..."

                    save_checkpoint(sess, model.name, dataset, experiment_name, learning_rate, global_step.eval(session=sess))

                # END IF

                epoch_count += 1

            # END IF


            time_pre_train = time.time()

            _, loss_train, predictions, gs, labels, params, vid_names, idt, vid_step = sess.run([train_op, tower_losses,
                                                                       tower_slogits, global_step,
                                                                       labels_tensor, model_params_array,
                                                                       names_tensor, input_data_tensor, video_step_update])

            if verbose:
                print vid_names

            for name in vid_names:
                if name != previous_vid_name:
                    videos_loaded += 1
                    previous_vid_name = name
                tot_count += 1

            # Transpose the extracted layers such that the mean is taken across the gpus and over any matrix with more than 1 dimension
            params_array = []
            for rl in range(len(return_layer[1:])):
                curr_params = np.array(params[rl])
                if len(curr_params.shape) > 1:
                    indices = np.arange(len(curr_params.shape)) + 1
                    indices[-1] = 0
                    curr_params = curr_params.transpose(indices)
                    params_array.append(np.mean(curr_params, axis=tuple(range(len(curr_params.shape))[1:])))

                else:
                    params_array.append([np.mean(curr_params)])

                # END IF

            # END FOR


            # Compute training epoch accuracy
            for gpu_idx in range(len(predictions)):
                for batch_idx in range(predictions[gpu_idx].shape[0]):
                    pred = np.mean(predictions[gpu_idx][batch_idx], 0).argmax()

                    if pred == labels[gpu_idx*batch_size + batch_idx][0]:
                        epoch_acc +=1

                    # END IF

                    batch_count+=1

                # END FOR

            # END FOR

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
                    curr_logger.add_scalar_value('tracked_training_variables/'+str(return_layer[1:][layer]+'_'+str(p)), float(params_array[layer][p]), step=gs)

                # END FOR

            # END FOR

        # END WHILE

        if verbose:
            print "Saving..."

        # END IF

        save_checkpoint(sess, model.name, dataset, experiment_name, learning_rate, gs)
        coord.request_stop()
        coord.join(threads)

        if verbose:
            print "Tot train time: ", tot_train_time
            print "Tot time:       ", time.time()-time_init

    # END WITH


def test(model, input_dims, output_dims, seq_length, size, dataset, loaded_dataset, experiment_name, num_vids, split, base_data_path, f_name, load_model, return_layer, clip_length, video_offset, clip_offset, num_clips, clip_overlap, metrics_method, batch_size, metrics_dir, loaded_checkpoint, verbose):
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

    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:
        is_training = False

        # Initializers for checkpoint and global step variable
        ckpt    = None
        gs_init = 0

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

        # Initialize model variables
        istraining       = False
        global_step      = tf.Variable(gs_init, name='global_step', trainable=False)
        number_of_videos = tf.Variable(num_vids, name='number_of_videos', trainable=False)
        video_step       = tf.Variable(1.0, name='video_step', trainable=False)

        data_path   = os.path.join(base_data_path, 'tfrecords_'+dataset, 'Split'+str(split), f_name)

        # Setting up tensors for models
        input_data_tensor, labels_tensor, names_tensor, video_step_update = load_dataset(model, 1, batch_size, output_dims, input_dims, seq_length, size, data_path, dataset, istraining, clip_length, video_offset, clip_offset, num_clips, clip_overlap, video_step, verbose)
        # input_data_tensor shape - [num_gpus*batch_size, input_dims, size[0], size[1], channels]

        # Model Inference
        logits = model.inference(input_data_tensor[0:batch_size,:,:,:,:],
                                 istraining,
                                 input_dims,
                                 output_dims,
                                 seq_length,
                                 scope,
                                 return_layer = return_layer)[0]

        # Logits
        softmax = tf.nn.softmax(logits)


        # Logger setup (Name format: Date, month, hour, minute and second, with a prefix of exp_test)
        log_name    = ("exp_test_%s_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                               dataset, experiment_name, metrics_method))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, metrics_dir, log_name))
        make_dir(os.path.join('results',model.name))
        make_dir(os.path.join('results',model.name, dataset))
        make_dir(os.path.join('results',model.name, dataset, experiment_name))
        make_dir(os.path.join('results',model.name, dataset, experiment_name, metrics_dir))

        # TF session setup
        sess    = tf.Session()
        init    = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord   = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        metrics = Metrics( output_dims, seq_length, curr_logger, metrics_method, is_training, model.name, experiment_name, dataset, metrics_dir, verbose=verbose)

        # Variables get randomly initialized into tf graph
        sess.run(init)

        # Model variables initialized from previous saved models
        initialize_from_dict(sess, ckpt, model.name)
        del ckpt

        acc               = 0
        count             = 0
        videos_loaded     = 0
        previous_vid_name = ''
        total_pred        = []

        if verbose:
            print "Begin Testing"

        # END IF

        while videos_loaded <= num_vids:
            output_predictions, labels, names, vid_step = sess.run([softmax, labels_tensor, names_tensor, video_step_update])

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

    # END WITH

    coord.request_stop()
    coord.join(threads)

    total_accuracy = metrics.total_classification()
    total_pred = metrics.get_predictions_array()

    if verbose:
        print "Total accuracy : ", total_accuracy
        print total_pred

    # Save results in numpy format
    np.save(os.path.join('results', model.name, loaded_dataset, experiment_name, metrics_dir, 'test_predictions_'+dataset+"_"+metrics_method+'.npy'), np.array(total_pred))


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', action= 'store', required=True,
            help= 'Model architecture (c3d, lrcn, tsn, vgg16, resnet)')

    parser.add_argument('--dataset', action= 'store', required=True,
            help= 'Dataset (UCF101, HMDB51)')

    parser.add_argument('--numGpus', action= 'store', type=int, default=1,
            help = 'Number of Gpus used for calculation')

    parser.add_argument('--train', action= 'store', required=True, type=int,
            help = 'Binary value to indicate training or evaluation instance')

    parser.add_argument('--load', action='store', type=int, default=0,
            help = 'Whether you want to load a saved model to train from scratch.')

    parser.add_argument('--size', action='store', required=True, type=int,
            help = 'Input frame size')

    parser.add_argument('--inputDims', action='store', required=True, type=int,
            help = 'Input Dimensions (Number of frames to pass as input to the model)')

    parser.add_argument('--outputDims', action='store', required=True, type=int,
            help = 'Output Dimensions (Number of classes in dataset)')

    parser.add_argument('--seqLength', action='store', required=True, type=int,
            help = 'Length of sequences for LSTM')

    parser.add_argument('--expName', action='store', required=True,
            help = 'Unique name of experiment being run')

    parser.add_argument('--numVids', action='store', required=True, type=int,
            help = 'Number of videos to be used for training')

    parser.add_argument('--lr', action='store', type=float, default=0.001,
            help = 'Learning Rate')

    parser.add_argument('--wd', action='store', type=float, default=0.0,
            help = 'Weight Decay')

    parser.add_argument('--nEpochs', action='store', type=int, default=1,
            help = 'Number of Epochs')

    parser.add_argument('--split', action='store', type=int, default=1,
            help = 'Dataset split to use')

    parser.add_argument('--baseDataPath', action='store', default='/z/dat',
            help = 'Path to datasets')

    parser.add_argument('--fName', action='store',
            help = 'Which dataset list to use (trainlist, testlist, vallist)')

    parser.add_argument('--saveFreq', action='store', type=int, default=1,
            help = 'Frequency in epochs to save model checkpoints')

    parser.add_argument('--loadedDataset', action= 'store', default='HMDB51',
            help= 'Dataset (UCF101, HMDB51)')

    parser.add_argument('--clipLength', action='store', type=int, default=-1,
            help = 'Length of clips to cut video into, -1 indicates using the entire video as one clip')

    parser.add_argument('--videoOffset', action='store', default='none',
            help = '(none or random) indicating where to begin selecting video clips assuming clipOffset is none')

    parser.add_argument('--clipOffset', action='store', default='none',
            help = '(none or random) indicating if clips are seleted sequentially or randomly')

    parser.add_argument('--clipOverlap', action='store', type=int, default=0,
            help = 'Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential')

    parser.add_argument('--numClips', action='store', type=int, default=-1,
            help = 'Number of clips to break video into, -1 indicates breaking the video into the maximum number of clips based on clipLength, clipOverlap, and clipOffset')

    parser.add_argument('--metricsMethod', action='store', default='avg_pooling',
            help = 'Which method to use to calculate accuracy metrics. (avg_pooling, last_frame, svm, svm_train or extract_features)')

    parser.add_argument('--returnLayer', nargs='+',type=str, default=['logits'],
            help = 'Which model layers to be returned by the models\' inference and logged.')

    parser.add_argument('--batchSize', action='store', type=int, default=1,
            help = 'Number of clips to load into the model each step.')

    parser.add_argument('--lossType', action='store', default='full_loss',
            help = 'String defining loss type associated with chosen model.')

    parser.add_argument('--metricsDir', action='store', type=str, default='default',
            help = 'Name of sub directory within experiment to store metrics. Unique directory names allow for parallel testing.')

    parser.add_argument('--loadedCheckpoint', action='store', type=int, default=-1,
            help = 'Specify the step of the saved model checkpoint that will be loaded for testing. Defaults to most recent checkpoint.')

    parser.add_argument('--modelAlpha', action='store', type=float, default=1.,
            help = 'Resampling factor for constant value resampling and alpha initialization')

    parser.add_argument('--inputAlpha', action='store', type=float, default=1.,
            help = 'Resampling factor for constant value resampling of input video, used mainly for testing models.')

    parser.add_argument('--resampleFrames', action='store', type=int, default=16,
            help = 'Number of frames remaining after resampling within model inference.')

    parser.add_argument('--verbose', action='store', type=int, default=1,
            help = 'Boolean switch to display all print statements or not')

    parser.add_argument('--optChoice', action='store', default='default',
            help = 'String indicating optimizer choice')

    args = parser.parse_args()

    print "Setup of current experiments"
    print "############################"
    print args
    print "############################ \n"
    model_name = args.model

    # Associating models
    #if model_name == 'vgg16':
    #    model = VGG16(args.inputDims, 25, verbose=args.verbose)

    if model_name == 'resnet':
        model = ResNet(args.inputDims, 25, verbose=args.verbose)

    elif model_name == 'resnet_cb_1':
	model = ResNet_cb_1(args.inputDims, args.modelAlpha, args.inputAlpha, verbose=args.verbose)

    elif model_name == 'resnet_cb_2':
	model = ResNet_cb_2(args.inputDims, args.modelAlpha, args.inputAlpha, verbose=args.verbose)

    elif model_name == 'resnet_cb_3':
	model = ResNet_cb_3(args.inputDims, args.modelAlpha, args.inputAlpha, verbose=args.verbose)

    elif model_name == 'resnet_offset_fixed':
        model = ResNet_Offset_Fixed(args.inputDims, 25, args.modelAlpha, args.inputAlpha, verbose=args.verbose)

    elif model_name == 'c3d':
        model = C3D(input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'c3d_cvr':
        model = C3D_CVR(cvr=args.modelAlpha, input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'c3d_rr':
        model = C3D_RR(input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'c3d_sr':
        model = C3D_SR(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'c3d_sr_quant':
        model = C3D_SR_QUANT(input_dims=args.inputDims, clip_length=args.clipLength, model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, num_vids=args.numVids, num_epochs=args.nEpochs, batch_size=args.batchSize, num_clips=args.numClips, num_gpus=args.numGpus, verbose=args.verbose)

    elif model_name == 'c3d_sr_step':
        model = C3D_SR_STEP(input_dims=args.inputDims, clip_length=args.clipLength, model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, num_vids=args.numVids, num_epochs=args.nEpochs, batch_size=args.batchSize, num_clips=args.numClips, num_gpus=args.numGpus, verbose=args.verbose)

    elif model_name == 'c3d_alpha':
        model = C3D_ALPHA(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames, verbose=args.verbose)

    elif model_name == 'c3d_alpha_sine':
        model = C3D_ALPHA_SINE(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames, verbose=args.verbose)

    elif model_name == 'c3d_alpha_sine_100':
        model = C3D_ALPHA_SINE_100(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames, verbose=args.verbose)

    elif model_name == 'c3d_alpha_exp':
        model = C3D_ALPHA_EXP(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames, verbose=args.verbose)

    elif model_name == 'c3d_alpha_div_100':
        model = C3D_ALPHA_DIV_100(model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames, verbose=args.verbose)

    elif model_name == 'i3d':
        model = I3D(verbose=args.verbose)

    elif model_name == 'i3d_cvr':
       model = I3D_CVR(cvr=args.modelAlpha, input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'i3d_rr':
       model = I3D_RR(input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'i3d_sr':
       model = I3D_SR(input_alpha=args.inputAlpha, verbose=args.verbose)

    elif model_name == 'tsn':
        num_seg = args.inputDims
        if args.train:
            num_seg = 3

        init = False
        if 'init' in args.expName:
            init = True
        model = TSN(args.inputDims, args.outputDims, args.expName, num_seg, init)

    elif model_name == 'tsn_RIL':
        num_seg = args.inputDims
        if args.train:
            num_seg = 3

        init = False
        if 'init' in args.expName:
            init = True

        model = TSN_RIL(args.inputDims, args.outputDims, args.expName, num_seg, init, model_alpha=args.modelAlpha, input_alpha=args.inputAlpha, resample_frames=args.resampleFrames)

    elif model_name == 'tsn_cvr':
        num_seg = args.inputDims
        if args.train:
            num_seg = 3

        init = False
        if 'init' in args.expName:
            init = True

        model = TSN_CVR(args.inputDims, args.outputDims, args.expName, num_seg, init, cvr=args.modelAlpha, input_alpha=args.inputAlpha)

    elif model_name == 'tsn_rr':
        num_seg = args.inputDims
        if args.train:
            num_seg = 3

        init = False
        if 'init' in args.expName:
            init = True

        model = TSN_RR(args.inputDims, args.outputDims, args.expName, num_seg, init, input_alpha=args.inputAlpha)

    elif model_name == 'tsn_sr':
        num_seg = args.inputDims
        if args.train:
            num_seg = 3

        init = False
        if 'init' in args.expName:
            init = True

        model = TSN_SR(args.inputDims, args.outputDims, args.expName, num_seg, init, model_alpha=args.modelAlpha, input_alpha=args.inputAlpha)



    #elif model_name == 'resnet_RIL_interp_median_v23_2_1':
    #    model = ResNet_RIL_Interp_Median_v23_2_1(args.inputDims, 25, verbose=args.verbose)

    elif model_name == 'resnet_RIL_interp_median_v23_4':
        model = ResNet_RIL_Interp_Median_v23_4(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v23_7_1':
    #    model = ResNet_RIL_Interp_Median_v23_7_1(inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v31_3':
    #    model = ResNet_RIL_Interp_Median_v31_3(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v34_3_lstm':
    #    model = ResNet_RIL_Interp_Median_v34_3_lstm(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v35_lstm':
    #    model = ResNet_RIL_Interp_Median_v35_lstm(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v36_lstm':
    #    model = ResNet_RIL_Interp_Median_v36_lstm(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v37_lstm':
    #    model = ResNet_RIL_Interp_Median_v37_lstm(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v38':
    #    model = ResNet_RIL_Interp_Median_v38(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v39':
    #    model = ResNet_RIL_Interp_Median_v39(args.inputDims, 25, verbose=args.verbose)

    #elif model_name == 'resnet_RIL_interp_median_v40':
    #    model = ResNet_RIL_Interp_Median_v40(args.inputDims, 25, verbose=args.verbose)

    else:
        print("Model not found, check the import and elif statements")

    # END IF

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
                return_layer        = args.returnLayer,
                clip_length         = args.clipLength,
                video_offset        = args.videoOffset,
                clip_offset         = args.clipOffset,
                num_clips           = args.numClips,
                clip_overlap        = args.clipOverlap,
                batch_size          = args.batchSize,
                loss_type           = args.lossType,
                metrics_dir         = args.metricsDir,
                loaded_checkpoint   = args.loadedCheckpoint,
                verbose             = args.verbose,
                opt_choice          = args.optChoice)

    else:
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
                verbose           = args.verbose)
