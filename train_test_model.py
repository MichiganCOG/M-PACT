import os
import time
import argparse
import numpy      as np
import tensorflow as tf

from utils                                        import *
from logger                                       import Logger
from Queue                                        import Queue
from models.lrcn.lrcn_model                       import LRCN
from models.vgg16.vgg16_model                     import VGG16
from models.resnet.resnet_model                   import ResNet
from load_dataset                                 import load_dataset

from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope    as vs
from tensorflow.python.ops import variables         as vars_

from models.resnet_RIL.resnet_RIL_interp_mean_model_v1 import ResNet_RIL_Interp_Mean_v1
from models.resnet_RIL.resnet_RIL_interp_mean_model_v2 import ResNet_RIL_Interp_Mean_v2
from models.resnet_RIL.resnet_RIL_interp_mean_model_v3 import ResNet_RIL_Interp_Mean_v3

from models.resnet_RIL.resnet_RIL_interp_median_model_v1 import ResNet_RIL_Interp_Median_v1
from models.resnet_RIL.resnet_RIL_interp_median_model_v2 import ResNet_RIL_Interp_Median_v2
from models.resnet_RIL.resnet_RIL_interp_median_model_v3 import ResNet_RIL_Interp_Median_v3

from models.resnet_RIL.resnet_RIL_interp_max_model_v1 import ResNet_RIL_Interp_Max_v1
from models.resnet_RIL.resnet_RIL_interp_max_model_v2 import ResNet_RIL_Interp_Max_v2
from models.resnet_RIL.resnet_RIL_interp_max_model_v3 import ResNet_RIL_Interp_Max_v3

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


""" Function used to read in the total number of videos in a given dataset split and shuffle the indices """
def gen_video_list(dataset, model_name, experiment_name, f_name, split, num_vids, shuffle, epoch):

    if epoch == 0 or num_vids == None:
        vid_file       = open(os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'),'r')
        lines          = vid_file.readlines()
        num_total_vids = len(lines)
        vid_file.close()

        if num_total_vids != None:
            vid_list = np.arange(num_total_vids)

            if num_total_vids != num_vids:
                vid_list=vid_list[:num_vids]

            # END IF

        else:
            print "No videos in list provided. Please Check!"

        # END IF

    else:
        vid_list = np.arange(num_vids)

    # END IF

    if shuffle:
        np.random.shuffle(vid_list)

    # END IF

    return vid_list.tolist()


def _validate(model, slogits, sess, experiment_name, logger, dataset, input_dims, output_dims, split, gs, size, x_placeholder, istraining_placeholder, j_placeholder, base_data_path):
    """
    Args:
        :model:                  tf-activity-recognition framework model object
        :slogits:                Tensorflow op that captures the output of inference op from model
        :sess:                   Tensorflow session object
        :experiment_name:        Name of current experiment
        :logger:                 Logger class object
        :dataset:                Name of dataset being processed
        :input_dims:             Number of frames used in input
        :output_dims:            Integer number of classes in current dataset
        :split:                  Split of dataset being used
        :gs:                     Integer for global step count
        :size:                   List detailing height and width of frame
        :x_placeholder:          Tensorflow placeholder for input frames
        :istraining_placeholder: Tensorflow placeholder for boolean indicating phase (TRAIN OR TEST)
        :j_placeholder:          Tensorflow placeholder for number of disjoing sets from application of a sliding window
        :base_data_path:         Full path to root directory containing datasets
    
    """



    if 'HMDB51' in dataset:
        f_name = 'vallist'

    else:
        f_name = 'testlist'
    
    # END IF

    vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, None, False, 0)

    count    = 0
    acc      = 0

    for vid_num in vid_list:
        loaded_data, labels= load_dataset( model, vid_num, f_name, os.path.join(base_data_path, dataset+'HDF5RGB','Split'+str(split)), os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, False, dataset)
        count +=1

        shape = np.array(loaded_data).shape
        labels = np.repeat(labels, input_dims)

        if len(shape) ==4:
            loaded_data = np.array([loaded_data])
            num_clips = 1

        else:
            num_clips = len(loaded_data)

        # END IF

        output_predictions = np.zeros((num_clips, output_dims))

        for clip in range(num_clips):
            input_data= np.reshape(loaded_data[clip], (1, -1, size[0], size[1], 3))
            labels = np.reshape(labels, (1, -1))

            pred = sess.run(slogits, feed_dict={x_placeholder: input_data,
                                                istraining_placeholder  : False,
                                                j_placeholder           : [input_data.shape[1]/K]})

            output_predictions[clip] = np.mean(pred, 0)

        # END FOR

        guess = np.mean(output_predictions, 0).argmax()

        if int(guess) == int(labels[0][0]):
            acc += 1
        
        # END IF

        print "validation acc: ", acc/float(count)
        logger.add_scalar_value('val/step_acc',acc/float(count), step=count)
    
    # END FOR

    logger.add_scalar_value('val/acc',acc/float(count), step=gs)






def train(model, input_dims, output_dims, seq_length, size, num_gpus, dataset, experiment_name, load_model, num_vids, n_epochs, split, base_data_path, f_name, learning_rate_init=0.001, wd=0.0, save_freq = 5, val_freq = 1, k=25):
    """
    Args:
        :model:              tf-activity-recognition framework model object
        :input_dims:         Number of frames used in input
        :output_dims:        Integer number of classes in current dataset
        :seq_length:         Length of output sequence expected from LSTM 
        :size:               List detailing height and width of frame
        :num_gpus:           Number of gpus to use when training
        :dataset:            Name of dataset being processed
        :experiment_name:    Name of current experiment
        :load_model:         Boolean variable indicating whether to load form a checkpoint or not
        :num_vids:           Number of videos to be used for training
        :n_epochs:           Total number of epochs to train 
        :split:              Split of dataset being used
        :base_data_path:     Full path to root directory containing datasets
        :f_name:             Prefix for HDF5 to be used
        :learning_rate_init: Initializer for learning rate
        :wd:                 Weight decay
        :save_freq:          Frequency, in epochs, with which to save
        :val_freq:           Frequency, in epochs, with which to run validaton
        :k:                  Width of temporal sliding window
    
    """

    with tf.name_scope("my_scope") as scope:
        is_training     = True
        global_step     = tf.Variable(0, name='global_step', trainable=False)
        reuse_variables = None

        # Setting up placeholders for models
        x_placeholder          = tf.placeholder(tf.float32, shape=[num_gpus, input_dims, size[0], size[1] ,3], name='x_placeholder')
        y_placeholder          = tf.placeholder(tf.int64, shape=[num_gpus, seq_length], name='y_placeholder')
        istraining_placeholder = tf.placeholder(tf.bool, name='istraining_placeholder')
        j_placeholder          = tf.placeholder(tf.int32, shape=[1], name='j_placeholder')

        tower_losses = []
        tower_grads  = []

        # Define Optimizer
        optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:'+str(gpu_idx)):
                with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
                    logits = model.inference(x_placeholder[gpu_idx,:,:,:,:], istraining_placeholder, input_dims, output_dims, seq_length, scope, k, j_placeholder, weight_decay=wd)

                    # Calculating Softmax for probability outcomes : Can be modified
                    # Make function internal to model
                    slogits = tf.nn.softmax(logits)

                    # Why retain learning rate here ?
                    lr = vs.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(learning_rate_init))

                # END WITH

                reuse_variables = True

                """ Within GPU mini-batch: 1) Calculate loss,
                                           2) Initialize optimizer with required learning rate and
                                           3) Compute gradients
                """
                total_loss = model.loss(logits, y_placeholder[gpu_idx, :])
                opt        = optimizer(lr)
                gradients  = opt.compute_gradients(total_loss, vars_.trainable_variables())

            # END WITH

        # END FOR

        """  After: 1) Computing gradients and losses need to be stored and averaged
                    2) Clip gradients by norm to required value
                    3) Apply mean gradient updates
        """
        gradients, variables = zip(*gradients)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 5.0)
        gradients            = list(zip(clipped_gradients, variables))
        grad_updates         = opt.apply_gradients(gradients, global_step=global_step, name="train")
        train_op             = control_flow_ops.with_dependencies([grad_updates], total_loss)


        # Logging setup initialization
        log_name     = ("exp_train_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name) )
        make_dir(os.path.join('results',model.name,   experiment_name+'_'+dataset))
        make_dir(os.path.join('results',model.name,   experiment_name+'_'+dataset, 'checkpoints'))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, log_name))

        # TF session setup
        config = tf.ConfigProto(allow_soft_placement=True)
        sess   = tf.Session(config=config)
        saver  = tf.train.Saver()
        init   = tf.global_variables_initializer()

        sess.run(init)

        vid_list = []

        if load_model:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name,   experiment_name+ '_'+dataset, 'checkpoints/checkpoint')))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)

            else:
                print "Failed loading checkpoint requested. Please check."
                exit()

            # END IF

        # END IF


        count          = 0
        acc            = 0
        tot_train_time = 0.0 
        tot_load_time  = 0.0 

        save_data      = []
        losses         = []
        total_pred     = []

        lr = learning_rate_init

        # Timing test setup
        time_init = time.time()

        for epoch in range(n_epochs):
            batch_count = 0
            epoch_acc   = 0

            # Generating the video list
            if len(vid_list) == 0 or epoch != 0:
                vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, num_vids, True, epoch)
            
            # END IF

            for vid_num in vid_list:
                mean_loss=[]

                # Setting up load time checks
                time_pre_load = time.time()

                loaded_data, labels= load_dataset(model, vid_num, f_name, 
                                                  os.path.join(base_data_path, dataset+'HDF5RGB','Split'+str(split)), 
                                                  os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'), 
                                                  os.path.join("datasets",dataset,"classInd.txt"), 
                                                  size, is_training, dataset)

                time_post_load = time.time()

                tot_load_time+=time_post_load - time_pre_load

                shape  = np.array(loaded_data).shape
                labels = np.repeat(labels, seqLength)
                print "vidNum: ", vidNum
                print "label: ", labels[0]

                if len(shape) ==4:
                    loaded_data = np.array([loaded_data])
                    num_clips = 1

                else:
                    num_clips = len(loaded_data)

                # END IF

                output_predictions = np.zeros((num_clips, output_dims))

                for clip in range(num_clips):
                    input_data = np.reshape(loaded_data[clip], (1, -1, size[0], size[1], 3))
                    labels     = np.reshape(labels, (1, -1))

                    time_pre_train = time.time()

                    _, loss_train, pred, gs = sess.run([train_op, total_loss, 
                                                       slogits, global_step], 
                                                       feed_dict={x_placeholder: input_data, y_placeholder: labels, 
                                                       j_placeholder: [input_data.shape[1]/k],
                                                       istraining_placeholder: True})

                    mean_loss.append(loss_train)

                    output_predictions[clip] = np.mean(pred, 0)
                    pred                     = np.mean(pred, 0).argmax()
                    print "pred: ", pred

                    if pred == labels[0][0]:
                        acc +=1
                    
                    # END IF

                    time_post_train = time.time()
                    tot_train_time +=time_post_train - time_pre_train

                # END FOR

                batch_count+=1

                guess = np.mean(output_predictions, 0).argmax()

                if int(guess) == int(labels[0][0]):
                    epoch_acc += 1
            
                # END IF

                curr_logger.add_scalar_value('load_time',       time_post_load - time_pre_load, step=gs)
                curr_logger.add_scalar_value('train/train_time',time_post_train - time_pre_train, step=gs)
                curr_logger.add_scalar_value('train/loss',      float(np.mean(mean_loss)), step=gs)
                curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(batch_count), step=gs)
        
            # END FOR

            if epoch % save_freq == 0:
                print "Saving..."
                saver.save(sess, os.path.join('results',model.name, experiment_name + '_'+dataset,'checkpoints/checkpoint'), global_step.eval(session=sess))

            #END IF

            if epoch % val_freq == 0:
                _validate(model, slogits, sess, experiment_name, curr_logger, dataset, input_dims, output_dims, split, gs, size, x_placeholder, istraining_placeholder, j_placeholder, base_data_path)

            # END IF

        # END FOR

    # END WITH
    
    print "Tot load time: ",  tot_load_time
    print "Tot train time: ", tot_train_time
    print "Tot time: ",       time.time()-time_init



def test(model, input_dims, output_dims, seq_length, size, dataset, loaded_dataset, experiment_name, num_vids, split, base_data_path, f_name, k=25):

    """
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
        :f_name:             Prefix for HDF5 to be used
        :k:                  Width of temporal sliding window
    
    """

    with tf.name_scope("my_scope") as scope:
        is_training = False

        x_placeholder = tf.placeholder(tf.float32, shape=[input_dims, size[0], size[1] ,3], name='x_placeholder')
        j_placeholder = tf.placeholder(tf.int32, shape=[1], name='j_placeholder')
        global_step   = tf.Variable(0, name='global_step', trainable=False)

        # Model Inference
        logits = model.inference(x_placeholder, 
                                 False, 
                                 input_dims, 
                                 output_dims, 
                                 seq_length, 
                                 scope, 0,
                                 j_placeholder)

        # Logits
        softmax  = tf.nn.softmax(logits)

        # Logger setup
        log_name = ("exp_test_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, log_name))

        # Session setup
        sess  = tf.Session()
        saver = tf.train.Saver()

        # Initialize Variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name, loaded_dataset, experiment_name, 'checkpoints/checkpoint')))

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)

        else:
            print "Invalid load dataset specified. Please check."
            exit()

        # END IF

        total_pred = []
        acc        = 0
        count      = 0

        vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, num_vids, False, 0)

        all_labels = []
        all_preds  = []

        for vid_num in vid_list:
            count +=1
            loaded_data, labels= load_dataset(model, 1419, f_name, #vid_num, f_name,
                                 os.path.join(base_data_path, dataset+'HDF5RGB', 'Split'+str(split)), 
                                 os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'), 
                                 os.path.join("datasets",dataset,"classInd.txt"), 
                                 size, is_training, dataset)

            labels = np.repeat(labels, input_dims)

            print "vidNum: ", vid_num
            print "label: ", labels[0]

            if len(np.array(loaded_data).shape) ==4:
                loaded_data = np.array([loaded_data])
                num_clips = 1
            else:
                num_clips = len(loaded_data)

            # END IF

            output_predictions = np.zeros((num_clips, output_dims))

            for ldata in range(num_clips):
                input_data = loaded_data[ldata]
                prediction = softmax.eval(session=sess, feed_dict={x_placeholder: input_data,
                                                                   j_placeholder: [input_data.shape[1]/k]})

                output_predictions[ldata] = np.mean(prediction, 0)

            guess = np.mean(output_predictions, 0).argmax()
            print "prediction: ", guess
            total_pred.append((guess, labels[0]))

            if int(guess) == int(labels[0]):
                acc += 1

            # END IF

            all_labels.append(labels[0])
            all_preds.append(guess)

            #np.save(experiment_name+'_rate_pred.npy', np.array([all_labels, all_preds]))
            curr_logger.add_scalar_value('test/acc',acc/float(count), step=count)

        # END FOR

    # END WITH

    print "Total accuracy : ", acc/float(count)
    print total_pred


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', action= 'store', required=True,
            help= 'Model architecture (c3d, lrcn, tsn, vgg16, resnet)')

    parser.add_argument('--dataset', action= 'store', required=True,
            help= 'Dataset (UCF101, HMDB51)')

    parser.add_argument('--numGpus', action= 'store', type=int,
            help = 'Number of Gpus used for calculation')

    parser.add_argument('--train', action= 'store', required=True, type=int,
            help = 'Binary value to indicate training or evaluation instance')

    parser.add_argument('--load', action='store', type=int,
            help = 'Whether you want to load a saved model to train from scratch.')

    parser.add_argument('--size', action='store', required=True, type=int,
            help = 'Input frame size')

    parser.add_argument('--inputDims', action='store', required=True, type=int,
            help = 'Input dimensions (number of frames to pass as input to the model)')

    parser.add_argument('--outputDims', action='store', required=True, type=int,
            help = 'Output Dimensions (number of classes in dataset)')

    parser.add_argument('--seqLength', action='store', required=True, type=int,
            help = 'Length of sequences for LSTM')

    parser.add_argument('--expName', action='store', required=True,
            help = 'Unique name of experiment being run')

    parser.add_argument('--numVids', action='store', required=True, type=int,
            help = 'Number of videos from training set to train on')

    parser.add_argument('--lr', action='store', type=float,
            help = 'Learning Rate')

    parser.add_argument('--wd', action='store', type=float,
            help = 'Weight Decay')

    parser.add_argument('--nEpochs', action='store', type=int,
            help = 'Number of Epochs')

    parser.add_argument('--split', action='store', type=int,
            help = 'Dataset split to use')

    parser.add_argument('--baseDataPath', action='store', required=True,
            help = 'Path to datasets')

    parser.add_argument('--fName', action='store',
            help = 'Which dataset list to use (trainlist, testlist, vallist)')

    parser.add_argument('--saveFreq', action='store', type=int,
            help = 'Frequency in epochs to save model checkpoints')

    parser.add_argument('--valFreq', action='store', type=int,
            help = 'Frequency in epochs to validate')

    parser.add_argument('--loadedDataset', action= 'store', default='HMDB51',
            help= 'Dataset (UCF101, HMDB51)')

    args = parser.parse_args()

    print "Setup of current experiment: ", args

    modelName = args.model

    # Associating Models
    if modelName=='lrcn':
        model = LRCN()

    elif modelName == 'vgg16':
        model = VGG16()

    elif modelName == 'resnet':
        model = ResNet()

    elif modelName == 'resnet_RIL_interp_mean_v1':
        model = ResNet_RIL_Interp_Mean_v1()

    elif modelName == 'resnet_RIL_interp_meanv2':
        model = ResNet_RIL_Interp_Mean_v2()

    elif modelName == 'resnet_RIL_interp_meanv3':
        model = ResNet_RIL_Interp_Mean_v3()

    elif modelName == 'resnet_RIL_interp_max_v1':
        model = ResNet_RIL_Interp_Max_v1()

    elif modelName == 'resnet_RIL_interp_max_v2':
        model = ResNet_RIL_Interp_Max_v2()

    elif modelName == 'resnet_RIL_interp_max_v3':
        model = ResNet_RIL_Interp_Max_v3()

    elif modelName == 'resnet_RIL_interp_median_v1':
        model = ResNet_RIL_Interp_Median_v1()

    elif modelName == 'resnet_RIL_interp_median_v2':
        model = ResNet_RIL_Interp_Median_v2()

    elif modelName == 'resnet_RIL_interp_median_v3':
        model = ResNet_RIL_Interp_Median_v3()

    else:
        print("Model not found")
    
    # END IF



    if args.train:
        train(  model              = model,
                inputDims          = args.inputDims,
                outputDims         = args.outputDims,
                seqLength          = args.seqLength,
                size               = [args.size, args.size],
                numGpus            = args.numGpus,
                dataset            = args.dataset,
                experiment_name    = args.expName,
                loadModel          = args.load,
                numVids            = args.numVids,
                nEpochs            = args.nEpochs,
                split              = args.split,
                baseDataPath       = args.baseDataPath,
                fName              = args.fName,
                learning_rate_init = args.lr,
                wd                 = args.wd,
                save_freq          = args.saveFreq,
                val_freq           = args.valFreq)

    else:
        test(   model           = model,
                input_dims      = args.inputDims,
                output_dims     = args.outputDims,
                seq_length      = args.seqLength,
                size            = [args.size, args.size],
                dataset         = args.dataset,
                loaded_dataset  = args.loadedDataset,
                experiment_name = args.expName,
                num_vids        = args.numVids,
                split           = args.split,
                base_data_path  = args.baseDataPath,
                f_name          = args.fName)

    # END IF
