# Basic imports
import os
import time
import argparse
import tensorflow      as tf
import numpy           as np
import multiprocessing as mp
import matplotlib.pyplot as plt

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
from utils.load_dataset_tfrecords import load_dataset_without_preprocessing


parser = argparse.ArgumentParser()

# Experiment parameters

parser.add_argument('--dataset', action= 'store', required=True,
        help= 'Dataset (UCF101, HMDB51)')

parser.add_argument('--numVids', action='store', required=True, type=int,
        help = 'Number of videos to be used for training')

parser.add_argument('--split', action='store', type=int, default=1,
        help = 'Dataset split to use')

parser.add_argument('--baseDataPath', action='store', default='/z/dat',
        help = 'Path to datasets')

parser.add_argument('--fName', action='store', required=True,
        help = 'Which dataset list to use (trainlist, testlist, vallist)')

parser.add_argument('--vidName', action='store', default='default',
        help = 'Name of video to load')


parser.add_argument('--verbose', action='store', type=int, default=1,
        help = 'Boolean switch to display all print statements or not')


args = parser.parse_args()

if args.verbose:
    print "Setup of current experiments"
    print "\n############################"
    print args
    print "############################ \n"

# END IF


def example_video(dataset, num_vids, split, base_data_path, f_name, vid_name, verbose):
    """
    Function used to test the performance and analyse a chosen model
    Args:
        :dataset:            Name of dataset being loaded
        :num_vids:           Number of videos to be used for training
        :split:              Split of dataset being used
        :base_data_path:     Full path to root directory containing datasets
        :f_name:             Specific video directory within a chosen split of a dataset
        :vid_name:           Name of video to load if desired
        :verbose:            Boolean to indicate if all print statement should be procesed or not


    Returns:
        Does not return anything
    """

    with tf.name_scope("my_scope") as scope:

        # Initialize model variables
        istraining       = False


        data_path   = os.path.join(base_data_path, 'tfrecords_'+dataset, 'Split'+str(split), f_name)

        # Setting up tensors for models
        input_data_tensor, labels_tensor, names_tensor = load_dataset_without_preprocessing(data_path, dataset, istraining, vid_name, verbose)

        # TF session setup
        config  = tf.ConfigProto(allow_soft_placement=True)
        sess    = tf.Session(config=config)
        init    = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord   = tf.train.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

        # Variables get randomly initialized into tf graph
        sess.run(init)


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
            output, labels, names = sess.run([input_data_tensor, labels_tensor, names_tensor])

            import pdb; pdb.set_trace()

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


            # END IF

        # END WHILE

        #########################################################################################################################################################

    # END WITH

    coord.request_stop()
    coord.join(threads)


if __name__=="__main__":

    example_video(  dataset           = args.dataset,
                        num_vids          = args.numVids,
                        split             = args.split,
                        base_data_path    = args.baseDataPath,
                        f_name            = args.fName,
                        vid_name          = args.vidName,
                        verbose           = args.verbose)

    # END IF

    import pdb; pdb.set_trace()
