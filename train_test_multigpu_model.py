import os
import time
import argparse
import tensorflow      as tf
import numpy           as np
import multiprocessing as mp

from Queue import Queue

from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as vars_
from tensorflow.python.ops import variable_scope as vs

from load_dataset import load_dataset
from logger       import Logger
from utils        import *

from models.lrcn.lrcn_model             import LRCN
from models.vgg16.vgg16_model           import VGG16
from models.resnet.resnet_model         import ResNet
from models.resnet_RIL.resnet_RIL_model import ResNet_RIL


def _average_gradients(tower_grads):

    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []
        curr_tower = 0

        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
            curr_tower += 1

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def gen_video_list(dataset, model_name, experiment_name, f_name, split, num_vids, shuffle, epoch):

    if epoch == 0 or num_vids == None:
        vid_file       = open(os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'),'r')
        lines          = vid_file.readlines()
        num_total_vids = len(lines)
        vid_file.close()

        if num_total_vids != None:
            vid_list = np.arange(num_total_vids).tolist()

            if num_total_vids != num_vids:
                vid_list=vid_list[:num_vids]

        else:
            print "No videos in list provided. Please check!"
            exit()

    else:
        vid_list = np.arange(num_vids).tolist()

    if shuffle:
        np.random.shuffle(vid_list)

    return vid_list


def load_video_into_queue(x_q, y_q, model, vid_list, num_gpus, f_name, size, base_data_path, dataset, split, input_dims, seq_length, is_training=False):

    if model.name == 'lrcn':
        num_vids_to_load = 1

    else:
        num_vids_to_load = num_gpus

    if num_vids_to_load > len(vid_list):
        num_vids_to_load = len(vid_list)

    # Legacy load setup : Can be modified
    for gpu_count in range(num_vids_to_load):
        if len(vid_list) != 0:
           vid_num = vid_list[gpu_count]

        else:
            break

        loaded_data, labels= load_dataset(model, vid_num, f_name,
                             os.path.join(base_data_path,
                             dataset+'HDF5RGB','Split'+str(split)),
                             os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'),
                             os.path.join("datasets",dataset,"classInd.txt"), size, is_training,
                             dataset)

        labels = np.repeat(labels, seq_length)
        shape  = np.array(loaded_data).shape

        if len(shape) ==4:
            loaded_data = np.array([loaded_data])
            num_clips = 1

        else:
            num_clips = len(loaded_data)

        for clip in range(num_clips):
            x_q.put(loaded_data[clip])
            y_q.put(labels)

    #-----------------------------------------------#

    if len(vid_list) >= num_vids_to_load:
        vid_list = vid_list[num_vids_to_load:]

    else:
        vid_list = []

    return x_q, y_q, vid_list


def _validate(model, tower_slogits, sess, experiment_name, logger, dataset, input_dims, output_dims, split, gs, size, x_placeholder, istraining_placeholder, j_placeholder, K, base_data_path, num_gpus, seq_length, x_q, y_q):

    if dataset == 'HMDB51' or dataset == "HMDB51Rate":
        f_name = 'vallist'

    else:
        f_name = 'testlist'

    vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, None, False, 0)

    count   = 0
    acc     = 0

    finish_val = False

    while not finish_val:
        # Load video into Q concurrently
        x_q, y_q, vid_list = load_video_into_queue(x_q, y_q, model, vid_list, num_gpus, f_name,
                             size, base_data_path, dataset, split, input_dims, seq_length, False)

        input_data = np.zeros((num_gpus, input_dims, size[0], size[1], 3))
        labels     = np.zeros((num_gpus, seq_length))

        excess_clips      = 0
        last_vid_index    = 0
        intra_batch_count = 0

        # To avoid doing an extra loop after data has been completely loaded and trained upon
        if len(vid_list) == 0 and x_q.qsize() == 0:
            break

        for gpu_count in range(num_gpus):
            if x_q.qsize() > 0:
                input_data[gpu_count] = x_q.get()
                labels[gpu_count]     = y_q.get()
                last_vid_index        = gpu_count
                intra_batch_count    += 1

            else:
                finish_val              = True
                input_data[gpu_count]   = input_data[last_vid_index]
                labels[gpu_count]       = labels[last_vid_index]

        count +=intra_batch_count

        predictions = sess.run([tower_slogits],
                               feed_dict={x_placeholder: input_data,
                               istraining_placeholder  : False,
                               j_placeholder           : [input_data.shape[1]/K]})

        # For ResNet and VGG16 based setup only : Need to add support for LRCN multi-GPU validation
        # ------------------------------------------------

        for acc_count in range(intra_batch_count):
            if (last_vid_index > acc_count) and finish_val:
                break
            guess = np.mean(predictions[0][acc_count]).argmax()

            if int(guess) == int(labels[acc_count][0]):
                acc+=1

        # --------------------------------------------------

        logger.add_scalar_value('val/gs'+str(gs)+'_step_acc',acc/float(count), step=count)

    logger.add_scalar_value('val/acc',acc/float(count), step=gs)


def train(model, input_dims, output_dims, seq_length, size, num_gpus, dataset, experiment_name, load_model, num_vids, n_epochs, split, base_data_path, f_name, learning_rate_init, wd, save_freq, val_freq, k=25):

    with tf.name_scope("my_scope") as scope:
        is_training = True
        global_step = tf.Variable(0, name='global_step', trainable=False)
        reuse_variables = None

        # Setting up placeholders for models
        x_placeholder          = tf.placeholder(tf.float32,
                                 shape=[num_gpus, input_dims,  size[0], size[1] ,3],
                                 name='x_placeholder')

        y_placeholder          = tf.placeholder(tf.int64,
                                 shape=[num_gpus, seq_length],
                                 name='y_placeholder')

        istraining_placeholder = tf.placeholder(tf.bool,
                                 name='istraining_placeholder')

        j_placeholder          = tf.placeholder(tf.int32,
                                 shape=[1],
                                 name='j_placeholder')

        tower_losses  = []
        tower_grads   = []
        tower_slogits = []

        # Define optimizer
        optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:'+str(gpu_idx)):
                with tf.name_scope('%s_%d' % ('tower', gpu_idx)) as scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
                        logits = model.inference(x_placeholder[gpu_idx,:,:,:,:],
                                                 istraining_placeholder,
                                                 input_dims,
                                                 output_dims,
                                                 seq_length,
                                                 scope, k,
                                                 j_placeholder,
                                                 weight_decay=wd,
                                                 cpuId = gpu_idx) # cpuId!! : Can be modified

                        # Calculating softmax for probability outcomes : Can be modified
                        slogits = tf.nn.softmax(logits)

                        lr = vs.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(learning_rate_init))
                        reuse_variables = True

                        """ Within GPU mini-batch: 1) Calculate loss,
                                                   2) Initialize optimizer with required learning rate and
                                                   3) Compute gradients
                                                   4) Aggregate losses, gradients and logits
                        """
                        total_loss = model.loss(logits, y_placeholder[gpu_idx, :])
                        opt = optimizer(lr)
                        gradients = opt.compute_gradients(total_loss, vars_.trainable_variables())

                        tower_losses.append(total_loss)
                        tower_grads.append(gradients)
                        tower_slogits.append(slogits)

        """  After: 1) Computing gradients and losses need to be stored and averaged
                    2) Clip gradients by norm to required value
                    3) Apply mean gradient updates
        """

        gradients            = _average_gradients(tower_grads)
        gradients, variables = zip(*gradients)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 5.0)
        gradients            = list(zip(clipped_gradients, variables))
        grad_updates         = opt.apply_gradients(gradients, global_step=global_step, name="train")
        train_op             = grad_updates


        # Logging setup initialization
        log_name     = ("exp_train_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name))
        make_dir(os.path.join('results',model.name, dataset))
        make_dir(os.path.join('results',model.name, dataset, experiment_name))
        make_dir(os.path.join('results',model.name, dataset, experiment_name, 'checkpoints'))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, log_name))

        # TF session setup
        config = tf.ConfigProto(allow_soft_placement=True)
        sess   = tf.Session(config=config)
        saver  = tf.train.Saver()
        init   = tf.global_variables_initializer()

        sess.run(init)

        vid_list = []

        if load_model:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name, dataset,  experiment_name, 'checkpoints/checkpoint')))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)


        count          = 0
        acc            = 0
        tot_train_time = 0.0
        tot_load_time  = 0.0

        losses     = []
        total_pred = []
        save_data  = []

        # Data loading setup
        x_q = Queue()
        y_q = Queue()

        # Timing test setup
        time_init = time.time()

        for epoch in range(n_epochs):
            batch_count = 0
            epoch_acc   = 0

            if len(vid_list) == 0 or epoch != 0:
                vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, num_vids, True, epoch)

            finish_epoch = False

            while not finish_epoch:
                # Load video into Q concurrently
                time_pre_load = time.time()
                x_q, y_q, vid_list = load_video_into_queue(x_q, y_q, model, vid_list, num_gpus, f_name, size, base_data_path, dataset, split, input_dims, seq_length, True)
                time_post_load = time.time()


                input_data = np.zeros((num_gpus, input_dims, size[0], size[1], 3))
                labels     = np.zeros((num_gpus, seq_length))

                excess_clips  = 0
                last_vid_index = 0

                intra_batch_count = 0

                # To avoid doing an extra loop after data has been completely loaded and trained upon
                if len(vid_list) == 0 and x_q.qsize() == 0:
                    break

                for gpu_count in range(num_gpus):
                    if x_q.qsize() > 0:
                        input_data[gpu_count] = x_q.get()
                        labels[gpu_count]     = y_q.get()
                        last_vid_index        = gpu_count
                        intra_batch_count    += 1
                    else:
                        finish_epoch          = True
                        input_data[gpu_count] = input_data[last_vid_index]
                        labels[gpu_count]     = labels[last_vid_index]

                batch_count+= intra_batch_count

                time_pre_train = time.time()
                _, loss_train, predictions, gs = sess.run([train_op, tower_losses,
                                                 tower_slogits, global_step],
                                                 feed_dict={x_placeholder: input_data,
                                                 y_placeholder         : labels,
                                                 istraining_placeholder: True,
                                                 j_placeholder         : [input_data.shape[1]/k]})

                time_post_train = time.time()

                for pred_idx in range(intra_batch_count):
                    pred = np.mean(predictions[pred_idx], 0).argmax()
                    if pred == labels[pred_idx][0]:
                        epoch_acc +=1

                tot_train_time += time_post_train - time_pre_train
                tot_load_time  += time_post_load  - time_pre_load

                curr_logger.add_scalar_value('load_time',       time_post_load - time_pre_load, step=gs)
                curr_logger.add_scalar_value('train/train_time',time_post_train - time_pre_train, step=gs)
                curr_logger.add_scalar_value('train/loss',      float(np.mean(loss_train)), step=gs)

            curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(batch_count), step=gs)

            if epoch % save_freq == 0:
                print "Saving..."
                saver.save(sess, os.path.join('results', model.name, dataset, experiment_name,'checkpoints/checkpoint'), global_step.eval(session=sess))

            if epoch % val_freq == 0:
                _validate(model, tower_slogits, sess, experiment_name, curr_logger, dataset, input_dims, output_dims, split, gs, size, x_placeholder, istraining_placeholder, j_placeholder, k, base_data_path, num_gpus, seq_length, x_q, y_q)

        print "Tot load time:  ", tot_load_time
        print "Tot train time: ", tot_train_time
        print "Tot time:       ", time.time()-time_init


def test(model, input_dims, output_dims, seq_length, size, num_gpus, dataset, experiment_name, num_vids, split, base_data_path, f_name, k=25):

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
                                 scope, k,
                                 j_placeholder)

        # Logits
        softmax = tf.nn.softmax(logits)

        # Logger setup
        log_name     = ("exp_test_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name))
        curr_logger = Logger(os.path.join('logs',model.name,dataset, log_name))

        # Initialize Variables
        sess  = tf.Session()
        saver = tf.train.Saver()
        init  = tf.global_variables_initializer()
        sess.run(init)


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name, dataset, experiment_name, 'checkpoints/checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)

        total_pred = []
        acc        = 0
        count      = 0

        vid_list = gen_video_list(dataset, model.name, experiment_name, f_name, split, num_vids, False, 0)

        for vid_num in vid_list:
            count +=1
            loaded_data, labels= load_dataset(model, vid_num, f_name,
                                 os.path.join(base_data_path, dataset+'HDF5RGB', 'Split'+str(split))
                               , os.path.join('datasets',dataset,f_name+'0'+str(split)+'.txt'),
                                 os.path.join("datasets",dataset,"classInd.txt"),
                                 size, is_training, dataset)

            labels = np.repeat(labels, seq_length)

            print "vidNum: ", vid_num
            print "label:  ", labels[0]

            if len(np.array(loaded_data).shape) ==4:
                loaded_data = np.array([loaded_data])
                num_clips = 1

            else:
                num_clips = len(loaded_data)

            output_predictions = np.zeros((len(loaded_data), output_dims))

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

        curr_logger.add_scalar_value('test/acc',acc/float(count), step=count)

        print "Total accuracy : ", acc/float(count)
        print total_pred

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
            help = 'Unique name of experiment being run')

    parser.add_argument('--lr', action='store', type=float, default=0.001,
            help = 'Learning Rate')

    parser.add_argument('--wd', action='store', type=float, default=0.0,
            help = 'Weight Decay')

    parser.add_argument('--nEpochs', action='store', type=int, default=1,
            help = 'Number of Epochs')

    parser.add_argument('--split', action='store', type=int, default=1,
            help = 'Dataset split to use')

    parser.add_argument('--baseDataPath', action='store', default='/z/home/madantrg/Datasets',
            help = 'Path to datasets')

    parser.add_argument('--fName', action='store',
            help = 'Which dataset list to use (trainlist, testlist, vallist)')

    parser.add_argument('--saveFreq', action='store', type=int, default=1,
            help = 'Frequency in epochs to save model checkpoints')

    parser.add_argument('--valFreq', action='store', type=int, default=3,
            help = 'Frequency in epochs to validate')

    args = parser.parse_args()

    print "Setup of current experiments: ",args
    model_name = args.model

    # Associating models
    if model_name=='lrcn':
        model = LRCN()

    elif model_name == 'vgg16':
        model = VGG16()

    elif model_name == 'resnet':
        model = ResNet()

    elif model_name == 'resnet_RIL':
        model = ResNet_RIL()

    else:
        print("Model not found")

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
                val_freq            = args.valFreq)

    else:
        test(   model           = model,
                input_dims      = args.inputDims,
                output_dims     = args.outputDims,
                seq_length      = args.seqLength,
                size            = [args.size, args.size],
                num_gpus        = args.numGpus,
                dataset         = args.dataset,
                experiment_name = args.expName,
                num_vids        = args.numVids,
                split           = args.split,
                base_data_path  = args.baseDataPath,
                f_name          = args.fName)
