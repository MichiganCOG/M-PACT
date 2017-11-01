
import tensorflow as tf
import numpy as np
import os

from models.lrcn.lrcn_model import LRCN
from models.vgg16.vgg16_model import VGG16
from models.resnet.resnet_model import ResNet

from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.ops import init_ops

from load_dataset import load_dataset
import argparse
from logger import Logger
import time
from tensorflow.python.ops import clip_ops
from Queue import Queue
from utils import *

def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
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

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def load_video_list(dataset, modelName, experiment_name, fName, split, numVids):
    try:

        vidList = np.load(os.path.join('results',modelName, experiment_name+'_'+dataset, 'checkpoints/vidList.npy'))
        print "loaded vidlist"
        print vidList
    except:
        vidFile = open(os.path.join('datasets',dataset,'trainlist0'+str(split)+'.txt'),'r')
        lines = vidFile.readlines()
        num_vids = len(lines)
        vidFile.close()
        vidList = np.arange(num_vids)
        if numVids == num_vids:
            np.random.shuffle(vidList)
        else:
            vidList=vidList[:numVids]

    return vidList

""" Function used to read in the total number of videos in a given dataset split and shuffle the indices """
def gen_video_list(dataset, modelName, experiment_name, fName, split, numVids, shuffle, epoch):

    if epoch == 0:
        vidFile  = open(os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'),'r')
        lines    = vidFile.readlines()
        num_vids = len(lines)
        vidFile.close()

        vidList = np.arange(num_vids)

        if numVids != None:
            if numVids != num_vids:
                vidList=vidList[:numVids]
    else:
        vidList = np.arange(numVids)

    if shuffle:
        np.random.shuffle(vidList)

    return vidList

def save_video_list(modelName, videoList, iterated_list, dataset, split, experiment_name):
    if not np.array_equal(iterated_list,videoList):
        lastVidUsed = iterated_list[-1]
        lastVidInd = np.where(videoList == iterated_list[-1])[0][0]
        vidFile = videoList[lastVidInd+1:]
        if len(vidFile) != 0:
            np.save(os.path.join('results',modelName, experiment_name+'_'+dataset, 'checkpoints/vidList.npy'), vidFile)



def _validate(model, slogits, sess, experiment_name, logger, dataset, inputDims, outputDims, split, gs, size, x_placeholder, baseDataPath):

    if dataset == 'HMDB51':
        fName = 'vallist'
    else:
        fName = 'testlist'

    vidList = gen_video_list(dataset, model.name, experiment_name, fName, split, None, False, 0)
    count   = 0
    acc     = 0

    for vidNum in vidList:
        count +=1

        loaded_data, labels= load_dataset( model, vidNum, fName, os.path.join(baseDataPath, dataset+'HDF5RGB','Split'+str(split)), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, False, dataset)
        shape = np.array(loaded_data).shape
        labels = np.repeat(labels, inputDims)

        if len(shape) ==4:

            loaded_data = np.array([loaded_data])
            numClips = 1
        else:
            numClips = len(loaded_data)

        output_predictions = np.zeros((numClips, outputDims))

        for clip in range(numClips):
            input_data= np.reshape(loaded_data[clip], (1, -1, size[0], size[1], 3))
            labels = np.reshape(labels, (1, -1))

            pred = sess.run(slogits, feed_dict={x_placeholder: input_data})
            output_predictions[clip] = np.mean(pred, 0)

        guess = np.mean(output_predictions, 0).argmax()

        if int(guess) == int(labels[0][0]):
            acc += 1

        print "validation acc: ", acc/float(count)
        logger.add_scalar_value('val/gs'+str(gs)+'_step_acc',acc/float(count), step=count)

    logger.add_scalar_value('val/acc',acc/float(count), step=gs)






def train(model, inputDims, outputDims, seqLength, size, numGpus, dataset, experiment_name, loadModel, numVids, nEpochs, split, baseDataPath, fName, learning_rate_init=0.001, wd=0.0, save_freq = 5, val_freq = 1):

    with tf.name_scope("my_scope") as scope:
        isTraining = True
        global_step = tf.Variable(0, name='global_step', trainable=False)
        reuse_variables = None

        # Setting up placeholders for models
        x_placeholder = tf.placeholder(tf.float32, shape=[numGpus, inputDims, size[0], size[1] ,3], name='x_placeholder')
        y_placeholder = tf.placeholder(tf.int64, shape=[numGpus, seqLength], name='y_placeholder')

        tower_losses = []
        tower_grads = []

        # Define Optimizer
        optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        for gpuIdx in range(numGpus):
            with tf.device('/gpu:'+str(gpuIdx)):

                with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):

                    logits = model.inference(x_placeholder[gpuIdx,:,:,:,:], True, inputDims, outputDims, seqLength, scope, weight_decay=wd, cpuId = gpuIdx)

                    # Calculating Softmax for probability outcomes : Can be modified
                    # Make function internal to model
                    slogits = tf.nn.softmax(logits)

                    # Why retain learning rate here ?
                    lr = vs.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(learning_rate_init))

                reuse_variables = True

                """ Within GPU mini-batch: 1) Calculate loss,
                                           2) Initialize optimizer with required learning rate and
                                           3) Compute gradients
                """
                total_loss = model.loss(logits, y_placeholder[gpuIdx, :])
                opt = optimizer(lr)
                gradients = opt.compute_gradients(total_loss, vars_.trainable_variables())

        """  After: 1) Computing gradients and losses need to be stored and averaged
                    2) Clip gradients by norm to required value
                    3) Apply mean gradient updates
        """
        gradients, variables = zip(*gradients)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 5.0)
        gradients = list(zip(clipped_gradients, variables))
        grad_updates = opt.apply_gradients(gradients, global_step=global_step, name="train")
        train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)


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

        vidList = []

        if loadModel:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name,   experiment_name+ '_'+dataset, 'checkpoints/checkpoint')))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)



        total_pred = []

        count     = 0
        losses    = []
        acc       = 0
        tot_step  = 0
        save_data = []

        lr = learning_rate_init

        tot_train_time = []
        tot_load_time  = []
        iterated_list  = []

        # Timing test setup
        time_init = time.time()

        for epoch in range(nEpochs):
            batch_count = 0
            epoch_acc = 0

            # Generating the video list
            if len(vidList) == 0 or epoch != 0:
                vidList = gen_video_list(dataset, model.name, experiment_name, fName, split, numVids, True, epoch)

            for vidNum in vidList:
                mean_loss=[]

                # Setting up load time checks
                time_pre_load = time.time()

                loaded_data, labels= load_dataset(model, vidNum, fName, os.path.join(baseDataPath, dataset+'HDF5RGB','Split'+str(split)), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, isTraining, dataset)

                time_post_load = time.time()
                tot_load_time.append((time_post_load - time_pre_load))

                shape  = np.array(loaded_data).shape
                labels = np.repeat(labels, seqLength)
                print "vidNum: ", vidNum
                print "label: ", labels[0]

                if len(shape) ==4:
                    loaded_data = np.array([loaded_data])
                    numClips = 1
                else:
                    numClips = len(loaded_data)

                output_predictions = np.zeros((numClips, outputDims))

                for clip in range(numClips):
                    input_data= np.reshape(loaded_data[clip], (1, -1, size[0], size[1], 3))
                    labels = np.reshape(labels, (1, -1))

                    time_pre_train = time.time()

                    _, loss_train, pred, gs = sess.run([train_op, total_loss, slogits, global_step], feed_dict={x_placeholder: input_data, y_placeholder: labels})

                    output_predictions[clip] = np.mean(pred, 0)
                    tot_step+=1
                    mean_loss.append(loss_train)
                    pred = np.mean(pred, 0).argmax()
                    print "pred: ", pred

                    if pred == labels[0][0]:
                        acc +=1

                    time_post_train = time.time()
                    count+=1
                    curr_logger.add_scalar_value('train/total_acc', acc/float(count), step=gs)
                    print "step  loss  acc: ", gs, loss_train, acc/float(count)

                    tot_train_time.append((time_post_train - time_pre_train))

                iterated_list.append(vidNum)
                batch_count+=1

                guess = np.mean(output_predictions, 0).argmax()

                if int(guess) == int(labels[0][0]):
                    epoch_acc += 1
                curr_logger.add_scalar_value('train/loss', float(np.mean(mean_loss)), step=gs)
                curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(batch_count), step=gs)
                curr_logger.add_scalar_value('train_time',time_post_train - time_pre_train, step=gs)
                curr_logger.add_scalar_value('load_time',time_post_load - time_pre_load, step=gs)

            if epoch % save_freq == 0:
                saver.save(sess, os.path.join('results',model.name, experiment_name + '_'+dataset,'checkpoints/checkpoint'), global_step.eval(session=sess))
                print "Saving..."
                print "Curr time: ", time.time()-time_init
                print "Tot load time: ", np.sum(np.array(tot_load_time))
                print "Tot train time: ", np.sum(np.array(tot_train_time))

            if epoch % val_freq == 0:
                _validate(model, slogits, sess, experiment_name, curr_logger, dataset, inputDims, outputDims, split, gs, size, x_placeholder, baseDataPath)

            print "Tot load time: ", np.sum(np.array(tot_load_time))

            print "Tot train time: ", np.sum(np.array(tot_train_time))
            print "Tot time: ", (time.time()-time_init)






def test(model, inputDims, outputDims, seqLength, size, dataset, experiment_name, numVids, split, baseDataPath, fName='testlist'):

    with tf.name_scope("my_scope") as scope:
        isTraining = False

        x_placeholder = tf.placeholder(tf.float32, shape=[inputDims, size[0], size[1] ,3], name='x_placeholder')
        y_placeholder = tf.placeholder(tf.int64, shape=[inputDims], name='y_placeholder')
        global_step   = tf.Variable(0, name='global_step', trainable=False)

        # Model Inference
        logits = model.inference(x_placeholder, False, inputDims, outputDims, seqLength, scope)

        # Logits
        softmax  = tf.nn.softmax(logits)

        # Logger setup
        log_name = ("exp_test_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name) )
        curr_logger = Logger(os.path.join('logs',model.name,dataset, log_name))

        # Session setup
        sess  = tf.Session()
        saver = tf.train.Saver()

        # Initialize Variables
        init = tf.global_variables_initializer()
        sess.run(init)


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name,   experiment_name+ '_'+dataset, 'checkpoints/checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)

        total_pred = []
        acc        = 0
        count      = 0

        acc        = 0
        count      = 0
        total_pred = []

        vidList = gen_video_list(dataset, model.name, experiment_name, fName, split, numVids, False, 0)

        for vidNum in vidList:
            count +=1
            loaded_data, labels= load_dataset(model, vidNum, fName, os.path.join(baseDataPath, dataset+'HDF5RGB', 'Split'+str(split)), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, isTraining, dataset)

            labels = np.repeat(labels, inputDims)

            print "vidNum: ", vidNum
            print "label: ", labels[0]

            if len(np.array(loaded_data).shape) ==4:
                loaded_data = np.array([loaded_data])
                numClips = 1
            else:
                numClips = len(loaded_data)
            output_predictions = np.zeros((len(loaded_data), outputDims))

            for ldata in range(numClips):

                input_data = loaded_data[ldata]

                prediction = softmax.eval(session=sess, feed_dict={x_placeholder: input_data})
                output_predictions[ldata] = np.mean(prediction, 0)

            guess = np.mean(output_predictions, 0).argmax()
            print "prediction: ", guess

            total_pred.append((guess, labels[0]))
            if int(guess) == int(labels[0]):
                acc += 1
            print "acc val: ", acc
            print "count val: ", count
            print "acc: ", acc/float(count)
            curr_logger.add_scalar_value('test/acc',acc/float(count), step=count)


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

    else:
        print("Model not found")




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
        test(   model = model,
                inputDims       = args.inputDims,
                outputDims      = args.outputDims,
                seqLength       = args.seqLength,
                size            = [args.size, args.size],
                dataset         = args.dataset,
                experiment_name = args.expName,
                numVids         = args.numVids,
                split           = args.split,
                baseDataPath    = args.baseDataPath,
                fName           = args.fName)
