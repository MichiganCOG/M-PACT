
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

def _average_gradients(tower_grads, clips_to_average, numGpus):
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
#    numGpusC = tf.constant(numGpus)
    #gather_index = tf.Variable()
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        currTower = 0
        for g, _ in grad_and_vars:
        #    import pdb;pdb.set_trace()
            # If videos were processed in the last iteration of the epoch
            # because there were more gpus in use than videos left, then
            # ignore the gradients of the excess videos
        #    if currTower < tf.gather_nd(clips_to_average, gather_index):
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            currTower += 1

        #TODO########################################################## Figure out how to split up the gradient averaging
        # Average over the 'tower' dimension.
        #import pdb;pdb.set_trace()
    #    grad = tf.cond(tf.equal(clips_to_average , numGpusC), lambda: tf.concat(axis=0, values=grads[:clips_to_average]), lambda: tf.concat(axis=0, values=grads))
        grad = tf.concat(axis=0, values=grads)
    #    grad = tf.gather(grad,tf.range(clips_to_average))
        grad = tf.reduce_mean(grad, 0)
        #import pdb;pdb.set_trace()
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
        vidFile = open(os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'),'r')
        lines = vidFile.readlines()
        num_vids = len(lines)
        vidFile.close()
        vidList = np.arange(num_vids)
        if numVids == num_vids:
            np.random.shuffle(vidList)
        else:
            vidList=vidList[:numVids]

    return vidList


def gen_video_list(dataset, modelName, experiment_name, fName, split, numVids, shuffle):


    vidFile = open(os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'),'r')
    lines = vidFile.readlines()
    num_vids = len(lines)
    vidFile.close()
    vidList = np.arange(num_vids)
    if numVids != None:
        if numVids != num_vids:
            vidList=vidList[:numVids]
    if shuffle:
        np.random.shuffle(vidList)
    return vidList

def save_video_list(modelName, vidFile, dataset, split, experiment_name):
    if len(vidFile) != 0:
        np.save(os.path.join('results',modelName, experiment_name+'_'+dataset, 'checkpoints/vidList.npy'), vidFile)



def _validate(model, slogits, sess, experiment_name, logger, dataset, inputDims, outputDims, split, gs, size, x_placeholder):
    if dataset == 'HMDB51':
        fName = 'vallist'
    else:
        fName = 'testlist'
    vidList = gen_video_list(dataset, model.name, experiment_name, fName, split, None, False)
    count =0
    acc=0
    for vidNum in vidList:

        count +=1

        loaded_data, labels= model.load_dataset( model, vidNum, fName, False, size, split, dataset)
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


def load_video_into_queue(x_q, y_q, model, vidList, numGpus):
    isTraining = True
    size = [224,224]
    time_pre_load = time.time()
    fName = 'trainlist'
    if model.name == 'lrcn':
        numVidsToLoad = 1
    else:
        numVidsToLoad = numGpus
    for gpuCount in range(numVidsToLoad):
        if vidList != []:
        #    import pdb;pdb.set_trace()
            vidNum = vidList[gpuCount]
        #    vidList = vidList[1:]
            print "Load vid ", vidNum
            if dataset == "Kinetics":
                loaded_data, labels= load_dataset(model, vidNum, fName, os.path.join(baseDataPath, dataset+'HDF5RGB'), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, isTraining, dataset)
            else:
                loaded_data, labels= load_dataset(model, vidNum, fName, os.path.join(baseDataPath, dataset+'HDF5RGB','Split'+str(split)), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, isTraining, dataset)
            labels = np.repeat(labels, inputDims)
        #    import pdb;pdb.set_trace()

            shape = np.array(loaded_data).shape

        #    print "vidNum: ", vidNum
        #    print "label: ", labels[0]
            if len(shape) ==4:
                loaded_data = np.array([loaded_data])
                numClips = 1
            else:
                numClips = len(loaded_data)

            for clip in range(numClips):
                x_q.put(loaded_data[clip])
                y_q.put(labels)
    if len(vidList) >= numVidsToLoad:
        vidList = vidList[numVidsToLoad:]
    else:
        vidList = []
        # ignoreClipsInd = numGpus
        # if vidList == [] and x_q.qsize()%numGpus != 0:
        #     for emptyVid in range(numGpus - x_q.qsize()%numGpus):
        #         x_q.put(np.zeros(inputDims, size[0], size[1], 3))
        #         y_q.put(np.ones(inputDims)-1)
        # ignoreClipsInd = numGpus - x_q.qsize()%numGpus
    return x_q, y_q, vidList



def train(model, inputDims, outputDims, seqLength, size, numGpus, dataset, experiment_name, loadModel, numVids, nEpochs, split, baseDataPath, fName, learning_rate_init=0.001, wd=0.0):
    with tf.name_scope("my_scope") as scope:
        isTraining = True
        global_step = tf.Variable(0, name='global_step', trainable=False)
        clips_to_average = tf.placeholder(tf.int32, shape = [], name = 'clips_to_average')
        reuse_variables = None
        if model.name == "resnet" or model.name == 'vgg16':
            seqLength = 2*seqLength
            x_placeholder = tf.placeholder(tf.float32, shape=[numGpus, inputDims*2, size[0], size[1] ,3], name='x_placeholder')
        else:
            x_placeholder = tf.placeholder(tf.float32, shape=[numGpus, inputDims, size[0], size[1] ,3], name='x_placeholder')
        y_placeholder = tf.placeholder(tf.int64, shape=[numGpus, inputDims], name='y_placeholder')
        tower_losses = []
        tower_grads = []
        tower_slogits = []
        optimizer = lambda lr: tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        for gpuIdx in range(numGpus):
            with tf.device('/gpu:'+str(gpuIdx)):
                with tf.name_scope('%s_%d' % ('tower', gpuIdx)) as scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):

                        if model.name=='resnet' or model.name == 'vgg16':
                            logits = model.inference(x_placeholder[gpuIdx,:,:,:,:], y_placeholder[gpuIdx,:], True, inputDims*2, outputDims, seqLength, scope, weight_decay=wd, cpuId = gpuIdx)
                            logits = logits[:25]
                        else:
                            logits = model.inference(x_placeholder[gpuIdx,:,:,:,:], y_placeholder[gpuIdx,:], True, inputDims, outputDims, seqLength, scope, weight_decay=wd, cpuId = gpuIdx)
                        slogits = tf.nn.softmax(logits)
                        lr = vs.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(learning_rate_init))

                        reuse_variables = True
                        total_loss = model.loss(logits, y_placeholder[gpuIdx, :])
                        opt = optimizer(lr)
                        gradients = opt.compute_gradients(total_loss, vars_.trainable_variables())
                        tower_losses.append(total_loss)
                        tower_grads.append(gradients)
                        tower_slogits.append(slogits)

        gradients = _average_gradients(tower_grads, clips_to_average, numGpus)
        gradients, variables = zip(*gradients)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, 5.0)
        gradients = list(zip(clipped_gradients, variables))

        grad_updates = opt.apply_gradients(gradients, global_step=global_step, name="train")
    #    train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)
        train_op = grad_updates



        config = tf.ConfigProto(allow_soft_placement=True)
        log_name     = ("exp_train_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset, #args.dataset,
                                                           experiment_name) )#args.experiment_name)
        make_dir(os.path.join('results',model.name,   experiment_name+'_'+dataset))
        make_dir(os.path.join('results',model.name,   experiment_name+'_'+dataset, 'checkpoints'))
        curr_logger = Logger(os.path.join('Logs',model.name,dataset, log_name))
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        sess.run(init)
        split = 1
        epochVidList = []
        if loadModel:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name,   experiment_name+ '_'+dataset, 'checkpoints/checkpoint')))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)
        #        epochVidList = load_video_list(dataset, model.name, experiment_name, fName, split, numVids)



        total_pred = []

        count = 0
        losses = []
        acc = 0
        tot_step = 0
        save_data = []
        lr = learning_rate_init
        time_init = time.time()
        tot_train_time = []
        tot_load_time = []

        x_q = Queue()#tf.FIFOQueue(capacity=100, shapes = [inputDims, size[0], size[1] ,3], dtypes=tf.float32)
        y_q = Queue()#tf.FIFOQueue(capacity=100, shapes = [inputDims], dtypes=tf.int64)
        vidNum_q = Queue()
        data_q = Queue()
        iterated_list=[]
        for epoch in range(nEpochs):
            epoch_count = 0
            epoch_acc = 0

            if len(epochVidList) == 0 or epoch != 0:
                epochVidList = gen_video_list(dataset, model.name, experiment_name, fName, split, numVids, True)
            vidList = epochVidList.tolist()
    #need to be able to add multiple videos at the same time some kind of  for x in range(0, len(vidList), numGpus)      vidNumGpu0 = vidList[x] vidNumGpuCount = vidList[x+gpuCount]
    # check that vidlist[x+numGpus] exists, could be end of vidlist  aka len(vidlist)!%numGpus

            finished = False
            while not finished:#len(vidList) != 0:

                pre_load = time.time()
                vidsLoaded = len(vidList)
                while x_q.qsize() < numGpus and vidList != []:
                    x_q, y_q, vidList = load_video_into_queue(x_q, y_q, model, vidList, numGpus)
                vidsLoaded=vidsLoaded-len(vidList)
                load_time = time.time()-pre_load
        #        print "Loaded vids, time: ", vidsLoaded, load_time
                for _ in range(vidsLoaded):
                    tot_load_time.append(load_time/float(numGpus))
                print "Total Load Time, vidsLoaded: ", np.sum(np.array(tot_load_time)), vidsLoaded

    # x_q holds only clips
    # go through the queue and take out numGpus clips and concatenate them into the same list
    # Alternative is to concatenate the clips if loaded_data%numGpus is something, then do something else with remaining clips

                mean_loss=[]


                # Each element of x_q and y_q contains int(numGpus) clips that will get trained on
# check that if a clip has been in excessclips that the next loop will still calculate the next numclips properly
            #    for clipBatch in range(numClips):

            #    import pdb; pdb.set_trace()
                input_data = np.zeros((numGpus, inputDims*2, size[0], size[1], 3))

                labels = np.zeros((numGpus, inputDims))
                excessClips = 0
                lastVidIndex = 0
                for gpuCount in range(numGpus):
                    if x_q.qsize() > 0:
                        input_data[gpuCount] = x_q.get()
                        labels[gpuCount] = y_q.get()
                        lastVidIndex = gpuCount
                    else:
                        # Other option is to create variable that says there are excess values and to eval that above and make if then statement for _average_gradients?
                        # If there are not enough clips to fill each gpu, then just train the last clip multiple time_post_train
                        #excessClips+=1
                        finished = True
                        input_data[lastVidIndex] = input_data[gpuCount]
                        labels[lastVidIndex] = labels[gpuCount]
                #        input_data[gpuCount] = input_data[gpuCount-1]
                #        labels[gpuCount] = labels[gpuCount-1]
                clipsAvail = numGpus - excessClips
                pre_train = time.time()

            #    import pdb; pdb.set_trace()
                _, loss_val, predictions, gs = sess.run([train_op, tower_losses, tower_slogits, global_step], feed_dict={x_placeholder: input_data, y_placeholder: labels, clips_to_average: clipsAvail})#, learning_rate: lr})
                # lrcn pred = [2,16,101]  resnet pred = [2,50,51]
                if excessClips != 0:
                    gs = gs-excessClips
                    predictions = predictions[:-excessClips]
                    loss_val = loss_val[:-excessClips]
                    #sess.run(global_step.assign(gs)) #See if global step can be forced to not increment outside of clips_to_average somewhere around the def of train_op
                post_train = time.time()
        #        import pdb; pdb.set_trace()
        #        print "Predictions shape ", np.array(predictions).shape
        #        print "input data shape ", input_data.shape
        #        print "x placeholder shape ", x_placeholder.shape
                mean_loss.append(np.mean(loss_val))
                for pred in predictions:
                    # numClips = 1
                    # output_predictions = np.zeros((numClips, outputDims))
                    # output_predictions[numClips] = np.mean(pred, 0)
                    tot_step+=1


                    pred = np.mean(pred, 0).argmax()
        #            print "pred: ", pred
                    if pred == labels[0][0]:
                        acc +=1

                    train_time = post_train-pre_train
        #            print "numGpus, Train time: ", numGpus, train_time
                    tot_train_time.append(train_time/float(numGpus))
                    print "Train Time: ", np.sum(np.array(train_time))
            #        time_post_train = time.time()
                    count+=1
                    curr_logger.add_scalar_value('train/total_acc', acc/float(count), step=gs)
                    print "gs count  loss  acc: ", gs, count, np.mean(loss_val), acc/float(count)
                    np.save('timing_4gpuloss_'+str(numGpus)+'_'+log_name+'.npy', np.array([tot_load_time, tot_train_time]))
                    print
                    print
            #        tot_train_time.append((time_post_train - time_pre_train))


                    # iterated_list.append(vidNum)
                    # epoch_count+=1
                    #
                    # guess = np.mean(output_predictions, 0).argmax()
                    #
                    # if int(guess) == int(labels[0][0]):
                    #     epoch_acc += 1

                curr_logger.add_scalar_value('train/loss', np.mean(loss_val), step=gs)

            #        curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(epoch_count), step=gs)
            #        curr_logger.add_scalar_value('train_time',time_post_train - time_pre_train, step=gs)
            #        curr_logger.add_scalar_value('load_time',time_post_load - time_pre_load, step=gs)

            #
            #     for clip in range(numClips - excessClips):
            #
            #
            #
            #     for clip in range(excessClips):
            #
            #
            #
            #
            #
            #
            #
            #     for clip in range(x_q.qsize()):
            # #        input_data= np.reshape(loaded_data[clip], (1, -1, size[0], size[1], 3))
            # #        labels = np.reshape(labels, (1, -1))
            #         input_data = x_q.get()
            #         labels = y_q.get()
            #
            #         if not np.all(labels==labels[0][0]):
            #             endOfVid = True
            #
            #         time_pre_train = time.time()
            #
            #         _, loss_val, pred, gs = sess.run([train_op, tower_losses, tower_slogits, global_step], feed_dict={x_placeholder: input_data, y_placeholder: labels})#, learning_rate: lr})
            #
            #         for pred in predictions:
            #             output_predictions[clip] = np.mean(pred, 0)
            #             tot_step+=1
            #
            #             mean_loss.append(np.mean(loss_val))
            #             pred = np.mean(pred, 0).argmax()
            #             print "pred: ", pred
            #             if pred == labels[0][0]:
            #                 acc +=1
            #
            #             time_post_train = time.time()
            #             count+=1
            #             curr_logger.add_scalar_value('train/total_acc', acc/float(count), step=gs)
            #             print "step  loss  acc: ", gs, np.mean(loss_val), acc/float(count)
            #
            #             tot_train_time.append((time_post_train - time_pre_train))
            #
            #     iterated_list.append(vidNum)
            #     epoch_count+=1
            #
            #     guess = np.mean(output_predictions, 0).argmax()
            #
            #     if int(guess) == int(labels[0][0]):
            #         epoch_acc += 1
            #     curr_logger.add_scalar_value('train/loss', float(np.mean(mean_loss)), step=gs)
            #     curr_logger.add_scalar_value('train/epoch_acc', epoch_acc/float(epoch_count), step=gs)
            #     curr_logger.add_scalar_value('train_time',time_post_train - time_pre_train, step=gs)
            #     curr_logger.add_scalar_value('load_time',time_post_load - time_pre_load, step=gs)
                epoch_count+=1
                if epoch_count%50 == 0:

                    saver.save(sess, os.path.join('results',model.name, experiment_name + '_'+dataset,'checkpoints/checkpoint'), global_step.eval(session=sess))
                    print "Saving..."
                #    print "Curr time: ", time.time()-time_init
                #    print "Tot load time: ", np.sum(np.array(tot_load_time))
                #    print "Tot train time: ", np.sum(np.array(tot_train_time))
                #   save_video_list(model.name, vidList, np.array(iterated_list), dataset, split, experiment_name)
    #        if epoch%3==0:
    #            _validate(model, slogits, sess, experiment_name, curr_logger, dataset, inputDims, outputDims, split, gs, size, x_placeholder)

    #        print "Tot load time: ", np.sum(np.array(tot_load_time))

    #        print "Tot train time: ", np.sum(np.array(tot_train_time))
    #        print "Tot time: ", (time.time()-time_init)






def test(model, inputDims, outputDims, seqLength, size, numGpus, dataset, experiment_name, numVids, split, baseDataPath, fName):#, dataSet, params):
    with tf.name_scope("my_scope") as scope:
        isTraining = False
        x_placeholder = tf.placeholder(tf.float32, shape=[inputDims, size[0], size[1] ,3], name='x_placeholder')# shape=[numGpus, inputDims, 224,224,3], name='x_placeholder')
        y_placeholder = tf.placeholder(tf.int64, shape=[inputDims], name='y_placeholder')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Model Inference
        logits = model.inference(x_placeholder, y_placeholder, False, inputDims, outputDims, seqLength, scope)

        # Logits
        softmax = tf.nn.softmax(logits)
        log_name     = ("exp_test_%s_%s_%s" % ( time.strftime("%d_%m_%H_%M_%S"),
                                                           dataset,
                                                           experiment_name) )

        curr_logger = Logger(os.path.join('Logs',model.name,dataset, log_name))
        sess = tf.Session()
        saver = tf.train.Saver()

        # Initialize Variables
        init = tf.global_variables_initializer()
        sess.run(init)


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join('results', model.name,   experiment_name+ '_'+dataset, 'checkpoints/checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'A better checkpoint is found. Its global_step value is: ', global_step.eval(session=sess)

        total_pred = []
        acc = 0
        count = 0
        splits = [3783,3734,3696]
        split = 1

        acc = 0
        count = 0
        total_pred = []
        vidList = gen_video_list(dataset, model.name, experiment_name, fName, split, numVids, False)
        for vidNum in vidList:

            count +=1
        #    fName = 'testlist'
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

    parser.add_argument('--numGpus', action= 'store', required=True, type=int,
            help = 'Number of Gpus used for calculation')

    parser.add_argument('--train', action= 'store', required=True, type=int,
            help = 'Binary value to indicate training or evaluation instance')

    parser.add_argument('--load', action='store', type=int,
            help = 'Whether you want to load a saved model to train from scratch.')

    parser.add_argument('--size', action='store', required=True, type=int,
            help = 'Input frame size')

    parser.add_argument('--inputDims', action='store', required=True, type=int,
            help = 'Input Dimensions')

    parser.add_argument('--outputDims', action='store', required=True, type=int,
            help = 'Output Dimensions')

    parser.add_argument('--seqLength', action='store', required=True, type=int,
            help = 'Length of sequences for LSTM')

    parser.add_argument('--expName', action='store', required=True,
            help = 'Unique name of experiment being run')

    parser.add_argument('--numVids', action='store', required=True, type=int,
            help = 'Unique name of experiment being run')

    parser.add_argument('--lr', action='store', required=True, type=float,
            help = 'Learning Rate')

    parser.add_argument('--wd', action='store', required=True, type=float,
            help = 'Weight Decay')

    parser.add_argument('--nEpochs', action='store', required=True, type=int,
            help = 'Number of Epochs')

    parser.add_argument('--split', action='store', type=int,
            help = 'Dataset split to use')

    parser.add_argument('--baseDataPath', action='store',
            help = 'Path to datasets')

    args = parser.parse_args()
#    import pdb;pdb.set_trace()
    loadModel = 1
    modelName = args.model
    numGpus = args.numGpus
    dataset = args.dataset
    loadModel = args.load
    inputDims = args.inputDims
    outputDims = args.outputDims
    seqLength = args.seqLength
    size = args.size
    experiment_name = args.expName
    numVids = args.numVids
    wd = args.wd
    lr = args.lr
    split = args.split
    nEpochs = args.nEpochs

    print args
    baseDataPath = '/z/home/madantrg/Datasets'

    if modelName=='lrcn':
        model = LRCN()
        # if args.train:
        #     inputDims = 16
        #     outputDims = 101
        #     seqLength = 16
        #     size = [227,227]
        # else:
        #     inputDims = 160
        #     outputDims = 101
        #     seqLength = 16
        #     size = [227,227]


    elif modelName == 'vgg16':
        model = VGG16()
        # inputDims = 25
        # outputDims = 101
        # seqLength = 25
        # size = [224,224]

    elif modelName == 'resnet':
        model = ResNet()
        # inputDims = 25
        # outputDims = 101
        # seqLength = 25
        # size = [224,224]

    else:
        print("Model not found")
    #numGpus = 2
    # if dataset == 'HMDB51':
    #     outputDims = 51
    # else:
    #     outputDims = 101
    print args.train

    print 'outputDims: ', outputDims
    if args.train:
        train(model, inputDims, outputDims, seqLength, [size, size], numGpus, dataset, experiment_name, loadModel, numVids, nEpochs, split, baseDataPath, 'trainlist',learning_rate_init=lr, wd=wd)

    else:

        test(model, inputDims, outputDims, seqLength, [size, size], numGpus, dataset, experiment_name, numVids, split, baseDataPath, 'testlist')
