import tensorflow as tf
import numpy as np
import os
import multiprocessing as mp
import threading

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
from utils import *
from Queue import Queue

sess = tf.Session()

numGpus = 2 
inputDims = 50
seqLength = 50

end_msg = False

def load_and_enqueue(model, vidList, fName, baseDataPath, dataset, split, size, isTraining):
    while True:
        if not end_msg:
            print "Loading video: ",vidList[0]
            loaded_data, labels= load_dataset(model, vidList[0], fName, os.path.join(baseDataPath, dataset+'HDF5RGB','Split'+str(split)), os.path.join('datasets',dataset,fName+'0'+str(split)+'.txt'), os.path.join("datasets",dataset,"classInd.txt"), size, isTraining, dataset)
            vidList = vidList[1:]
        else:
            break

        sess.run(en_Q, feed_dict={x_placeholder: loaded_data, y_placeholder: labels})

if __name__=="__main__":
    model = ResNet()
    vidNum = [1,2]
    fName = 'trainlist'
    baseDataPath = '/z/home/madantrg/Datasets'
    dataset = 'HMDB51'
    split = 1
    size=  [224,224]
    isTraining =  False

    x_placeholder          = tf.placeholder(tf.float32, shape=[inputDims,  size[0], size[1] ,3], name='x_placeholder')
    y_placeholder          = tf.placeholder(tf.int64,   shape=[], name='y_placeholder')
    istraining_placeholder = tf.placeholder(tf.bool,    name='istraining_placeholder')

    Q                      = tf.FIFOQueue(numGpus, [tf.float32, tf.int64], shapes=[[inputDims, size[0], size[1], 3], []])
    
    en_Q = Q.enqueue([x_placeholder, y_placeholder])

    Data, Labels  = Q.dequeue_many(numGpus)

    ss = Data

    t = threading.Thread(target=load_and_enqueue, args=(model, vidNum, fName, baseDataPath, dataset, split, size, isTraining))
    t.start()


    start_ = time.time()
    for runs  in range(1):
        D = sess.run([ss])
    end_msg = True
    print "Time to load 5*2 videos is: ", time.time()-start_
    import pdb;pdb.set_trace()
