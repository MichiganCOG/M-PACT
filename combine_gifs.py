import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from utils import make_dir
import os
from models.resnet_RIL.resnet_RIL_interp_mean_model_v1  import ResNet_RIL_Interp_Mean_v1
from models.resnet_RIL.resnet_RIL_interp_median_model_v1  import ResNet_RIL_Interp_Median_v1
from models.resnet_RIL.resnet_RIL_interp_max_model_v1  import ResNet_RIL_Interp_Max_v1

from models.resnet_RIL.resnet_RIL_interp_mean_model_v2  import ResNet_RIL_Interp_Mean_v2
from models.resnet_RIL.resnet_RIL_interp_median_model_v2  import ResNet_RIL_Interp_Median_v2
from models.resnet_RIL.resnet_RIL_interp_max_model_v2  import ResNet_RIL_Interp_Max_v2

from models.resnet_RIL.resnet_RIL_interp_mean_model_v3  import ResNet_RIL_Interp_Mean_v3
from models.resnet_RIL.resnet_RIL_interp_median_model_v3  import ResNet_RIL_Interp_Median_v3

from models.resnet_RIL.resnet_RIL_interp_median_model_v14  import ResNet_RIL_Interp_Median_v14
from models.resnet_RIL.resnet_RIL_interp_median_model_v21  import ResNet_RIL_Interp_Median_v21
from models.resnet_RIL.resnet_RIL_interp_median_model_v22  import ResNet_RIL_Interp_Median_v22
from models.resnet_RIL.resnet_RIL_interp_median_model_v23  import ResNet_RIL_Interp_Median_v23
from models.resnet_RIL.resnet_RIL_interp_median_model_v26  import ResNet_RIL_Interp_Median_v26
from models.resnet_RIL.resnet_RIL_interp_median_model_v27  import ResNet_RIL_Interp_Median_v27

from extract_RAIN_mp4_TFRecords import test

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# framesMean, input_data = test( model             = ResNet_RIL_Interp_Mean_v3(),
#                 input_dims        = 250,
#                 output_dims       = 51,
#                 seq_length        = 50,
#                 size              = [224,224],
#                 dataset           = 'HMDB51',
#                 loaded_dataset    = 'HMDB51',
#                 experiment_name   = 'tfrecords_resnet_rain_interp_mean_v3_HMDB51',
#                 num_vids          = 1530,
#                 split             = 1,
#                 base_data_path    = '/z/dat',
#                 f_name            = 'testlist',
#                 load_model        = 1)
#
# tf.reset_default_graph()

extractEnd = 1

if extractEnd == 0:
    inputDims = 250
else:
    inputDims = 100

framesMedian, input_data = test( model             = ResNet_RIL_Interp_Median_v22(),
                input_dims        = inputDims,
                output_dims       = 51,
                seq_length        = 50,
                size              = [224,224],
                dataset           = 'HMDB51',
                loaded_dataset    = 'HMDB51',
                experiment_name   = 'tfrecords_resnet_rain_interp_median_v22_HMDB51',
                num_vids          = 1530,
                split             = 1,
                base_data_path    = '/z/dat',
                f_name            = 'testlist',
                load_model        = 1,
                extract_end       = extractEnd)

# tf.reset_default_graph()
#
# framesMax, input_data = test( model             = ResNet_RIL_Interp_Max_v1(),
#                 input_dims        = 250,
#                 output_dims       = 51,
#                 seq_length        = 50,
#                 size              = [224,224],
#                 dataset           = 'HMDB51',
#                 loaded_dataset    = 'HMDB51',
#                 experiment_name   = 'tfrecords_resnet_rain_interp_max_v1_HMDB51',
#                 num_vids          = 1530,
#                 split             = 1,
#                 base_data_path    = '/z/dat',
#                 f_name            = 'testlist',
#                 load_model        = 1)





def save_gif(frames, name, model, dataset, vid_num):
    my_dpi = 16

    #fig = plt.figure(figsize=(4*224/my_dpi*2,224/my_dpi*2), dpi=my_dpi)
    #fig = plt.figure(figsize=(3*224/my_dpi*2,224/my_dpi*2), dpi=my_dpi)
    fig = plt.figure(figsize=(2*224/my_dpi*2,224/my_dpi*2), dpi=my_dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    def animate(i):
        if i < frames.shape[0]:
        #    frame = np.zeros((224,4*224,3))
        #    frame = np.zeros((224,3*224,3))
            frame = np.zeros((224,2*224,3))
            frame[:,:,0] = frames[i][:,:,0] + _R_MEAN
            frame[:,:,1] = frames[i][:,:,1] + _G_MEAN
            frame[:,:,2] = frames[i][:,:,2] + _B_MEAN
            return ax.imshow(frame/255.0, aspect='auto')
        else:
        #    frame = np.zeros((224,4*224,3))
        #    frame = np.zeros((224,3*224,3))
            frame = np.zeros((224,2*224,3))
            return ax.imshow(frame/255.0, aspect='auto')
    ims = map(lambda x: (animate(x), ax.set_title('')), range(frames.shape[0]+25))
    anim = animation.ArtistAnimation(fig, ims, interval=frames.shape[0]+25,)
    make_dir(os.path.join('results/gifs', dataset.split('Rate')[0]+'_'+model))
    make_dir(os.path.join('results/gifs', dataset.split('Rate')[0]+'_'+model, vid_num))
    anim.save(os.path.join('results/gifs', dataset.split('Rate')[0]+'_'+model,vid_num,name+'_'+model+'.mp4'), writer='imagemagick', fps=25, dpi=my_dpi)
    print "Saved ", name
    plt.cla()
    plt.clf()
    plt.close(fig)

if extractEnd == 1:
    zer = np.zeros((50,224,224,3))
else:
    zer = np.zeros((200,224,224,3))
zer[:,:,0] -= _R_MEAN
zer[:,:,1] -= _G_MEAN
zer[:,:,2] -= _B_MEAN
#framesMean = np.concatenate([framesMean, zer], axis=0)
# framesMax = np.concatenate([framesMax, zer], axis=0)
framesMedian = np.concatenate([framesMedian, zer], axis=0)

#frames = np.concatenate([input_data, framesMedian, framesMean, framesMax], axis = 2)
#frames = np.concatenate([input_data, framesMedian, framesMean], axis = 2)
#import pdb; pdb.set_trace()
frames = np.concatenate([input_data, framesMedian], axis = 2)
save_gif(frames, 'Combined', "RAINv22", 'HMDB51', str(1))












#
