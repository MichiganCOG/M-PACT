""" THIS FILE IS USED TO PREPROCESS DATA FOR THE LRCN MODEL """

import os
import h5py
import numpy as np

from skimage.transform import resize
from scipy.ndimage     import zoom

def oversample(images, crop_dims):
    """
    This code is taken from:
    https://github.com/LisaAnne/lisa-caffe-public/blob/lstm_video_deploy/python/caffe/io.py
    Crop images into the four corners, center, and their mirrored versions.
    Args:
        :images:    Iterable [H x W x C] ndarray
        :crop_dims: List detailing final height and width of cropped frames.
    
    Return:    
        :crops:     (10*N x H x W x C) ndarray of crops for number of inputs N.
    """

    im_shape  = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix  = np.empty((5, 4), dtype=int)
    curr      = 0

    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
        
        # END FOR

    # END FOR

    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([-crop_dims / 2.0,
                                                               crop_dims / 2.0])
    crops_ix    = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix    = 0

    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1

        # END FOR

        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirror
    
    # END FOR

    return crops


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Args:
        :im:           (H x W x C) ndarray
        :new_dims:     (height, width) tuple of new dimensions.
        :interp_order: interpolation order, default is linear.
    
    Return:
        :im:           resized ndarray with shape (new_dims[0], new_dims[1], C)
    """

    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()

        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='nearest')
            resized_im = resized_std * (im_max - im_min) + im_min

        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]), dtype=np.float32)
            ret.fill(im_min)
            return ret

        # END IF

    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)

    # END IF

    return resized_im.astype(np.float32)


def preprocess(index, data, label, size, is_training):
    """
    Args:
        :index:       Integer used to index a video from a given text file listing all available videos
        :data:        Original data
        :label:       True labels   
        :size:        List detailing final height and width from preprocessing  
        :is_training: Boolean value indicating phase (TRAIN or TEST)

    Return:
        :vid_clips:  Preprocessed video clips
    """

    clip_length = 16
    offset      = 8
    size        = size[0]

    data_mean = np.array((103.939, 116.779, 128.68)).reshape(1, 1, 1, 3)
    data      = data - data_mean

    if data.shape[1] < 240:
        tempData = []

        for i in range(data.shape[0]):
            tempData.append(resize_image(data[i],(240,320)))

        # END FOR

        data = np.array(tempData)

    # END IF

    input_data = []
    vid_length = data.shape[0]

    for j in range(0, vid_length, offset):
        if (j + clip_length) < vid_length:
            input_data.extend(data[j:j + clip_length])

        else:  # video may not be divisible by clip_length
            input_data.extend(data[-clip_length:])
    
        # END IF

    # END FOR

    input_data = np.array(input_data)
    vid_clips  = []

    if is_training:
        for i in range(0,len(input_data),clip_length):
            clip_input = input_data[i:i+clip_length]
            new_clip   = []

            for cl in clip_input:
                new_clip.append(resize_image(cl, (size,size)))
        
            # END FOR

            vid_clips.append(np.array(new_clip))

        # END FOR

    else:
        for i in range(0,len(input_data),clip_length):
            clip_input = input_data[i:i+clip_length]
            clip_input = oversample(clip_input,[size, size])
            vid_clips.append(clip_input)

        # END FOR

    # END IF

    return vid_clips
