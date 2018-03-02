import tensorflow as tf
import numpy as np
import math

_R_MEAN = 123
_G_MEAN = 117
_B_MEAN = 104

_RESIZE_SIDE_MIN = tf.constant(256, dtype=tf.int32)

def resample_model(video, sample_dims, frame_count, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)
    r_alpha = alpha * tf.cast(frame_count, tf.float32) / float(sample_dims)
    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output


def resample_input(video, sample_dims, frame_count, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """

    sample_dims = tf.cast(sample_dims, tf.float32)
    indices = tf.range(start=0., limit=sample_dims, delta=1., dtype=tf.float32)
    r_alpha = alpha * tf.cast(frame_count, tf.float32) / sample_dims
    indices = tf.multiply(tf.tile([r_alpha], [tf.cast(sample_dims, tf.int32)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output

def resample_model_sinusoidal_quant(video, sample_dims, frame_count, num_vids, tracker):
    """Return video sampled at random rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """
    alpha_list = tf.convert_to_tensor([0.4, 0.8, 1.5, 2.5])

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)

    curr_epoch = tf.cast(tracker / num_vids, tf.int32)

    alpha_ind = tf.mod(curr_epoch, 4)
    r_alpha = alpha_list[alpha_ind] * tf.cast(frame_count, tf.float32) / float(sample_dims)

    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))

    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output, alpha_list[alpha_ind]

def resample_model_sinusoidal(video, sample_dims, frame_count, tracker):
    """Return video sampled at random rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """
    alpha       = 1.6
    upper_limit = 3.0
    lower_limit = 0.2

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)

    # Sinusoidal variation with alpha being the DC offset
    r_alpha = (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32))) * tf.cast(frame_count, tf.float32) / float(sample_dims)

    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))

    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output, (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32)))

def crop_and_resize_(input_data_tensor, offsets, crop_sizes, size):
    input_seg_list = []
    num_element = len(offsets)
    for i in range(num_element):
        input_seg = tf.gather(input_data_tensor, [i])
        input_seg = tf.map_fn(lambda img: tf.slice(img, offsets[i], crop_sizes[i]), input_seg)
        input_seg = tf.map_fn(lambda img: tf.reshape(img, crop_sizes[i]), input_seg)
        input_seg = tf.map_fn(lambda img: resize_bilinear(img, size, align_corners=True), input_seg)
        input_seg_list.append(input_seg)

    shape = input_data_tensor.get_shape().as_list()
    input_data_tensor = tf.stack(input_seg_list, 1)
    input_data_tensor = tf.reshape(input_data_tensor, [-1, tf.to_int32(size[0]), tf.to_int32(size[1]), shape[-1]])

    return input_data_tensor

def resize_bilinear(img, size, align_corners=True):
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_bilinear(img, size, align_corners=True)
    img = tf.squeeze(img)
    img.set_shape([None, None, 3])
    return img

def _loop_video(offset_tensor, input_data_tensor, frames, height, width, channel, footprint):
    loop_factor       = tf.cast(tf.add(tf.divide(tf.subtract(footprint, frames), frames), 1), tf.int32)
    loop_stack        = tf.stack([loop_factor,1,1,1])
    input_data_tensor = tf.tile(input_data_tensor, loop_stack)
    reshape_stack     = tf.stack([tf.multiply(frames, loop_factor),height,width,channel])
    input_data_looped = tf.reshape(input_data_tensor, reshape_stack)
    output_data       = tf.concat([offset_tensor, input_data_looped], axis = 0)

    return output_data


def _mean_image_subtraction(images, means):
    num_channels = 3
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    results = [0]*num_channels
    for i in range(num_channels):
        results[num_channels-1-i] = channels[i]-means[i]

    return tf.concat(axis=3, values=results)

def _get_new_length(height, width):
    height = tf.to_float(height)
    width = tf.to_float(width)
    side_min = tf.to_float(_RESIZE_SIDE_MIN)

    scale = tf.cond(tf.greater(height, width), lambda: tf.div(side_min, width), lambda: tf.div(side_min, height))

    new_height = tf.round(tf.multiply(height, scale))
    new_width = tf.round(tf.multiply(width, scale))

    return new_height, new_width

def get_cropping_offsets_(new_height, new_width, istraining, num_sample):
    new_height = tf.to_float(new_height)
    new_width = tf.to_float(new_width)
    side_min = tf.to_float(_RESIZE_SIDE_MIN)

    offsets = []
    crop_sizes = []

    scale_h = .875
    scale_w = .875
    loc = 4
    if istraining:
        scale_h = [1., .875, .75, .66][np.random.randint(4)]
        scale_w = [1., .875, .75, .66][np.random.randint(4)]
        loc = np.random.randint(5)

    nh = side_min*scale_h
    nw = side_min*scale_w
    crop_size = tf.to_int32(tf.stack([nh, nw, 3]))

    offset = tf.to_int32(tf.stack([0, 0, 0]))
    if loc == 1:
        offset = tf.to_int32(tf.stack([tf.maximum(new_height-nh-1, 0.), 0, 0]))
    elif loc == 2:
        offset = tf.to_int32(tf.stack([tf.maximum(new_height-nh-1, 0.), tf.maximum(new_width-nw-1, 0.), 0]))
    elif loc == 3:
        offset = tf.to_int32(tf.stack([0, tf.maximum(new_width-nw-1, 0.), 0]))
    elif loc == 4:
        offset = tf.to_int32(tf.stack([tf.div(new_height-nh-1, 2.), tf.div(new_width-nw-1, 2.), 0]))

    for i in range(num_sample):
        offsets.append(offset)
        crop_sizes.append(crop_size)

    return offsets, crop_sizes

def preprocess(input_data_tensor, frames, height, width, channel, size, label, istraining, num_seg, input_dims, cvr, input_alpha, num_vids, tracker):
    new_height, new_width = _get_new_length(height, width)
    new_height = tf.to_int32(new_height)
    new_width = tf.to_int32(new_width)
    h_size = tf.convert_to_tensor(size[0], dtype=tf.float32)
    w_size = tf.convert_to_tensor(size[1], dtype=tf.float32)

    num_seg = 3

    # Reduce the total video to have exactly footprint frames
    footprint = input_dims*num_seg
    input_data_tensor = tf.cond(tf.less(frames, footprint),
                        lambda: _loop_video(input_data_tensor, input_data_tensor, frames, height, width, channel, footprint),
                        lambda: input_data_tensor)
    frames = tf.shape(input_data_tensor)[0]
    indices = tf.range(0, frames, tf.div(frames,footprint))
    input_data_tensor = tf.gather(input_data_tensor, indices[:footprint])
    frames = tf.to_int32(footprint)

    input_data_tensor = tf.cast(input_data_tensor, tf.float32)
    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)
    SAMPLE_NUM = input_dims

    # Preprocessing for each segment
    input_seg_list = []
    for i in range(num_seg):
        input_seg_tensor = tf.gather(input_data_tensor, tf.range(i*SAMPLE_NUM, (i+1)*SAMPLE_NUM))
        input_seg_tensor = tf.map_fn(lambda img: resize_bilinear(img, [new_height, new_width], align_corners=True), input_seg_tensor)

        offsets, crop_sizes = get_cropping_offsets_(new_height, new_width, istraining, SAMPLE_NUM)
        input_seg_tensor = crop_and_resize_(input_seg_tensor, offsets, crop_sizes, size)

        # Flipping block-wise in the training
        if istraining and np.random.randint(2):
            input_seg_tensor = tf.map_fn(lambda img: tf.image.flip_left_right(img), input_seg_tensor)

        # Apply the SR resampling
        if istraining:
            input_seg_tensor, alpha_tensor = resample_model_sinusoidal_quant(input_seg_tensor, SAMPLE_NUM/num_seg, SAMPLE_NUM, num_vids, tracker)

        else:
            input_seg_tensor = resample_model(input_seg_tensor, SAMPLE_NUM/num_seg, SAMPLE_NUM, cvr)
            alpha_tensor      = tf.convert_to_tensor(cvr)

        # END IF

        input_seg_list.append(input_seg_tensor)

    input_data_tensor = tf.stack(input_seg_list, 1)

    input_data_tensor = tf.reshape(input_data_tensor, [SAMPLE_NUM, tf.to_int32(h_size), tf.to_int32(w_size), channel])
    input_data_tensor = _mean_image_subtraction(input_data_tensor, [_R_MEAN, _G_MEAN, _B_MEAN])
    input_data_tensor = tf.map_fn(lambda img: tf.image.rot90(img, 1), input_data_tensor)

    return input_data_tensor, alpha_tensor
