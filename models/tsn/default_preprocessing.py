import tensorflow as tf
import numpy      as np

from utils.preprocessing_utils import *


def preprocess_for_train(image, output_height, output_width, resize_side):
    """Preprocesses the given image for training.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """

    ############################################################################
    #             TODO: Add preprocessing done during training phase           #
    #         Preprocessing option found in utils/preprocessing_utils.py       #
    #                                                                          #
    #  EX:    image = aspect_preserving_resize(image, resize_side)             #
    #         image = central_crop([image], output_height, output_width)[0]    #
    #         image.set_shape([output_height, output_width, 3])                #
    #         image = tf.to_float(image)                                       #
    #         return image                                                     #
    ############################################################################
    return 1


def preprocess_for_eval(image, output_height, output_width, resize_side):
    """Preprocesses the given image for evaluation.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """

    ############################################################################
    #             TODO: Add preprocessing done during training phase           #
    #         Preprocessing option found in utils/preprocessing_utils.py       #
    #                                                                          #
    #  EX:    image = aspect_preserving_resize(image, resize_side)             #
    #         image = central_crop([image], output_height, output_width)[0]    #
    #         image.set_shape([output_height, output_width, 3])                #
    #         image = tf.to_float(image)                                       #
    #         return image                                                     #
    ############################################################################

    image = tf.gather(image, 0)
    image = tf.reshape(resize(image, 256, 340), [256,340,3])
    image = mean_image_subtraction(image, [123, 117, 104])
    images = oversample(tf.convert_to_tensor([image]), [output_height, output_width])
    return images

def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=RESIZE_SIDE_MIN):
    """Preprocesses the given image.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    Returns:
    A preprocessed image.
    """
    if is_training:
      return preprocess_for_train(image, output_height, output_width,
                             resize_side_min)

    else:
      return preprocess_for_eval(image, output_height, output_width,
                             resize_side_min)

    # END IF



def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step, num_segs = 3, input_alpha=1.0):
    """
    Preprocessing function corresponding to the chosen model
    Args:
        :input_data_tensor: Raw input data [clip_length/frames x height x width x channels]
        :frames:            Total number of frames
        :height:            Height of frame
        :width:             Width of frame
        :channel:           Total number of color channels
        :input_dims:        Number of frames to be provided as input to model
        :output_dims:       Total number of labels
        :seq_length:        Number of frames expected as output of model
        :size:              Output size of preprocessed frames
        :label:             Label of current sample
        :istraining:        Boolean indicating training or testing phase
        :num_segs:          Number of segments to evenly divice the video into

    Return:
        Preprocessing input data and labels tensor
    """

    # CV2 uses BGR so convert from RGB
    #input_data_tensor = input_data_tensor[...,::-1]

    # Allow for resampling of input during testing for evaluation of the model's stability over video speeds
    input_data_tensor = tf.cast(input_data_tensor, tf.float32)
    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)

    # During training, segment video into input_dims/seq_length segments, then randomly extract a seq_length snippet from each segment
    if istraining:
        combined_snippet_len = input_dims
        snippet_length = seq_length # = 1 For original authors

        num_segs       = combined_snippet_len/snippet_length

        # Ensure enough frames to extract snippet_length number of frames from each of num_segs segments that the video is split into
        input_data_tensor = tf.cond(tf.less(frames, snippet_length * num_segs),
                                    lambda: loop_video_with_offset(input_data_tensor, input_data_tensor, 0, frames, height, width, channel, snippet_length * num_segs),
                                    lambda: input_data_tensor)

        frames = input_data_tensor.get_shape().as_list()[0]
        segment_length = frames/num_segs

        # For each segment the video is split into, randomly extract 'snippet_length' number of sequential frames within that segment
        for seg in range(num_segs):
            random_extract_index = tf.random_uniform(dype=tf.int32, minval=seg * segment_length, maxval= (seg+1)*segment_length - snippet_length, shape=np.asarray([1]))[0]
            output_data_tensor.append(tf.gather(input_data_tensor, tf.range(random_extract_index, random_extract_index+snippet_length)))

        # END FOR

        output_data_tensor = tf.concat(output_data_tensor, axis=0)

    # During testing, resample video down to seq_length/10 number of frames, then oversample (each frame x10 crops and mirrors) to seq_length frames
    else:
        snippet_length = input_dims/10 # Equivalent to seq_length/10

        # Ensure enough frames to extract snippet_length number of frames from each video
        input_data_tensor = tf.cond(tf.less(frames, snippet_length),
                                    lambda: loop_video_with_offset(input_data_tensor, input_data_tensor, 0, frames, height, width, channel, snippet_length),
                                    lambda: input_data_tensor)

        frames_after_loop = tf.shape(input_data_tensor)[0]

        # Uniformly resample video down to snippet_length number of frames
        output_data_tensor = resample_input(input_data_tensor, snippet_length, frames_after_loop, 1.0)

        # Prepare output_data_tensor for oversampling which will result in 10x the number of output frames per frame
        # Pad the current output tensor since tf.map_fn requires identical dimension for input and output
        output_data_tensor = tf.pad(tf.expand_dims(output_data_tensor, axis=1), [[0,0],[0,9],[0,0],[0,0],[0,0]])

    # END IF

    # Apply preprocessing related to individual frames (cropping, flipping, resize, etc.... )
    output_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=size[0]), output_data_tensor)

    # Ensure that the final output is the correct dimensionality, for testing this will result in [combined_snippet_len*10, out_H, out_W, chan]
    output_data_tensor = tf.reshape(output_data_tensor, [input_dims, size[0], size[1], 3])

    output_data_tensor = tf.map_fn(lambda img: tf.image.rot90(img, 1), output_data_tensor)

    # CV2 uses BGR so convert from RGB
    output_data_tensor = output_data_tensor[...,::-1]


    return output_data_tensor
