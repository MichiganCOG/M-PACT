import tensorflow as tf
import numpy as np
from utils.preprocessing_utils import *

'''
Sinusoidal Resampling
'''


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
  image = aspect_preserving_resize(image, resize_side_min)
  image = central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  return image




def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, tracker, input_alpha=1.0):
    """
    Preprocessing function corresponding to the chosen model
    Args:
        :input_data_tensor: Raw input data
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
        :tracker:           Variable counting the total number of videos that have been loaded during training
        :input_alpha:       Value to resample video to during testing analysis of speed robustness
        
    Return:
        Preprocessing input data and labels tensor
    """

    _mean_image = np.load('models/c3d/crop_mean.npy')

    input_data_tensor = input_data_tensor[...,::-1]

    num_frames_per_clip = input_dims

#    tracker = [v for v in tf.global_variables() if v.name == 'my_scope/global_step:0'][0]

    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)

    if istraining:
        input_data_tensor, alpha_tensor = resample_model_sinusoidal(input_data_tensor, input_dims, frames, tracker)
    else:
        input_data_tensor = resample_model(input_data_tensor, input_dims, frames, 1.0)
        alpha_tensor      = tf.convert_to_tensor(1.0)

    # END IF

    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=size[0]), input_data_tensor)

    input_data_tensor = input_data_tensor - _mean_image[...,::-1].tolist()

    return input_data_tensor, alpha_tensor
