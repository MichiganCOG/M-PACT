import tensorflow as tf
import numpy as np


_R_MEAN = 123.68 
_G_MEAN = 116.78 
_B_MEAN = 103.94 

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = _random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=256,
                         resize_side_max=512):
  """Preprocesses the given image for training.
  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side_min)
  image = _central_crop([image], output_height, output_width)[0]
  #image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  #image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  #image = tf.image.random_flip_left_right(image)
  image = image * 2./255. - 1.
  return image

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
  image = _aspect_preserving_resize(image, resize_side)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  #image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  image = image * 2./255. - 1.
  return image


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=256,
                     resize_side_max=512):
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
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].
  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                               resize_side_min, resize_side_max)

  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)

  # END IF

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

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)
    r_alpha = alpha * tf.cast(frame_count, tf.float32) / float(sample_dims)
    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))

    return output

def resample_model(video, sample_dims, frame_count, alpha):
    """Return video sampled at desired rate (model based)
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


def _loop_video_with_offset(offset_tensor, input_data_tensor, offset_frames, frames, height, width, channel, footprint):
    """
    Loop the video the number of times necessary for the number of frames to be > footprint
    Args:
        :offset_tensor:     Raw input data from offset frame number
        :input_data_tensor: Raw input data
        :frames:            Total number of frames
        :height:            Height of frame
        :width:             Width of frame
        :channel:           Total number of color channels
        :footprint:         Total length of video to be extracted before sampling down

    Return:
        Looped video
    """

    loop_factor       = tf.cast(tf.add(tf.divide(tf.subtract(footprint, offset_frames), frames), 1), tf.int32)
    loop_stack        = tf.stack([loop_factor,1,1,1])
    input_data_tensor = tf.tile(input_data_tensor, loop_stack)
    reshape_stack     = tf.stack([tf.multiply(frames, loop_factor),height,width,channel])
    input_data_looped = tf.reshape(input_data_tensor, reshape_stack)

    output_data       = tf.concat([offset_tensor, input_data_looped], axis = 0)

    return output_data

def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, input_alpha=1.0):
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
        :input_alpha:       Desired resampling rate (input)

    Return:
        Preprocessing input data and labels tensor
    """

    # Setup different temporal footprints for training and testing phase
    if istraining:
        footprint = 64

    else:
        footprint = 250

    # END IF

    temporal_offset   = tf.cond(tf.greater(frames, 250), lambda: tf.random_uniform(dtype=tf.int32, minval=0, maxval=frames - 250 + 1, shape=np.asarray([1]))[0], lambda: tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=np.asarray([1]))[0])

    input_data_tensor = tf.cond(tf.less(frames - temporal_offset, 250), 
                                lambda: _loop_video_with_offset(input_data_tensor[temporal_offset:,:,:,:], input_data_tensor, frames-temporal_offset, frames, height, width, channel, 250),
                                lambda: input_data_tensor[temporal_offset:temporal_offset + 250, :, :, :])

    # Remove excess frames after looping to reduce to footprint size
    input_data_tensor = tf.slice(input_data_tensor, [0,0,0,0], tf.stack([250, height, width, channel]))
    input_data_tensor = tf.reshape(input_data_tensor, tf.stack([250, height, width, channel]))

    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    # Resample input to desired rate (input fluctuation only, not related to model)
    input_data_tensor = resample_input(input_data_tensor, 250, 250, input_alpha)

    if istraining:
        rr = tf.random_uniform(dtype=tf.float32, minval=0.2, maxval=3.0, shape=np.asarray([1]))[0]
    
    else:
        rr= 1.0

    # END IF

    # Resample input to desired rate (resampling as a model requirement)
    input_data_tensor = resample_model(input_data_tensor, footprint, 250, rr)

    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=_RESIZE_SIDE_MIN), input_data_tensor)

    return input_data_tensor, rr
