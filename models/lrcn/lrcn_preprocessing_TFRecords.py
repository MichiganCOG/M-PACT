# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.
The preprocessing steps for VGG were introduced in the following technical
report:
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0
More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy      as np


#slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512




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

    image = tf.gather(images, 0)
    crop_h = crop_dims[0]
    crop_w = crop_dims[1]
    offset_h = tf.subtract(image.shape[0].value, crop_h)
    offset_w = tf.subtract(image.shape[1].value, crop_w)

    crops = []

    h_offsets = [0, offset_h]
    w_offsets = [0, offset_w]

    for h in h_offsets:
        for w in w_offsets:
            crops.append(_crop(image, h, w, crop_h, crop_w))

    crops.append(_central_crop([image], crop_h, crop_w)[0])

    # Mirror the crops
    for i in range(len(crops)):
        crops.append(tf.image.flip_left_right(crops[i]))

    return tf.convert_to_tensor(crops)
    #
    # offsets = (0,0), (0, crop_w), (crop_h, 0), (crop_h, crop_w), central_crop
    # flips = tf.image.flip_left_right(offsets)
    #
    # im_shape  = np.array(images[0].shape)
    # crop_dims = np.array(crop_dims)
    # im_center = im_shape[:2] / 2.0
    #
    # # Make crop coordinates
    # h_indices = (0, im_shape[0] - crop_dims[0])
    # w_indices = (0, im_shape[1] - crop_dims[1])
    # crops_ix  = np.empty((5, 4), dtype=int)
    # curr      = 0
    #
    # for i in h_indices:
    #     for j in w_indices:
    #         crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
    #         curr += 1
    #
    #     # END FOR
    #
    # # END FOR
    #
    # crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([-crop_dims / 2.0,
    #                                                            crop_dims / 2.0])
    # crops_ix    = np.tile(crops_ix, (2, 1))
    #
    # # Extract crops
    # crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
    #                   im_shape[-1]), dtype=np.float32)
    # ix    = 0
    #
    # for im in images:
    #     for crop in crops_ix:
    #         crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
    #         ix += 1
    #
    #     # END FOR
    #
    #     crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirror
    #
    # # END FOR
    #
    # return crops


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


def _resize_image(image, height, width):
  """Resize images
  Args:
    image: A 3-D image `Tensor`.
    new_height: An integer, image height after resize
    new_width: An integer, image width after resize
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  new_height = tf.convert_to_tensor(height, dtype=tf.int32)
  new_width = tf.convert_to_tensor(width, dtype=tf.int32)

  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=True)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([height, width, 3])
  return resized_image

#
# def preprocess_for_train(image,
#                          output_height,
#                          output_width,
#                          resize_side_min=_RESIZE_SIDE_MIN,
#                          resize_side_max=_RESIZE_SIDE_MAX):
#   """Preprocesses the given image for training.
#   Note that the actual resizing scale is sampled from
#     [`resize_size_min`, `resize_size_max`].
#   Args:
#     image: A `Tensor` representing an image of arbitrary size.
#     output_height: The height of the image after preprocessing.
#     output_width: The width of the image after preprocessing.
#     resize_side_min: The lower bound for the smallest side of the image for
#       aspect-preserving resizing.
#     resize_side_max: The upper bound for the smallest side of the image for
#       aspect-preserving resizing.
#   Returns:
#     A preprocessed image.
#   """
#   resize_side = tf.random_uniform(
#       [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
#
#   image = _aspect_preserving_resize(image, resize_side_min)
#   image = _random_crop([image], output_height, output_width)[0]
#   image.set_shape([output_height, output_width, 3])
#   image = tf.to_float(image)
#   image = tf.image.random_flip_left_right(image)
#   return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


# def preprocess_for_eval(image, output_height, output_width, resize_side):
#   """Preprocesses the given image for evaluation.
#   Args:
#     image: A `Tensor` representing an image of arbitrary size.
#     output_height: The height of the image after preprocessing.
#     output_width: The width of the image after preprocessing.
#     resize_side: The smallest side of the image for aspect-preserving resizing.
#   Returns:
#     A preprocessed image.
#   """
#   image = _aspect_preserving_resize(image, resize_side)
#   image = _central_crop([image], output_height, output_width)[0]
#   image.set_shape([output_height, output_width, 3])
#   image = tf.to_float(image)
#   return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])



def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
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





def _loop_video(input_data_tensor, frames, height, width, channel, input_dims):
    # Loop the video the number of times necessary for the number of frames to be > input_dims
    loop_factor = tf.cast(tf.add(tf.divide(input_dims, frames), 1), tf.int32)
    loop_stack = tf.stack([loop_factor,1,1,1])
    input_data_tensor = tf.tile(input_data_tensor, loop_stack)
    reshape_stack = tf.stack([tf.multiply(frames, loop_factor),height,width,channel])
    return tf.reshape(input_data_tensor, reshape_stack)


def _sample_video(video, frame_count, offset):
    # Return frame_count number of frames from video at every offset
    indices = range(0, frame_count, offset)
    output = tf.gather(video, tf.convert_to_tensor(indices))
    return output

def _video_resize(video, height, width):
    return tf.map_fn(lambda img: _resize_image(img, height, width), video)


#import pdb;
def preprocess_for_eval(input_data_tensor, clip_length, sets, size):
    #return input_data_tensor
    # input_data_tensor = tf.tile(input_data_tensor, [1,10,1,1,1])
    shape = input_data_tensor.shape
    #import pdb; pdb.set_trace()
    # input_data_tensor = tf.reshape(input_data_tensor, (sets, 10, clip_length, shape[2].value, shape[3].value, shape[4].value))
    # input_data_tensor = tf.transpose(input_data_tensor, (0,2,1,3,4,5))

    input_data_tensor = tf.pad(tf.expand_dims(input_data_tensor, axis=2), [[0,0],[0,0],[0,9],[0,0],[0,0],[0,0]], 'CONSTANT')
    #tf.concat([input_data_tensor, tf.zeros([shape[0].value, shape[1].value*9, shape[2].value, shape[3].value, shape[4].value])], axis=1)
    #print input_data_tensor
    vid_clips = tf.map_fn(lambda clip: tf.map_fn(lambda repeated_frame: oversample(repeated_frame, [size, size]), clip), input_data_tensor)
    vid_clips = tf.reshape(vid_clips, (sets, tf.multiply(clip_length, 10), size, size, shape[4].value))

    return vid_clips


def preprocess_for_train(input_data_tensor, size):

    vid_clips = tf.map_fn(lambda clip: tf.map_fn(lambda img: _resize_image(img, size, size), clip), input_data_tensor)

    return vid_clips



def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):



    clip_length = 16
    offset      = 8
    size        = size[0]
    input_data_tensor = tf.to_float(input_data_tensor)
    # Possibly _B_MEAN, _G_MEAN, _R_MEAN


    input_data_tensor = _video_resize(input_data_tensor, 240, 320)    # tf.map_fn(lambda img: _image_resize(img, 240, 320), input_data_tensor)
    #pdb.set_trace()
    input_data_tensor = tf.map_fn(lambda img: _mean_image_subtraction(tf.to_float(img), [_R_MEAN, _G_MEAN, _B_MEAN]), input_data_tensor)
    #pdb.set_trace()

    sets = tf.cast(tf.divide(frames, offset), tf.int32)
    excess_frames = tf.subtract(frames, tf.multiply(sets, offset)) #frames - (sets*offset)+offset
    sets = tf.subtract(sets, 1)
#    sets = tf.cast(tf.divide(frames, clip_length), tf.int32)
#    excess_frames = tf.subtract(frames, tf.multiply(sets, clip_length))
    clips_indices = tf.range(tf.multiply(sets, clip_length))                    # [0,1,2,3,4,5,...15,16,17,18,...31, 32, 33, 34,...  47, 48, 49...]


    subtractions = tf.range(sets)
    subtractions = tf.tile(subtractions, [clip_length])
    subtractions = tf.reshape(subtractions, (clip_length, sets))
    subtractions = tf.transpose(subtractions)
    subtractions = tf.reshape(subtractions, [tf.multiply(clip_length, sets)])
    subtractions = tf.multiply(subtractions, offset)                            # [0,0,0,0,0,0,...0,  8,8,8,... 8, 16, 16, 16,...  16, 24, 24, ...]

    clips_indices = tf.subtract(clips_indices, subtractions)                    # [0,1,2,3,4,5,...15, 8,9,10...23, 16, 17, 18...., 31, 24, 25...., frames-16, frames-15, frames-14, ..., frames-2, frames-1]


    final_ind, sets = tf.cond( tf.equal(excess_frames, 0),
                         lambda: (tf.convert_to_tensor([], dtype=tf.int32), sets),
                         lambda: (tf.range(tf.subtract(frames, clip_length), frames), tf.add(sets, 1)) )

    clips_indices = tf.concat(values=[clips_indices, final_ind], axis=0)


    clips_indices = tf.reshape(clips_indices, (sets, clip_length))

    input_data_tensor = tf.gather(input_data_tensor, clips_indices)


#    import pdb; pdb.set_trace()
    input_data_tensor = tf.slice(input_data_tensor, [0,0,0,0,0], tf.stack([10, clip_length, height, width, channel]))
    input_data_tensor = tf.reshape(input_data_tensor, tf.stack([10, clip_length, 240, 320, 3]))
#    import pdb; pdb.set_trace()
    if istraining:
        vid_clips = preprocess_for_train(input_data_tensor, size)
    else:
        vid_clips = preprocess_for_eval(input_data_tensor, clip_length, 10, size)#sets, size)

    labels_tensor = tf.tile( [label], [seq_length])

    return vid_clips, labels_tensor



if __name__=='__main__':
    input_data_tensor = tf.range(60*240*320*3)
    input_data_tensor = tf.reshape(input_data_tensor, (60,240,320,3))
    vid_clips = preprocess(input_data_tensor, 60, 240, 320, 3, 160, 51, 50, [224,224], 11, False)
    sess = tf.Session()
    vc = sess.run(vid_clips)
    import pdb; pdb.set_trace()
