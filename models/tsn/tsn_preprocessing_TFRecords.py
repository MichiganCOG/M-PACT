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
                                       align_corners=True)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image



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


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_height,
                         resize_width):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_height: Height for the intermediate resize
        resize_width: Width for the intermediate resize
    Returns:
        A preprocessed image.
    """
    image = tf.to_float(image)
    image = resize_image(image, resize_height, resize_width)
    image = resize_image(image, output_height, output_width)

    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval_video(video, output_height, output_width, resize_height, resize_width):
    """Preprocesses the given video for evaluation.
    Args:
        video: A `Tensor` representing an video of arbitrary size.
        output_height: The height of the video after preprocessing.
        output_width: The width of the video after preprocessing.
        resize_height: Height for the intermediate resize
        resize_width: Width for the intermediate resize
    Returns:
        A preprocessed video.
    """

    video = tf.map_fn(lambda frame: preprocess_for_eval_image(frame, resize_height, resize_width), video)
    video = tf.pad(tf.expand_dims(video, axis=1), [[0,0],[0,9],[0,0],[0,0],[0,0]], 'CONSTANT')
    #tf.concat([input_data_tensor, tf.zeros([shape[0].value, shape[1].value*9, shape[2].value, shape[3].value, shape[4].value])], axis=1)
    #print input_data_tensor
    video = tf.map_fn(lambda repeated_frame: oversample(repeated_frame, [output_height, output_width]), video)
#    import pdb; pdb.set_trace()
    video = tf.reshape(video, (-1, 10, output_height, output_width, 3))

    return video



def preprocess_for_eval_image(image, resize_height, resize_width):
    """Preprocesses the given image for evaluation.
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_height: Height for the intermediate resize
        resize_width: Width for the intermediate resize
    Returns:
        A preprocessed image.
    """
    image = tf.to_float(image)
    image = _resize_image(image, resize_height, resize_width)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])




def preprocess_video(video, output_height, output_width,
                     resize_height, resize_width, is_training=False):
    """Preprocesses the given video.
    Args:
        image: A `Tensor` representing a video of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_height: Height for the intermediate resize
        resize_width: Width for the intermediate resize
        is_training: `True` if we're preprocessing the vide for training and
            `False` otherwise.
    Returns:
        A preprocessed video.
    """
    if is_training:
        return tf.map_fn(lambda image: preprocess_for_train(image, output_height, output_width,
                                resize_height, resize_width), input_data_tensor)
    else:
        return preprocess_for_eval_video(video, output_height, output_width,
                            resize_height, resize_width)


def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):



    size = size[0]
    if istraining:
        num_frames_per_video = 1

    else:
        num_frames_per_video = 25



    steps = tf.cast((frames - 1) / (num_frames_per_video - 1), tf.int32)#(frames - 1) / (num_frames_per_video - 1)
    end_frame = tf.minimum(tf.cast(2 + steps * (num_frames_per_video-1), tf.int32), frames+1)
    #import pdb; pdb.set_trace()
    frame_ticks = tf.cond( tf.greater(steps, 0),
        lambda:
                tf.range(1, end_frame, steps),
        lambda:
                tf.tile([1], [num_frames_per_video])
    )

    frame_ticks = frame_ticks - 1
    # if steps > 0:
    #
    # else:
    #     frame_ticks = tf.tile([0], num_frames_per_video)

    #data_mean = np.array((_R_MEAN,_G_MEAN,_B_MEAN))#(103.939, 116.779, 128.68))
    # Check if result is uint8 or float32
    #import pdb; pdb.set_trace()
    input_data_tensor = tf.gather(input_data_tensor, frame_ticks)
    input_data_tensor = tf.to_float(input_data_tensor)
    output_data_tensor = preprocess_video(input_data_tensor, size, size, 256, 340, istraining)
    labels_tensor = tf.tile( [label], [num_frames_per_video])
    return output_data_tensor, labels_tensor
