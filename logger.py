""" BASIC LOGGER WRAPPER CLASS FOR TENSORBOARD USED IN THIS FRAMEWORK """

# Copied from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):
    def __init__(self, log_dir):
        """
        Args:
            :log_dir: Full path to the log directory to be used for current Logger instance
        """

        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def add_scalar_value(self, tag, value, step):
        """
        Args:
            :tag:   Name of scalar variable to be updated
            :value: Scalar value to be updated
            :step:  Current step value for scalar value to be updated
        """

        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
