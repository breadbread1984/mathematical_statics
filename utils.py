#!/usr/bin/python3

import tensorflow as tf;

def mean(x):
  assert len(x.shape) == 1;
  return tf.math.reduce_mean(tf.cast(x, dtype = tf.float32));

def var(x):
  assert len(x.shape) == 1;
  return tf.math.reduce_sum(tf.math.square(tf.cast(x, dtype = tf.float32) - mean(x)))/(x.shape[0] - 1);

def stdvar(x):
  return tf.math.sqrt(var(x));

