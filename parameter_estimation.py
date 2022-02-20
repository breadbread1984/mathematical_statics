#!/usr/bin/python3

import tensorflow as tf;

def moment(x, order = 1):
  assert type(order) is int;
  return tf.math.reduce_mean(x**order);

def uniform_moment_estimation(x = None, fst_moment = None, snd_moment = None):
  assert x is not None or (fst_moment is not None and snd_moment is not None);
  moment1 = moment(x, 1) if x is not None else fst_moment;
  moment2 = moment(x, 2) if x is not None else snd_moment;
  a = moment1 - tf.math.sqrt(3 * (moment2 - moment1 ** 2));
  b = moment1 + tf.math.sqrt(3 * (moment2 - moment1 ** 2));
  return a,b;

def normal_moment_estimation(x = None, fst_moment = None, snd_moment = None):
  assert x is not None or (fst_moment is not None and snd_moment is not None);
  moment1 = moment(x, 1) if x is not None else fst_moment;
  moment2 = moment(x, 2) if x is not None else snd_moment;
  u = moment1;
  sigma2 = moment2 - moment1 ** 2;
  return u, sigma2;

if __name__ == "__main__":
  x = tf.random.uniform(minval = 3, maxval = 5, shape = (100,), dtype = tf.float32);
  print(uniform_moment_estimation(x));
  x = tf.random.normal(mean = 0.5, stddev = 1.2, shape = (100,), dtype = tf.float32);
  print(normal_moment_estimation(x));
