#!/usr/bin/python3

import tensorflow as tf;

def moment(x, order = 1):
  assert type(order) is int;
  return tf.math.reduce_mean(x**order);

def bernoull9i_moment_estimation(x = None, fst_moment = None, snd_moment = None):
  # NOTE: mean(x) = p, var(x) = p(1-p)
  # moment(x,1) = mean(x) = p, moment(x,2) = var(x) + mean(x)^2 = p(1-p) + p^2
  assert x is not None or (fst_moment is not None and snd_moment is not None);
  moment1 = moment(x, 1) if x is not None else fst_moment;
  moment2 = moment(x, 2) if x is not None else snd_moment;
  p = moment1;
  return p;

def uniform_moment_estimation(x = None, fst_moment = None, snd_moment = None):
  # NOTE: mean(x) = (a+b)/2, var(x) = (b-a)^2/12
  # moment(x,1) = mean(x) = (a+b)/2, moment(x,2) = var(x) + mean(x)^2 = (b-a)^2/12 + (a+b)^2/4
  assert x is not None or (fst_moment is not None and snd_moment is not None);
  moment1 = moment(x, 1) if x is not None else fst_moment;
  moment2 = moment(x, 2) if x is not None else snd_moment;
  a = moment1 - tf.math.sqrt(3 * (moment2 - moment1 ** 2));
  b = moment1 + tf.math.sqrt(3 * (moment2 - moment1 ** 2));
  return a,b;

def normal_moment_estimation(x = None, fst_moment = None, snd_moment = None):
  # NOTE: mean(x) = u, var(x) = sigma^2
  # moment(x,1) = mean(x) = u, moment(x,2) = var(x) + mean(x)^2 = sigma^2 + u^2
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
