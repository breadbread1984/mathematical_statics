#!/usr/bin/python3

import tensorflow_probability as tfp;
from utils import *;

def mean_interval(x, conf = 0.95):
  assert len(x.shape) == 1;
  assert type(conf) is float and 0 <= conf <= 1;
  # NOTE: (sample_mean(x) - total_mean) / (sample_stdvar(x) / sqrt(n)) ~ t(n-1)
  sample_mean = mean(x);
  sample_stdvar = stdvar(x);
  n = x.shape[0];
  alpha = 1 - conf;
  # NOTE: tensorflow probability doesn't implement of student quantile yet
  '''
  student_dist = tfp.distributions.StudentT(df = n - 1, loc = 0, scale = 1);
  t = student_dist.quantile(1 - alpha / 2);
  '''
  # NOTE: currently we have to use scipy's implement of quantile
  from scipy import stats;
  t = stats.t(df = n - 1).ppf(1 - alpha / 2);

  low_bound = sample_mean - sample_stdvar / tf.math.sqrt(tf.cast(n, dtype = tf.float32)) * t;
  upper_bound = sample_mean + sample_stdvar / tf.math.sqrt(tf.cast(n, dtype = tf.float32)) * t;
  return low_bound, upper_bound;

def var_interval(x, conf = 0.95):
  assert len(x.shape) == 1;
  assert type(conf) is float and 0 <= conf <= 1;
  # NOTE: (n - 1) sample_var(x) / total_var ~ chi2(n - 1)
  sample_var = var(x);
  n = x.shape[0];
  alpha = 1 - conf;

  chi2_dist = tfp.distributions.Chi2(n - 1);
  c1 = chi2_dist.quantile(alpha / 2)
  c2 = chi2_dist.quantile(1 - alpha / 2);

  low_bound = (n - 1) * sample_var / c2;
  upper_bound = (n - 1) * sample_var / c1;
  return low_bound, upper_bound;

def stdvar_interval(x, conf = 0.95):
  low_bound, upper_bound = var_interval(x, conf);
  return tf.math.sqrt(low_bound), tf.math.sqrt(upper_bound);

def mean_diff_interval(x1, x2, conf = 0.95):
  # NOTE: ((sample_mean(x1) - sample_mean(x2)) - (total_mean(x1) - total_mean(x2)))/(sample_var(x1 union x2) sqrt(1 / n1 + 1 / n2)) ~ t(n1 + n2 - 2)
  sample_mean1 = mean(x1);
  sample_mean2 = mean(x2);
  sample_var1 = var(x1);
  sample_var2 = var(x2);
  n1 = x1.shape[0];
  n2 = x2.shape[0];
  sample_var = ((n1 - 1) * sample_var1 + (n2 - 1) * sample_var2) / (n1 + n2 - 2);
  sample_stdvar = tf.math.sqrt(sample_var);
  alpha = 1 - conf;
  # NOTE: tensorflow probability doesn't implement of student quantile yet
  '''
  student_dist = tfp.distributions.StudentT(df = n1 + n2 - 2, loc = 0, scale = 1);
  t = student_dist.quantile(1 - alpha / 2);
  '''
  # NOTE: currently we have to use scipy's implement of quantile
  from scipy import stats;
  t = stats.t(df = n1 + n2 - 2).ppf(1 - alpha / 2);

  low_bound = sample_mean1 - sample_mean2 - sample_stdvar * t * tf.math.sqrt(1 / n1 + 1 / n2);
  upper_bound = sample_mean1 - sample_mean2 + sample_stdvar * t * tf.math.sqrt(1 / n1 + 1 / n2);
  return low_bound, upper_bound;

if __name__ == "__main__":
  samples = tf.constant([506, 508, 499, 503, 504, 510, 497, 512, 514, 505, 493, 496, 506, 502, 509, 496]);
  print(mean_interval(samples, conf = 0.95));
  print(var_interval(samples, conf = 0.95));
  print(stdvar_interval(samples, conf = 0.95));
  samples1 = tf.random.normal(mean = 500., stddev = 1.10, shape = (10,));
  samples2 = tf.random.normal(mean = 496., stddev = 1.20, shape = (20,));
  print(mean_diff_interval(samples1, samples2, conf = 0.95));
