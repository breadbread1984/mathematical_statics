#!/usr/bin/python3

import tensorflow_probability as tfp;
from utils import *;

def mean_interval(x = None, smean = None, sstdvar = None, snum = None, conf = 0.95):
  assert x is not None or (smean is not None and sstdvar is not None  and snum is not None);
  if x is not None: assert len(x.shape) == 1;
  assert type(conf) is float and 0 <= conf <= 1;
  # NOTE: (sample_mean(x) - total_mean) / (sample_stdvar(x) / sqrt(n)) ~ t(n-1)
  sample_mean = mean(x) if x is not None else smean;
  sample_stdvar = stdvar(x) if x is not None else sstdvar;
  n = x.shape[0] if x is not None else snum;
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

def var_interval(x = None, svar = None, snum = None, conf = 0.95):
  assert x is not None or svar is not None;
  if x is not None: assert len(x.shape) == 1;
  assert type(conf) is float and 0 <= conf <= 1;
  # NOTE: (n - 1) sample_var(x) / total_var ~ chi2(n - 1)
  sample_var = var(x) if x is not None else svar;
  n = x.shape[0] if x is not None else snum;
  alpha = 1 - conf;

  chi2_dist = tfp.distributions.Chi2(n - 1);
  c1 = chi2_dist.quantile(alpha / 2)
  c2 = chi2_dist.quantile(1 - alpha / 2);

  low_bound = (n - 1) * sample_var / c2;
  upper_bound = (n - 1) * sample_var / c1;
  return low_bound, upper_bound;

def stdvar_interval(x = None, svar = None, snum = None, conf = 0.95):
  low_bound, upper_bound = var_interval(x, svar, snum, conf);
  return tf.math.sqrt(low_bound), tf.math.sqrt(upper_bound);

def mean_diff_interval(x1 = None, x2 = None, smean1 = None, smean2 = None, svar1 = None, svar2 = None, snum1 = None, snum2 = None, conf = 0.95):
  assert (x1 is not None and x2 is not None) or (smean1 is not None and smean2 is not None and svar1 is not None and svar2 is not None and snum1 is not None and snum2 is not None);
  if (x1 is not None and x2 is not None):
    assert len(x1.shape) == 1;
    assert len(x2.shape) == 1;
  assert type(conf) is float and 0 <= conf <= 1;
  # NOTE: ((sample_mean(x1) - sample_mean(x2)) - (total_mean1 - total_mean2))/(sample_var(x1 union x2) sqrt(1 / n1 + 1 / n2)) ~ t(n1 + n2 - 2)
  sample_mean1 = mean(x1) if x1 is not None else smean1;
  sample_mean2 = mean(x2) if x2 is not None else smean2;
  sample_var1 = var(x1) if x1 is not None else svar1;
  sample_var2 = var(x2) if x2 is not None else svar2;
  n1 = x1.shape[0] if x1 is not None else snum1;
  n2 = x2.shape[0] if x2 is not None else snum2;
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

def var_ratio_interval(x1, x2, conf = 0.95):
  # NOTE: (sample_var(x1)/sample_var(x2)/(total_var1/total_var2)) ~ F(n1 - 1, n2 - 1)
  sample_var1 = var(x1);
  sample_var2 = var(x2);
  n1 = x1.shape[0];
  n2 = x2.shape[0];
  alpha = 1 - conf;


if __name__ == "__main__":
  samples = tf.constant([506, 508, 499, 503, 504, 510, 497, 512, 514, 505, 493, 496, 506, 502, 509, 496]);
  print(mean_interval(samples, conf = 0.95));
  print(var_interval(samples, conf = 0.95));
  print(stdvar_interval(samples, conf = 0.95));
  print(mean_diff_interval(smean1 = 500., smean2 = 496., svar1 = 1.10**2, svar2 = 1.20**2, snum1 = 10, snum2 = 20, conf = 0.95));
