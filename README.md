# abito
[![Build Status](https://travis-ci.com/avito-tech/abito.svg?branch=master)](https://travis-ci.com/avito-tech/abito)
[![Coverage Status](https://coveralls.io/repos/github/avito-tech/abito/badge.svg?branch=master)](https://coveralls.io/github/avito-tech/abito?branch=master)

Python package for hypothesis testing. Suitable for using in A/B-testing software.
Tested for Python >= 3.5. Based on numpy and scipy.

##### Features
1. Convenient interface to run significance tests.
2. Support of ratio-samples. Linearization included (delta-method).
3. Bootstrapping: can measure significance of any statistic, even quantiles. Multiprocessing is supported.
4. Ntile-bucketing: compress samples to get better performance.
5. Trim: get rid of heavy tails.


## Installation
```
pip install abito
```

## Usage

The most powerful tool in this package is the Sample:
```python
import abito as ab
```

Let's draw some observations from Poisson distribution and initiate Sample instance from them.
```python
import numpy as np

observations = np.random.poisson(1, size=10**6)
sample = ab.sample(observations)
```

Now we can calculate any statistic in numpy-way.
```python
print(sample.mean())
print(sample.std())
print(sample.quantile(q=[0.05, 0.95]))
```

To compare with other sample we can use t_test or mann_whitney_u_test:
```python
observations_control = np.random.poisson(1.005, size=10**6)
sample_control = Sample(observations_control)

print(sample.t_test(sample_control))
print(sample.mann_whitney_u_test(sample_control))
```

### Bootstrap
Or we can use bootstrap to compare any statistic:
```python
sample.bootstrap_test(sample_control, stat='mean', n_iters=100)
```

To improve performance, it's better to provide observations in weighted form: unique values + counts. Or, we can compress samples, using built-in method:
```python
sample.reweigh(inplace=True)
sample_control.reweigh(inplace=True)
sample.bootstrap_test(sample_control, stat='mean', n_iters=10000)
```
Now bootstrap is working lightning-fast. To improve performance further you can set parameter n_threads > 1 to run bootstrapping using multiprocessing.

### Compress
```python
observations = np.random.normal(100, size=10**8)
sample = ab.sample(observations)

compressed = sample.compress(n_buckets=100, stat='mean')

%timeit sample.std()
%timeit compressed.std()
```