# abito

### Install
```
pip install git+https://github.com/lnkov/abito.git --upgrade
```

### Features
1. Based on statistical tests from scipy.stats: t-test, Mann-Whitney U, Shapiro-Wilk, Levene, Mood, Median
2. Works with weighted samples
3. Can trim sample tails
4. Works with Ratio samples

### How to use
```python
>>> from abito import RatioSample
>>> sample = RatioSample(num=[1, 2, 3], den=[4, 5, 6])
>>> sample.t_test_1samp(0.5)
TTestResult(statistic=-1.4433756729740654, p_value=0.2857142857142853, mean_diff=-0.10000000000000003, mean_diff_std=0.06928203230275506)

>>> sample_control = RatioSample(num=[1, 2, 8], den=[4, 5, 10])
>>> sample.t_test(sample_control)
TTestResult(statistic=-0.9481011064982815, p_value=0.42240549320152565, mean_diff=-0.1789473684210527, mean_diff_std=0.18874291696797746)
```
