from collections import namedtuple
from numpy import nan


TTestResult = namedtuple('TTestResult', [
    'statistic',
    'p_value',
    'mean_diff',
    'mean_diff_std',
])
TTestResult.__new__.__defaults__ = (nan,) * len(TTestResult._fields)


MannWhitneyUTestResult = namedtuple('MannWhitneyUTestResult', [
    'u_statistic',
    'z_statistic',
    'p_value',
])
MannWhitneyUTestResult.__new__.__defaults__ = (nan,) * len(MannWhitneyUTestResult._fields)


ShapiroTestResult = namedtuple('ShapiroTestResult', [
    'statistic',
    'p_value',
])
ShapiroTestResult.__new__.__defaults__ = (nan,) * len(ShapiroTestResult._fields)


MedianTestResult = namedtuple('MedianTestResult', [
    'statistic',
    'p_value',
    'grand_median',
])
MedianTestResult.__new__.__defaults__ = (nan,) * len(MedianTestResult._fields)


LeveneTestResult = namedtuple('LeveneTestResult', [
    'statistic',
    'p_value',
])
LeveneTestResult.__new__.__defaults__ = (nan,) * len(LeveneTestResult._fields)


MoodTestResult = namedtuple('MoodTestResult', [
    'statistic',
    'p_value',
])
MoodTestResult.__new__.__defaults__ = (nan,) * len(MoodTestResult._fields)


BootstrapTestResult = namedtuple('BootstrapTestResult', [
    'statistic',
    'p_value',
    'mean_diff',
    'mean_diff_std'
])
BootstrapTestResult.__new__.__defaults__ = (nan,) * len(BootstrapTestResult._fields)
