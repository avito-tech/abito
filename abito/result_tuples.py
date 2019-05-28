from typing import NamedTuple


TTestResult = NamedTuple('TTestResult', [
    ('statistic', float),
    ('p_value', float),
    ('mean_diff', float),
    ('mean_diff_std', float),
])


MannWhitneyUTestResult = NamedTuple('MannWhitneyUTestResult', [
    ('u_statistic', float),
    ('z_statistic', float),
    ('p_value', float),
])


ShapiroTestResult = NamedTuple('ShapiroTestResult', [
    ('statistic', float),
    ('p_value', float),
])


MedianTestResult = NamedTuple('MedianTestResult', [
    ('statistic', float),
    ('p_value', float),
    ('grand_median', float),
])


LeveneTestResult = NamedTuple('LeveneTestResult', [
    ('statistic', float),
    ('p_value', float),
])


MoodTestResult = NamedTuple('MoodTestResult', [
    ('statistic', float),
    ('p_value', float),
])


BootstrapTestResult = NamedTuple('BootstrapTestResult', [
    ('statistic', float),
    ('p_value', float),
    ('mean_diff', float),
    ('mean_diff_std', float)
])
