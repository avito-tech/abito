import pathlib
from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

LONG_DESCRIPTION = (
    'Python package for hypothesis testing. Suitable for using in A/B-testing software. ' 
    'Based on statistical tests from scipy.stats: t-test, Mann-Whitney U, Shapiro-Wilk, Levene, Mood, Median. '
    'Works with weighted samples. Can trim sample tails. Works with Ratio samples.'
)

setup(
    name='abito',
    version='0.0.1',
    license='MIT',
    description='Package for hypothesis testing in A/B-experiments',
    long_description=LONG_DESCRIPTION,
    author='Danila Lenkov',
    author_email='dlenkoff@gmail.com',
    url='https://github.com/avito-tech/abito',
    packages=find_packages(exclude=("tests",)),
    install_requires=required,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: MIT',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ]
)
