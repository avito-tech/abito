import pathlib
from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name='abito',
    version='0.0.5',
    license='MIT',
    description='Package for hypothesis testing in A/B-experiments',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Danila Lenkov',
    author_email='dlenkoff@gmail.com',
    url='https://github.com/avito-tech/abito',
    packages=find_packages(exclude=("tests",)),
    install_requires=required,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ]
)
