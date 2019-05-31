from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='abito',
    version='0.1',
    license='MIT',
    description='Package for hypothesis testing in A/B-experiments',
    author='Danila Lenkov',
    author_email='dlenkoff@gmail.com',
    url='https://github.com/avito-tech/abito',
    packages=['abito'],
    install_requires=required,
)
