from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='abito',
    version='0.1',
    description='Package for significance testing in A/B experiments',
    author='Danila Lenkov',
    author_email='dlenkoff@gmail.com',
    packages=['abito'],
    install_requires=required,
)
