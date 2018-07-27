# -*- coding: utf-8 -*-
from os import path
from codecs import open
from setuptools import setup, find_packages
from sys import argv

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
setup(
        name='nn4omtf',
        version='1',
        description='Neural Networks tools for OMTF',
        long_description=long_description, # Load form README.rst
        url='https://github.com/jlysiak/fuw-nn4omtf',
        author='Jacek Łysiak',
        author_email='jaceklysiako.o@gmail.com',
        keywords='neuralnetworks cms physics muon',
        license='MIT',
        classifiers=[
            'Programming Language :: Python :: 3.6'
            'Programming Language :: Python :: 3.5'
            ],
        python_requires='>=3.5',
        install_requires=['seaborn', 'numpy', 'matplotlib', 'ipython'], 
        # ^---- + ROOT + TF
        packages=find_packages(),
    )
