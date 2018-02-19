# -*- coding: utf-8 -*-
from os import path
from codecs import open
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name='nn4omtf',
        version='1.0.0dev1',
        description='Neural Networks tools for OMTF',
        long_description=long_description, # Load form README.rst
        url='https://github.com/jlysiak/fuw-nn4omtf',
        author='Jacek Łysiak',
        author_email='jaceklysiako.o@gmail.com',
        keywords='neuralnetworks cms physics muon',
        license='MIT',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3.6'
            ],
        python_requires='>=3.6',
        install_requires=['tensorflow==1.4.0', 'numpy', 'matplotlib'], # Those are obligatory. ROOT lib is required in one sub-lib, however it's hard to install
        packages=find_packages()
    )
