#!/usr/bin/env python

from distutils.core import setup

setup(name='nnomtf',
        version='1.0',
        description='Neural Networks tools for OMTF',
        author='Jacek Łysiak',
        author_email='jaceklysiako.o@gmail.com',
        packages=['nnomtf', 'nnomtf.utils', 'nnomtf.dataset', 'nnomtf.network']
    )
