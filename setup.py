#!/usr/bin/env python
"""
Catalog repo for LSST DESC
Copyright (c) 2017 LSST DESC
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup

setup(
    name='descqa',
    version=2.3.0,
    description='DESCQA: LSST DESC QA Framework for mock galaxy catalogs',
    url='https://github.com/LSSTDESC/descqa',
    author='Yao-Yuan Mao',
    author_email='yymao.astro@gmail.com',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='DESCQA',
    packages=['descqa'],
    install_requires=['numpy', 'pyyaml', 'requests', 'h5py', 'astropy', 'matplotlib', 'GCR'],
    package_data={'descqa': ['configs/*.yaml', 'data/*']},
)
