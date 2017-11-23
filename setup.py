#!/usr/bin/env python
"""
Catalog repo for LSST DESC
Copyright (c) 2017 LSST DESC
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'descqa', 'version.py')) as f:
    exec(f.read())

setup(
    name='descqa',
    version='2.3.0',
    description='DESCQA: LSST DESC QA Framework for mock galaxy catalogs',
    url='https://github.com/LSSTDESC/descqa',
    author='LSST DESC',
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
    install_requires=['pyyaml'],
    extras_require = {
        'full':  ['future', 'numpy', 'scipy', 'matplotlib', 'healpy', 'GCR>=0.6.1'],
    },
    package_data={'descqa': ['configs/*.yaml', 'data/*']},
)
