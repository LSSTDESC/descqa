#!/usr/bin/env python
"""
DESCQA: LSST DESC QA Framework for mock galaxy catalogs
Copyright (c) 2018 LSST DESC
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'descqa', 'version.py')) as f:
    exec(f.read()) #pylint: disable=W0122

setup(
    name='descqa',
    version=__version__, #pylint: disable=E0602
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
        'Programming Language :: Python :: 3.6',
    ],
    keywords='DESCQA',
    packages=['descqa'],
    install_requires=['future', 'pyyaml', 'jinja2'],
    extras_require={
        'full': ['numpy', 'scipy', 'matplotlib', 'GCR>=0.8.7', 'healpy', 'treecorr', 'camb', 'scikit-learn', 'pandas', 'astropy', 'POT', 'numba',
                 'pyccl', 'CatalogMatcher @ https://github.com/LSSTDESC/CatalogMatcher/archive/master.zip'],
    },
    package_data={'descqa': ['configs/*.yaml', 'data/*']},
)
