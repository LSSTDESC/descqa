# DESCQA

[![DOI](https://zenodo.org/badge/47784041.svg)](https://zenodo.org/badge/latestdoi/47784041)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.3847%2F1538--4365%2Faaa6c3-brightgreen.svg)](https://doi.org/10.3847/1538-4365/aaa6c3)
[![arXiv:1709.09665](https://img.shields.io/badge/astro--ph.IM-arXiv%3A1709.09665-B31B1B.svg)](https://arxiv.org/abs/1709.09665)
[![ascl:1804.011](https://img.shields.io/badge/ascl-1804.011-blue.svg?colorB=262255)](http://ascl.net/1804.011)

This repository contains the DESCQA framework that validates simulated galaxy catalogs. For more information about this framework, please check out the [DESCQA paper](https://arxiv.org/abs/1709.09665).

A [web interface](https://portal.nersc.gov/projecta/lsst/descqa/) hosted on NERSC displays recent validation results from the DESCQA framework.

**! Important !** Starting from DESCQA v2 (current version), we have separated the configurations and readers of catalogs from DESCQA and moved them to a standalone repo, the [GCRCatalogs](https://github.com/LSSTDESC/gcr-catalogs) repo. We have also changed much of the validation tests. If you are looking for the catalogs and tests in DESCQA v1 (as presented in the [companion paper](https://arxiv.org/abs/1709.09665)), please see the [v1 subdiectory](v1).


## Quick start guide

1. First of all, try accessing the catalogs! You can find information about the catalogs in [this Confluence page](https://confluence.slac.stanford.edu/x/Z0uKDQ) and [this presentation](https://docs.google.com/presentation/d/1W5lZrQci9J4jaTdLWUIwkPKtq1lbDT3SzTh-YgIkl6k/edit?usp=sharing) ([video](https://youtu.be/4k9Yj6aI1uc)). However, the easiest way to start is to go to https://jupyter.nersc.gov and login with your NERSC account, and follow [this tutorial notebook](https://github.com/LSSTDESC/gcr-catalogs/blob/master/examples/GCRCatalogs%20Demo.ipynb) ([download link](https://raw.githubusercontent.com/LSSTDESC/gcr-catalogs/master/examples/GCRCatalogs%20Demo.ipynb), you can then upload the notebook through the jupyter interface).

2. Once you can access the catalogs, try to make some plots about things you are interested in. You can find [some ideas that have been proposed](https://github.com/LSSTDESC/descqa/labels/validation%20test), but you are more than welcome to come up with new ones!

3. Now that you are able to make some plots, think about how to "validate" the catalogs (i.e., are there any observation/theory data that can be plotted on the same figure for comparison? How to decide whether a catalog is satisfactory?)

4. Now we can integrate your work into the [DESCQA web interface](https://portal.nersc.gov/projecta/lsst/descqa/v2/)! This step is slightly more involved, but you can follow [the instruction here](CONTRIBUTING.md).



## Code structure

- `run_master.sh`: a convenient shell script to run DECSQA
- `fix_web_permission.sh`: a convenient shell script to ensure permissions are set correctly.
- `index.cgi`: CGI script for web interface
- `descqa/`: package that contains all the validation test classes and relevant utilities
- `descqa/configs/`: directory that hosts all validation test config YAML files
- `descqa/data/`: directory that hosts small data files that validation tests need
- `descqaqweb/`: package that contains the web interface
- `descqaqrun/`: package that contains the execution scripts
- `v1`: catalog readers and validation tests for DESCQA v1


## Dependencies

See `install_requires` and `extras_require` in [setup.py](setup.py) for the full installation dependencies.

Note, to *run* DESCQA (v2), [GCRCatalogs](https://github.com/LSSTDESC/gcr-catalogs) is also required.


### Additional dependencies For v1

- Python 2.7 only
- kcorrect
- fast3tree
- https://bitbucket.org/yymao/helpers
