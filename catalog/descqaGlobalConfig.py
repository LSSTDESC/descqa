#!/usr/bin/env python

import os, sys

# Root directory of DESC QA stuff.

DESCQARootDir = os.environ['DESCQA_ROOT_DIR']

# Make sure we pick up the local test config

sys.path.append(os.environ['PWD'])

# Locations of files to be read by all DESC QA scripts.

#DESCQAinput   = '/home/ricker/Projects/surveys/lsst/desc/code/descqa/data/'
#simfolder     = DESCQAinput
#obsfolder     = DESCQAinput

# Locations in which to place output files/plots. Current directory is
# preferred for use with FlashTest.

#outputbase    = './'
#plotbase      = './'
#plotfolder    = './'
#mockbase      = './'

# Make sure we pick up all of the DESC QA functions.

#pathsep		= path_sep(/search_path) 
#!PATH		= expand_path('+'+DESCQARootDir+'functions/') + $
#		  pathsep + !PATH
