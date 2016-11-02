#!/usr/bin/env python

import os, sys

# Root directory of DESC QA stuff.

DESCQARootDir = os.environ['DESCQA_ROOT_DIR']
DESCQACatalogDir = os.environ['DESCQA_SIM_DATA_DIR']
DESCQAObsDir = os.environ['DESCQA_OBS_DATA_DIR']

# Catalog test path and subdirectories.
DESCQACatalogTestDir     = os.path.join(DESCQARootDir, 'catalog')
DESCQACatalogConfigDir   = os.path.join(DESCQACatalogTestDir, 'configs')
DESCQACatalogFunctionDir = os.path.join(DESCQACatalogTestDir, 'functions')
DESCQACatalogScriptDir   = os.path.join(DESCQACatalogTestDir, 'scripts')

# Make sure we pick up the local test config

sys.path.append(os.environ['PWD'])

class ObsCatalog:
	def __init__(self,name,path,columns):
		self.name = name
		self.path = os.path.join(DESCQAObsDir,path)
		self.columns = columns

class SimCatalog:
	def __init__(self,name,path):
		self.name = name
		self.path = os.path.join(DESCQACatalogDir,path)


# observation catalogs
LIWHITE_STELLAR_MASS='Li White'
MB2_STELLAR_MASS='Massive Black II'

def get_obs_catalog(name):
	if(name is LIWHITE_STELLAR_MASS)				: return ObsCatalog(name,'LIWHITE/StellarMassFunction/massfunc_data.txt',(0,4))
	if(name is MB2_STELLAR_MASS)						: return ObsCatalog(name,'MASSIVEBLACKII/StellarMassFunction/massfunc_data.txt',(0,1))
	if(name is MB2_STELLAR_MASS_HALO_MASS)	: return ObsCatalog(name,'MASSIVEBLACKII/StellarMassHaloMass/tab.txt',(0,2,3,4))
	raise 'Unknown catalog:',name


# simulation catalogs
CMU_MB2 = 'CMU MB2'
GALACTICUS_MB2 = 'Galacticus MB2'
SAG = 'SAG'
SHAM_MB2 = 'SHAM MB2'
SHAM_LIWHITE = 'SHAM Li White'
YALE_MB2 = 'Yale MB2'
YALE_LIWHITE = 'Yale Li White'
IHOD = 'iHOD'

def get_sim_catalog(name):
	if(name is CMU_MB2)					: return SimCatalog(name,'catalog.hdf5.MB2')
	if(name is GALACTICUS_MB2)  : return SimCatalog(name,'galacticus_mb2_anl.hdf5.galacticus')
	if(name is SAG_MB2)					: return SimCatalog(name,'SAGcatalog.sag')
	if(name is SHAM_LIWHITE)	  : return SimCatalog(name,'SHAM_0.94118.npy')
	raise 'Unknown catalog:',name

"""
With the changes above, the par files will change from the current version as follows:

--- current version

import os

sim_folder = os.environ['DESCQA_SIM_DATA_DIR']
obs_folder = os.environ['DESCQA_OBS_DATA_DIR']

sim_catalog = sim_folder + '/catalog.hdf5.MB2'
obs_catalog = obs_folder + '/LIWHITE/StellarMassFunction/massfunc_data.txt'
obs_catalog_usecols = (0,4)

zlo = 0.05
zhi = 0.15

--- version 2

from descqaGlobalConfig import *

sim = get_sim_catalog(CMU_MB2)
sim_catalog = sim.path
obs = get_obs_catalog(LIWHITE)
obs_catalog = obs.path
obs_catalog_usecols = obs.columns

zlo = 0.05
zhi = 0.15


--- version 3

from descqaGlobalConfig import *

sim_catalog = get_sim_catalog(CMU_MB2)
obs_catalog = get_obs_catalog(LIWHITE)

"""

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



if __name__ == '__main__':
	o = get_obs_catalog(LIWHITE_STELLAR_MASS)
	print o
	print o.path
