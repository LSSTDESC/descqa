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
LIWHITE_STELLAR_MASS=0
MB2_STELLAR_MASS=1
MB2_STELLAR_MASS_HALO_MASS=2
SDSS_COLOR=3

def get_obs_catalog(name):
	if(name == LIWHITE_STELLAR_MASS)		: return ObsCatalog('Li White','LIWHITE/StellarMassFunction/massfunc_data.txt',(0,4))
	if(name == MB2_STELLAR_MASS)			: return ObsCatalog('Massive Black II','MASSIVEBLACKII/StellarMassFunction/massfunc_data.txt',(0,1))
	if(name == MB2_STELLAR_MASS_HALO_MASS)	: return ObsCatalog('Massive Black II','MASSIVEBLACKII/StellarMassHaloMass/tab.txt',(0,2,3,4))
	if(name == SDSS_COLOR)					: return ObsCatalog('SDSS','SDSS/',(0,1))
	raise Exception('Unknown obs catalog: %d' % name)


# simulation catalogs
CMU_MB2 = 0
GALACTICUS_MB2 = 1
SAG = 2
SHAM_MB2 = 3
SHAM_LIWHITE = 4
YALE_MB2 = 5
YALE_LIWHITE = 6
IHOD = 7

def get_sim_catalog(name):
	if(name is CMU_MB2)			: return SimCatalog('CMU MB2','catalog.hdf5.MB2')
	if(name is GALACTICUS_MB2)  : return SimCatalog('Galacticus MB2','galacticus_mb2_anl.hdf5.galacticus')
	if(name is SAG)				: return SimCatalog('SAG','SAGcatalog.sag')
	if(name is SHAM_MB2)	    : return SimCatalog('SHAM MB2','SHAM_0.94118_MBII.npy')
	if(name is SHAM_LIWHITE)    : return SimCatalog('SHAM Li White','SHAM_0.94118.npy')
	if(name is YALE_MB2)	  	: return SimCatalog('Yale MB2','yale_cam_age_matching_MBII_rho-1.0_sigma0.2_z0.0.hdf5.yale')
	if(name is YALE_LIWHITE)    : return SimCatalog('Yale Li White','yale_cam_age_matching_LiWhite_2009_rho-1.0_sigma0.2_z0.0.hdf5.yale')
	if(name is IHOD)	  		: return SimCatalog('iHOD','iHODcatalog_v0.h5.iHOD')
	raise Exception('Unknown sim catalog: %d' % name)

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
