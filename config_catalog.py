# This python script sets the catalog configurations

import os as _os

_CATALOG_DIR = '/project/projectdirs/lsst/descqacmu/catalog'
_READER_DIR = '/project/projectdirs/lsst/descqacmu/src/reader'

class _CatalogConfig():
    def __init__(self, reader, file):
        if not _os.path.isfile(_os.path.join(_READER_DIR, reader+'.py')):
            raise ValueError('reader module {} does not exist in {}.'.format(reader, _READER_DIR))
        self.reader = reader
        
        self.file = _os.path.join(_CATALOG_DIR, file)
        if not _os.path.exists(self.file):
            raise ValueError('catalog file {} does not exist'.format(self.file))
    

# configurations below

SHAM_LiWhite = _CatalogConfig('SHAMGalaxyCatalog', 'SHAM_0.94118.npy')

SHAM_MB2 = _CatalogConfig('SHAMGalaxyCatalog', 'SHAM_0.94118_MBII.npy')

CAM_LiWhite = _CatalogConfig('YaleCAMGalaxyCatalog', 
        'yale_cam_age_matching_LiWhite_2009_z0.0.hdf5')

CAM_MB2 = _CatalogConfig('YaleCAMGalaxyCatalog', 
        'yale_cam_age_matching_MBII_z0.0.hdf5')

Galacticus = _CatalogConfig('GalacticusGalaxyCatalog',
        'galacticus_mb2_anl.hdf5.galacticus')

MB2 = _CatalogConfig('MB2GalaxyCatalog', 'catalog.hdf5.MB2')

SAG = _CatalogConfig('SAGGalaxyCatalog', 'SAGcatalog.sag')

iHOD = _CatalogConfig('iHODGalaxyCatalog', 'iHODcatalog_v0.h5.iHOD')
