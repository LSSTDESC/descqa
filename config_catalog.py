# This python script sets the catalog configurations

import os as _os

_CATALOG_DIR = '/project/projectdirs/lsst/descqacmu/catalog'
_READER_DIR = '/project/projectdirs/lsst/descqacmu/src/reader'

class _CatalogConfig():
    def __init__(self, reader, **kwargs):
        if not isinstance(reader, basestring):
            raise ValueError('`reader` must be a string ')
        self.reader = reader
        _prohibited_leys = ('base_catalog_dir',)
        if any (k in kwargs for k in _prohibited_leys):
            raise ValueError('Do not manually set the following keys: {}'.format(', '.join(_prohibited_leys)))
        if 'fn' in kwargs and len(kwargs) == 1: # old style
            kwargs['fn'] = _os.path.join(_CATALOG_DIR, kwargs['fn'])
        else:
            kwargs['base_catalog_dir'] = _CATALOG_DIR
        self.kwargs = kwargs
    

# configurations below

SHAM_LiWhite = _CatalogConfig('SHAMGalaxyCatalog', match_to='LiWhite')

SHAM_MB2 = _CatalogConfig('SHAMGalaxyCatalog', match_to='MB2')

CAM_LiWhite = _CatalogConfig('YaleCAMGalaxyCatalog', 
        fn='yale_cam_age_matching_LiWhite_2009_z0.00.hdf5')

CAM_MB2 = _CatalogConfig('YaleCAMGalaxyCatalog', 
        fn='yale_cam_age_matching_MBII_z0.00.hdf5')

Galacticus = _CatalogConfig('GalacticusGalaxyCatalog',
        filename='galacticus_anl_mstar1e7_zrange.hdf5')

MB2 = _CatalogConfig('MB2GalaxyCatalog', fn='catalog.hdf5.MB2')

SAG = _CatalogConfig('SAGGalaxyCatalog', fn='SAGcatalog.sag')

iHOD_LiWhite = _CatalogConfig('iHODGalaxyCatalog', fn='iHODcatalog_lw09.h5.iHOD')

iHOD_MB2 = _CatalogConfig('iHODGalaxyCatalog', fn='iHODcatalog_mb2.h5.iHOD')
