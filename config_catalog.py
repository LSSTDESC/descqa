__all__ = ['CATALOG_CONFIG', 'CATALOG_DIR', 'READER_DIR']

CATALOG_DIR = '/project/projectdirs/lsst/descqacmu/catalog'
READER_DIR = '/project/projectdirs/lsst/descqacmu/src/reader'

CATALOG_CONFIG = [ \
    {'name':'SHAM-LiWhite', 'file':'SHAM_0.94118.npy', 'reader':'SHAMGalaxyCatalog'},
    {'name':'SHAM-MB2', 'file':'SHAM_0.94118_MB2.npy', 'reader':'SHAMGalaxyCatalog'},
    {'name':'CAM-LiWhite', 'file':'yale_cam_age_matching_LiWhite_2009_z0.0.hdf5', 'reader':'YaleCAMGalaxyCatalog'},
    {'name':'CAM-MB2', 'file':'yale_cam_age_matching_MBII_z0.0.hdf5', 'reader':'YaleCAMGalaxyCatalog'}
]

