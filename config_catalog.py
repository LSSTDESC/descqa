# This python script sets the catalog configurations

# ---- DO NOT change this section ----
import os as _os

class _CatalogConfig():
    def __init__(self, reader, **kwargs):
        if not isinstance(reader, basestring):
            raise ValueError('`reader` must be a string ')
        self.reader = reader
        _prohibited_leys = ('base_catalog_dir',)
        if any (k in kwargs for k in _prohibited_leys):
            raise ValueError('Do not manually set the following keys: {}'.format(', '.join(_prohibited_leys)))
        self.kwargs = kwargs

    def set_data_dir(self, dirpath):
        self.kwargs['base_catalog_dir'] = dirpath
        if 'fn' in self.kwargs: # old style, should remove in the future
            self.kwargs['fn'] = _os.path.join(dirpath, self.kwargs['fn'])

# ---- End of DO NOT CHANGE ----

SHAM_LiWhite = _CatalogConfig('SHAMGalaxyCatalog', match_to='LiWhite',description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and has been tuned to reproduce the Li & White stellar mass function. The abundance matching technique, also known as subhalo abundance matching (SHAM), is a generic scheme to connect one galaxy property (e.g., stellar mass or luminosity) with one halo property (e.g., halo mass) by assuming an approximately monotonic relation between these two properties.  The two properties are matched at the same cumulative number density, and the resulting galaxy catalog, by explicit construction, preserves the input stellar mass (or luminosity) function.')

SHAM_MB2 = _CatalogConfig('SHAMGalaxyCatalog', match_to='MB2',
        description='This catalog is similar to SHAM Li White but has been tuned to the stellar mass function measured from a hydro-simulation, MassiveBlackII.')

CAM_LiWhite = _CatalogConfig('YaleCAMGalaxyCatalog', 
        fn='yale_cam_age_matching_LiWhite_2009_z0.00.hdf5',
        description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and has been tuned to reproduce the Li & White stellar mass function. The galaxy catalog was created using the conditional abundance matching technique described in Hearin et al. 2014.')

CAM_MB2 = _CatalogConfig('YaleCAMGalaxyCatalog', 
        fn='yale_cam_age_matching_MBII_z0.00.hdf5',
        description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and has been tuned to reproduce the Li & White stellar mass function. The galaxy catalog was created using the conditional abundance matching technique described in Hearin et al. 2014.')

Galacticus = _CatalogConfig('GalacticusGalaxyCatalog',
        filename='galacticus_anl_mstar1e7_zrange.hdf5',
        description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and employs a semi-analytic model Galacticus. Galacticus models the baryonic physics of galaxy growth within an evolving, merging hierarchy of dark matter halos. Baryonic processes (including gas cooling and inflow, star formation, stellar and AGN feedback, and galaxy merging) are described by a collection of differential equations which are integrated to a specified tolerance along each branch of the merger tree, plus instantaneous transformations associated with merger events. The result is a catalog of galaxies at all redshifts including both physical properties (stellar masses, sizes, morphologies) and observational properties (e.g. luminosities in any specified bandpass. The parameters of the model are constrained through either particle swarm optimization or MCMC techniques to match a wide variety of data on the galaxy population.')

MB2 = _CatalogConfig('MB2GalaxyCatalog', fn='catalog.hdf5.MB2',description='NA')

SAG = _CatalogConfig('SAGGalaxyCatalog', fn='SAGcatalog.sag',description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and employs a semi-analytic approach to galaxy formation. This particular approach is based on the model developed by Springel et al. 2001}, which, as is usual with semi-analytic models, combines merger trees extracted from a dark matter cosmological simulation with a set of coupled differential equations for the baryonic processes taking place within these merger trees as time evolves.')

iHOD_LiWhite = _CatalogConfig('iHODGalaxyCatalog', fn='iHODcatalog_lw09.h5.iHOD',description='This catalog is based on the dark matter only (DMO) version of the MB-2 simulation, and has been tuned to reproduce the Li & White stellar mass function. The iHOD model aims to provide a probabilistic mapping between halos and galaxies, assuming that the enormous diversity in the individual galaxy assembly histories inside similar halos would reduce to a stochastic scatter about the mean galaxy-to-halo connection by virtue of the central limit theorem. Building on the global HOD parameterization of Leauthaud et al. 2011, the iHOD formalism was developed to solve the mapping between galaxy stellar mass and halo mass, using the spatial clustering and the galaxy-galaxy lensing of galaxies in SDSS.  Compared to the traditional HOD methods, iHOD can include ~80% more galaxies while taking into account the stellar mass incompleteness of galaxy samples in a self-consistent fashion.')

iHOD_MB2 = _CatalogConfig('iHODGalaxyCatalog', fn='iHODcatalog_mb2.h5.iHOD',description='This catalog is similar to iHOD LiWhite but has been tuned to the stellar mass function measured from a hydro-simulation, MassiveBlackII.')

iHOD_v0 = _CatalogConfig('iHODGalaxyCatalog', fn='iHODcatalog_v0.h5.iHOD',description='This catalog is similar to iHOD but has not been rematched to a specific mass function')
