# This python script sets the validation configurations

# ---- DO NOT change this section ----
import os as _os

class _ValidationConfig():
    def __init__(self, module, **kwargs):
        if not isinstance(module, basestring):
            raise ValueError('`module` must be a string')
        self.module = module
        _prohibited_leys = ('test_name', 'catalog_name', 'base_output_dir', 'base_data_dir')
        if any (k in kwargs for k in _prohibited_leys):
            raise ValueError('Do not manually set the following keys: {}'.format(', '.join(_prohibited_leys)))
        self.kwargs = kwargs

    def set_data_dir(self, dirpath):
        self.kwargs['base_data_dir'] = dirpath
# ---- End of DO NOT CHANGE ----

smf_LiWhite = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='LiWhite2009',
         bins=(7.0,12.0,26),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(7.0,12.0),
         description='We calculate the stellar-mass density as a function of the total stellar mass for each galaxy. Stellar masses are defined as the mass locked up in long-lived stars and stellar remnants (the most common definition).  For the SAM models, the total stellar mass is the sum of the disk and spheroid components. The densities are derived from the number counts of galaxies in each stellar mass bin, divided by the simulation volume. These densities are compared with the data from Li and White 2009.'
)


smf_MB2 = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='MassiveBlackII',
         bins=(7.0,12.0,26),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(9.0,12.0),
         description='We calculate the stellar-mass density as a function of the total stellar mass for each galaxy. Stellar masses are defined as the mass locked up in long-lived stars and stellar remnants (the most common definition).  For the SAM models, the total stellar mass is the sum of the disk and spheroid components. The densities are derived from the number counts of galaxies in each stellar mass bin, divided by the simulation volume. These densities are compared with the data from the MassiveBlackII simulation.'
)

smhm_MB2 = _ValidationConfig('StellarMassHaloMassTest',
         observation='MassiveBlackII',
         bins=(7.0,15.0,25),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(8.0,15.0),
         description=''
)

hmf_T = _ValidationConfig('HaloMassFunctionTest',
         observation='Tinker',
         ztest=0.05,
         bins=(7.0,15.0,25),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(5.e10,1.e15),
         description='The mass distribution of halos is one of the essential components of precision cosmology, and occupies a central place in the paradigm of structure formation.  There are two common ways to define halos in a simulation.  One is based on identifying overdense regions above a certain threshold.  The other method, the FOF algorithm, is based on finding neighbors of particles and neighbors of neighbors as defined by a given separation distance. In DESCQA, we calculate the halo mass function from each catalog, and compare it against some well-established analytic fits in the literature.  We assume Poisson error bars.  We use the Bhattacharya et al. 2001 fit for the FOF halos, and Tinker et al. 2008 fit for the case of SO halos.'
)

color_DEEP2 = _ValidationConfig('ColorDistributionTest',
         data_dir='DEEP2',
         data_name='DEEP2',
         raw_data_fname='/global/projecta/projectdirs/lsst/descqa/data/rongpu/DEEP2_uniq_Terapix_Subaru.fits',
         colors=['u-g','g-r','r-i','i-z'],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=24.1,
         zlo=0.65,
         zhi=0.75,
         load_validation_catalog_q = True,
         summary='L2Diff',
         description='For each of the mock catalogs, we calculate the distributions of <i>u-g</i>, <i>g-r</i>, <i>r-i</i>, and <i>i-z</i> colors, and compare with observations.  The SDSS dataset includes <i>ugriz</i> photometry and spectroscopic redshifts from the SDSS main galaxy sample Gunn98, York2000. The SDSS dataset is most complete in 0.07<z<0.09. The comparison of color distributions are done in these redshift ranges. This DEEP dataset (compiled by Zhou et al. 2017 in prep.) includes CFHT MegaCam <i>ugriz</i> photometry from CFHTLS Hudelot12 and Subaru Y-band photometry, and cross-matched with DEEP2 Newman13 and DEEP3 Cooper11,Cooper12 redshifts. The CFHTLS+Subaru+DEEP2/3 dataset is most complete in 0.6<z<0.85. The comparison of color distributions are done in these redshift ranges.',
)

color_SDSS = _ValidationConfig('ColorDistributionTest',
         data_dir='SDSS',
         data_name='SDSS',
         raw_data_fname='/global/projecta/projectdirs/lsst/descqa/data/rongpu/SpecPhoto_sdss_extinction_corrected_trimmed.fit',
         colors=['u-g','g-r','r-i','i-z'],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=17.77,
         zlo=0.045,
         zhi=0.055,
         load_validation_catalog_q = True,
         summary='L2Diff',
         description='For each of the mock catalogs, we calculate the distributions of <i>u-g</i>, <i>g-r</i>, <i>r-i</i>, and <i>i-z</i> colors, and compare with observations. The SDSS dataset includes <i>ugriz</i> photometry and spectroscopic redshifts from the SDSS main galaxy sample Gunn98, York2000. The SDSS dataset is most complete in 0.07<z<0.09. The comparison of color distributions are done in these redshift ranges.  The DEEP dataset (compiled by Zhou et al. 2017 in prep.) includes CFHT MegaCam <i>ugriz</i> photometry from CFHTLS Hudelot12 and Subaru Y-band photometry, and cross-matched with DEEP2 Newman13 and DEEP3 Cooper11,Cooper12 redshifts. The CFHTLS+Subaru+DEEP2/3 dataset is most complete in 0.6<z<0.85. The comparison of color distributions are done in these redshift ranges.',
)

wprp_SDSS_m98 = _ValidationConfig('WprpTest',
         datafile='SDSS/wprp_Reddick-Tinker_sm-9.8.dat',
         dataname='SDSS',
         sm_cut=10.0**9.8,
         zmax=40.0,
         rbins=(-1.0,1.3,13),
         njack=10,
         summary='L2Diff',
         description='For each of the mock catalogs, we calculate the projected two-point correlation function, w<sub>p</sub>(r<sub>p</sub>), in the thin-plane approximation.  We use the catalog at one single epoch and then add redshift space distortion along one spatial axis (z-axis).  We then calculate the projected pair counts, with a projection depth of 80 Mpc/h. We assume periodic boundary conditions for all three spatial axes. We estimate the sample variance of w<sub>p</sub>(r<sub>p</sub>) using the jackknife technique.'
)

wprp_MB2_m98 = _ValidationConfig('WprpTest',
         datafile='MASSIVEBLACKII/wprp_sm-9.8.dat',
         dataname='MB2',
         sm_cut=10.0**9.8,
         zmax=40.0,
         rbins=(-1.0,1.3,13),
         njack=10,
         summary='L2Diff',
         description='For each of the mock catalogs, we calculate the projected two-point correlation function, w_p(r_p), in the thin-plane approximation.  We use the catalog at one single epoch and then add redshift space distortion along one spatial axis (z-axis).  We then calculate the projected pair counts, with a projection depth of 80 Mpc/h. We assume periodic boundary conditions for all three spatial axes. We estimate the sample variance of w_p(r_p) using the jackknife technique.'
)

