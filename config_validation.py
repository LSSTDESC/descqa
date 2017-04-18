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

SMF_LiWhite = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='LiWhite2009',
         bins=(7.0, 12.1, 18),
         validation_range=(10.0**9.1, 10.0**11.8),
         zlo=0.045,
         zhi=0.065,
         description='We calculate the stellar-mass density as a function of the total stellar mass for each galaxy. Stellar masses are defined as the mass locked up in long-lived stars and stellar remnants (the most common definition).  For the SAM models, the total stellar mass is the sum of the disk and spheroid components. The densities are derived from the number counts of galaxies in each stellar mass bin, divided by the simulation volume. These densities are compared with the data from Li and White 2009.'
)


SMF_MBII = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='MassiveBlackII',
         bins=(7.0, 12.1, 18),
         validation_range=(10.0**9.1, 10.0**11.8),
         zlo=0.045,
         zhi=0.065,
         description='We calculate the stellar-mass density as a function of the total stellar mass for each galaxy. Stellar masses are defined as the mass locked up in long-lived stars and stellar remnants (the most common definition).  For the SAM models, the total stellar mass is the sum of the disk and spheroid components. The densities are derived from the number counts of galaxies in each stellar mass bin, divided by the simulation volume. These densities are compared with the data from the MassiveBlackII simulation.'
)


HMF = _ValidationConfig('HaloMassFunctionTest',
         observation='Tinker',
         bins=(8.0, 15.0, 29),
         validation_range=(10.0**12.0, 10.0**14.5),
         ztest=0.0625,
         zlo=0.045,
         zhi=0.065,
         description='The mass distribution of halos is one of the essential components of precision cosmology, and occupies a central place in the paradigm of structure formation.  There are two common ways to define halos in a simulation.  One is based on identifying overdense regions above a certain threshold.  The other method, the FOF algorithm, is based on finding neighbors of particles and neighbors of neighbors as defined by a given separation distance. In DESCQA, we calculate the halo mass function from each catalog, and compare it against some well-established analytic fits in the literature.  We assume Poisson error bars.  We use the Bhattacharya et al. 2001 fit for the FOF halos, and Tinker et al. 2008 fit for the case of SO halos.'
)


SMHM = _ValidationConfig('StellarMassHaloMassTest',
         observation='MassiveBlackII',
         bins=(7.5, 15., 26),
         validation_range=(10.0**12.0, 10.0**14.4),
         zlo=0.045,
         zhi=0.065,
         description='Mean stellar mass as a function of halo mass for host halos.'
)

ColorDist = _ValidationConfig('ColorDistributionTest',
         # data_dir='SDSS',
         # data_name='SDSS',
         sdss_fname='sdss_output_coeffs_z_0.06_0.09.dat',
         colors=['u-g','g-r','r-i','i-z'],
         translate={'u':'SDSS_u:rest:','g':'SDSS_g:rest:','r':'SDSS_r:rest:','i':'SDSS_i:rest:','z':'SDSS_z:rest:'},
         limiting_band='r',
         # limiting_mag=17.77,
         limiting_abs_mag=-20.4,
         zlo=0.045,
         zhi=0.065,
         threshold=0.03,
         description='For each of the mock catalogs, we calculate the distributions of <i>M_u-M_g</i>, <i>M_g-M_r</i>, <i>M_r-M_i</i> and <i>M_i-M_z</i> colors, where the magnitudes are k-corrected absolute magnitudes, and compare with SDSS colors. The SDSS dataset includes <i>ugriz</i> photometry and spectroscopic redshifts from the SDSS main galaxy sample (Gunn98, York2000). SDSS galaxies in the redshift range of 0.06<z<0.09 are used for this comparison.',
)

WpRp_SDSS = _ValidationConfig('WprpTest',
         datafile='SDSS/wprp_Reddick-Tinker_sm-9.8.dat',
         observation='SDSS',
         bins=(-1.0,1.3,12),
         description='For each of the mock catalogs, we calculate the projected two-point correlation function, w<sub>p</sub>(r<sub>p</sub>), in the thin-plane approximation.  We use the catalog at one single epoch and then add redshift space distortion along one spatial axis (z-axis).  We then calculate the projected pair counts, with a projection depth of 80 Mpc/h. We assume periodic boundary conditions for all three spatial axes. We estimate the sample variance of w<sub>p</sub>(r<sub>p</sub>) using the jackknife technique.'
)


WpRp_MBII = _ValidationConfig('WprpTest',
         datafile='MASSIVEBLACKII/wprp_sm-9.8.dat',
         observation='MBII',
         bins=(-1.0,1.3,12),
         description='For each of the mock catalogs, we calculate the projected two-point correlation function, w_p(r_p), in the thin-plane approximation.  We use the catalog at one single epoch and then add redshift space distortion along one spatial axis (z-axis).  We then calculate the projected pair counts, with a projection depth of 80 Mpc/h. We assume periodic boundary conditions for all three spatial axes. We estimate the sample variance of w_p(r_p) using the jackknife technique.'
)
