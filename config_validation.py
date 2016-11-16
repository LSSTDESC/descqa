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
)


smf_MB2 = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='MassiveBlackII',
         bins=(7.0,12.0,26),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(7.0,12.0),
)

smhm_MB2 = _ValidationConfig('StellarMassHaloMassTest',
         observation='MassiveBlackII',
         bins=(7.0,15.0,25),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(8.0,15.0),
)

hmf_ST = _ValidationConfig('HaloMassFunctionTest',
         observation='Sheth-Tormen',
         ztest=0.05,
         bins=(7.0,15.0,25),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
         summary_details=True,
         validation_range=(5.e10,1.e15),
)

color_DEEP2 = _ValidationConfig('ColorDistributionTest',
         data_dir='DEEP2',
         data_name='DEEP2',
         colors=['u-g','g-r','r-i','i-z'],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=24.1,
         # zlo=0.6,
         # zhi=0.725,
         # zlo=0.725,
         # zhi=0.85,
         zlo=0.65,
         zhi=0.75,
         # zlo=0.75,
         # zhi=0.85,
         # zlo=0.,
         # zhi=1.,
         load_validation_catalog_q = True,
         summary='L2Diff',
)

color_SDSS = _ValidationConfig('ColorDistributionTest',
         data_dir='SDSS',
         data_name='SDSS',
         colors=['u-g','g-r','r-i','i-z'],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=17.77,
         # zlo=0.073,
         # zhi=0.080,
         # zlo=0.080,
         # zhi=0.088,
         # zlo=0.075,
         # zhi=0.085,
         zlo=0.045,
         zhi=0.055,
         load_validation_catalog_q = True,
         summary='L2Diff',
)

wprp_SDSS_m98 = _ValidationConfig('WprpTest',
         datafile='SDSS/wprp_Reddick-Tinker_sm-9.8.dat',
         dataname='SDSS',
         sm_cut=10.0**9.8,
         zmax=40.0,
         rbins=(-1.0,1.3,13),
         njack=10,
         summary='L2Diff'
)

wprp_MB2_m98 = _ValidationConfig('WprpTest',
         datafile='MASSIVEBLACKII/wprp_sm-9.8.dat',
         dataname='MB2',
         sm_cut=10.0**9.8,
         zmax=40.0,
         rbins=(-1.0,1.3,13),
         njack=10,
         summary='L2Diff'
)

