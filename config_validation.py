# This python script sets the validation configurations

import os as _os

_VALIDATION_CODE_DIR = '/project/projectdirs/lsst/descqacmu/src/validation_code'
_VALIDATION_DATA_DIR = '/project/projectdirs/lsst/descqacmu/src/validation_data'

class _ValidationConfig():
    def __init__(self, module, **kwargs):
        if not isinstance(module, basestring):
            raise ValueError('`module` must be a string')
        self.module = module

        _prohibited_leys = ('test_name', 'catalog_name', 'base_output_dir', 'base_data_dir')
        if any (k in kwargs for k in _prohibited_leys):
            raise ValueError('Do not manually set the following keys: {}'.format(', '.join(_prohibited_leys)))
        kwargs['base_data_dir'] = _VALIDATION_DATA_DIR
        self.kwargs = kwargs


# configurations below

smf_LiWhite = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='LiWhite2009',
         bins=(7.0,12.0,26),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
)


smf_MB2 = _ValidationConfig('BinnedStellarMassFunctionTest',
         observation='MassiveBlackII',
         bins=(7.0,12.0,26),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
)

smhm_MB2 = _ValidationConfig('StellarMassHaloMassTest',
         observation='MassiveBlackII',
         bins=(7.0,15.0,36),
         zlo=0.0,
         zhi=0.1,
         summary='L2Diff',
)
         
color_DEEP2 = _ValidationConfig('ColorDistributionTest',
         data_dir='DEEP2', 
         data_name='DEEP2',
         colors=['u-g','g-r','r-i','i-z'],
         color_bin_args=[(-0.5, 2.6, 50), (-0.2, 2.2, 50), (-0.2, 1.7, 50), (-0.5, 0.9, 50)],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=17.77,
         zlo=0.6,
         zhi=0.725,
         # zlo=0.,
         # zhi=1.,
         test_q = True,
         plot_pdf_q = True,
         summary='L2Diff',
)

color_SDSS = _ValidationConfig('ColorDistributionTest',
         data_dir='SDSS', 
         data_name='SDSS',
         colors=['u-g','g-r','r-i','i-z'],
         color_bin_args=[(0.6, 2.5, 70), (0.1, 1.2, 70), (0.15, 0.65, 70), (-0.1, 0.55, 70)],
         translate={'u':'SDSS_u:observed:','g':'SDSS_g:observed:','r':'SDSS_r:observed:','i':'SDSS_i:observed:','z':'SDSS_z:observed:'},
         limiting_band='r',
         limiting_mag=17.77,
         # zlo=0.073,
         # zhi=0.080,
         # zlo=0.,
         # zhi=1.,
         zlo=0.045,
         zhi=0.055,
         test_q = False,
         plot_pdf_q = True,
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

