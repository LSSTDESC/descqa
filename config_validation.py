__all__ = ['VALIDATION_CONFIG', 'VALIDATION_CODE_DIR', 'VALIDATION_DATA_DIR']

VALIDATION_CODE_DIR = '/project/projectdirs/lsst/descqacmu/src/validation_code'
VALIDATION_DATA_DIR = '/project/projectdirs/lsst/descqacmu/src/validation_data'

VALIDATION_CONFIG = [ \
        {'name':'smf-LiWhite',
         'module':'BinnedStellarMassFunctionTest',
         'data_dir':'LIWHITE/StellarMassFunction',
         'data_args':{'file':'massfunc_dataerr.txt', 'name':'LiWhite', 'usecols':(0,5,6)},
         'test_args':{'bins':(7.0,12.0,26),'zlo':0.0,'zhi':0.1,'summary':'L2Diff'},#bins is (logmin, logax, number of bins)
        }, # end of smf-LiWhite
        
        {'name':'smf-MB2',
         'module':'BinnedStellarMassFunctionTest',
         'data_dir':'LIWHITE/StellarMassFunction',
         'data_args':{'file':'massfunc_dataerr.txt', 'name':'MB-II', 'usecols':(0,1,2)},
         'test_args':{'bins':(7.0,12.0,26),'zlo':0.0,'zhi':0.1,'summary':'L2Diff'},#bins is (logmin, logax, number of bins)
        }, # end of smf-MB2

        {'name':'color-DEEP2',
         'module':'ColorDistributionTest',
         'data_dir':'DEEP2',
         'data_args':{'file':'deep2_g-r_z_0.6_0.725_bins_-0.2_2_40.txt', 'name':'DEEP2'},
         'test_args':{'bins':(-0.2,2.,40),'zlo':0.0,'zhi':0.75,'band1':'SDSS_g:observed:','band2':'SDSS_r:observed:','summary':'L2Diff'},#bins is (logmin, logax, number of bins)
        }, # end of color-DEEP2

        {'name':'wprp-9.8', 
         'module':'WprpTest',
         'data_dir':'.',
         'data_args':{'mb2':'MASSIVEBLACKII/wprp_sm-9.8.dat', 'sdss':'SDSS/wprp_Reddick-Tinker_sm-9.8.dat'},
         'test_args':{'sm_cut': 10.0**9.8, 'zmax':40.0, 'rbins':(-1.0,1.3,13), 'njack':10}, # for wp(rp), the numbers are all in h=1 units.
        }, # end of wprp
]
