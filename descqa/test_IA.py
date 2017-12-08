"""
Make prediction of gal-gal, gal-shape, and shape-shape correlations
in 3d or projected 2d comoving separation.
Can be used as a DESC-QA validation test.
"""

from __future__ import division 
from __future__ import print_function

import numpy as np
import pylab
import os
from FASTPT import FASTPT
from FASTPT import P_extend 
from FASTPT import HT
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
from scipy.interpolate import interp1d
# from Cosmo_JAB import cosmo

print('FAST-PT version ',FASTPT.__version__)

#define some folders
home=os.getenv('HOME')
dir1='dat_files/'

# Define some functions
def xi_outputs(k,P,k_smooth = 0.,r_filt = 0.):
	if k_smooth != 0:
		smooth=np.exp(-1.*(k/k_smooth)**2)
		Puse = P*smooth
	else:
		Puse = P
	r,xi=HT.k_to_r(k,Puse,1.5,-1.5,.5, (2.*np.pi)**(-1.5))
	if r_filt != 0:
		filt=1.-np.exp(-1.*(r/r_filt)**2)
		xi_filt=xi*filt
	else:
		xi_filt=xi
	k2,P_filt=HT.r_to_k(r,xi_filt,-1.5,1.5,.5, 4.*np.pi*np.sqrt(np.pi/2.))
	r2,wp2=HT.k_to_r(k,Puse,1.,-1.,2., 4.*np.pi*np.sqrt(np.pi/2.))
	r2,wp2_filt=HT.k_to_r(k,P_filt,1.,-1.,2., 4.*np.pi*np.sqrt(np.pi/2.))
	r3,wp0=HT.k_to_r(k,Puse,1.,-1.,0., 4.*np.pi*np.sqrt(np.pi/2.))
	return r, wp2, wp2_filt, xi, xi_filt, wp0


# get matter power spectrum
# right now, just pulling from a file
# could be replaced
Pin=np.loadtxt(dir1+'Pk_Planck15.dat')
k=Pin[:,0]
P=Pin[:,1]

#check log spacing
dk=np.diff(np.log(k))
delta_L=(np.log(k[-1])-np.log(k[0]))/(k.size-1)
dk_test=np.ones_like(dk)*delta_L
log_sample_test='ERROR! (k,Pk) values are not sampled evenly in log space!'
np.testing.assert_array_almost_equal(dk, dk_test, decimal=6, err_msg=log_sample_test, verbose=False)

# perform J0, J2, J4 transforms

r, wp2, wp2_filt, xi, xi_filt, wp0 = xi_outputs(k,P,k_smooth = 0.,r_filt = 0.)

# Combine and add prefactors

prefactor = 1. #confirm
wgplus = prefactor * wp2

# fit an amplitude, or apply an expected amplitude

amplitude = 1.
wgplus *= amplitude

pylab.loglog(r,wgplus)
pylab.show()

# output prediction

# how is the validation done?