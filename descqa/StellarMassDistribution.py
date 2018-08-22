from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from base import BaseValidationTest, TestResult
from plotting import plt

import sys
import subprocess
import matplotlib.pylab as plt
%matplotlib inline

import GCRCatalogs

import astropy.units     as     u
from   astropy.cosmology import Planck13, z_at_value
from   astropy.cosmology import Planck15 as cosmo
from   astropy.cosmology import FlatLambdaCDM



__all__ = ['StellarMassTest']

class StellarMassTest(BaseValidationTest):

    """
    This validation test looks at stellar mass distribution
    of DC2 catalogs to make sure it matches the distribution
    of CMASS galaxies which have constraints on both
    magnitude and color og galaxies and also checks the
    number density of galaxies per square degree as the 
    score to pass the test.
    """

    DC2 = FlatLambdaCDM(H0=71, Om0=0.265, Ob0=0.0448)
    
    def __init__(self, catalog_name, catSize, **kwargs):
        
        self.catalog_name  = catalog_name # catalog name  (cosmoDC2_v0.1 for instance)
        self.catSize       = catSize      # catalog size in sq deg (430 for instace)
        self._other_kwargs = kwargs
        

    def get_smass(self, catalog_name, catSize):
        gc         = GCRCatalogs.load_catalog(catalog_name)
        data       = gc.get_quantities(['stellar_mass', 'mag_true_i_lsst', 'mag_true_r_lsst', 'mag_true_g_lsst', 'x','y','z'])
        smass      = data['stellar_mass']
        x, y, z    = data['x'], data['y'], data['z']
        log10smass = np.log10(smass)
    
        # calculating the reshifts from comoving distance
        com_dist  = np.sqrt((x**2) + (y**2)+(z**2))

        min_indx  = np.where(com_dist == np.min(com_dist ))[0][0]
        max_indx  = np.where(com_dist == np.max(com_dist ))[0][0]

        zmin      = z_at_value(self.DC2.comoving_distance, com_dist[min_indx] * u.Mpc)
        zmax      = z_at_value(self.DC2.comoving_distance, com_dist[max_indx] * u.Mpc)

        zgrid     = np.logspace(np.log10(zmin), np.log10(zmax), 50)
        cosmology = self.DC2
        CDgrid    = cosmology.comoving_distance(zgrid)*self.DC2.H0/100.
        #  use interpolation to get redshifts for satellites only
        new_redshifts = np.interp(com_dist, CDgrid, zgrid)
    
        r = data['mag_true_r_lsst']
        i = data['mag_true_i_lsst']
        g = data['mag_true_g_lsst']

        # applying CMASS cuts
        dperp = (r-i) - (g-r)/8.
        cond1 = dperp > 0.55
        cond2 = i < (19.86 + 1.6*(dperp - 0.8))
        cond3 = (i < 19.9) & (i > 17.5)
        cond4 = (r-i) < 2
        cond5 = i < 21.5
    
        # applying the cuts to stellar mass
        smass_cmass_cut = smass[np.where( (cond1==True) & (cond2==True) & (cond3==True) & (cond4==True) & (cond5==True))]

        print 
        print ("minimum cmass-cut = ", np.min(np.log10(smass_cmass_cut)))
        print ("maximum cmass-cut = ", np.max(np.log10(smass_cmass_cut)))
        print

        numDen = len(smass_cmass_cut) / float(catSize)
        return np.log10(smass), np.log10(smass_cmass_cut), new_redshifts, numDen
    
    def run_on_single_catalog(self, catalog_name, catSize, output_dir, label):
        
        log_smass_tot, log_smass_cmass, redshift, numDen = self.get_smass(catalog_name, catSize)
        
        plt.figure(1, figsize=(12,6))
        plt.hist(log_smass_cmass, bins=np.linspace(9,13,50), color="teal", 
                 linewidth=2, histtype = "step", normed = "True", label=label)
        plt.legend(loc='best')
        plt.xlabel(r"$\log(M_{\star})$", fontsize=20)
        plt.ylabel("N", fontsize=20)
        plt.title("[n DC2 = {0} , n CMASS = 101] gals/sq deg".format("%.1f" % numDen))
        plt.savefig(output_dir + "Mstellar_distribution.png")
        plt.show()
        plt.close()
        
        # CMASS stellar mass mean
        log_cmass_mean = 11.25
        
        # score is defined as error away from CMASS stellar mass mean 
        score = (np.mean(log_smass_cmass) - log_cmass_mean)/log_cmass_mean
        
        return TestResult(score, passed=True)
