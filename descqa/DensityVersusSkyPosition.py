from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from scipy.stats import binned_statistic
from .base import BaseValidationTest, TestResult
from .plotting import plt
import healpy as hp

__all__ = ['DensityVersusSkyPosition']

def create_hp_map(ra, dec, nside):
    """
    Auxiliary function to generate HEALPix maps from catalogs.
    It reads the ra and dec in degrees and returns a HEALPix map
    """
    pixnums = hp.ang2pix(nside, ra, dec, lonlat=True)
    return np.bincount(pixnums, minlength=hp.nside2npix(nside)).astype(float)

class DensityVersusSkyPosition(BaseValidationTest):
    """
    This test checks the object density as a function
    of another map (e.g: extinction, airmass, etc)
    """
    def __init__(self,**kwargs): # pylint: disable=W0231
        
        self.kwargs = kwargs
        self.test_name = kwargs['test_name']
        self.validation_path = kwargs['validation_map_filename']
        self.nside = kwargs['nside'] 
        self.validation_data = hp.ud_grade(hp.read_map(self.validation_path), nside_out=self.nside)
        self.xlabel = kwargs['xlabel']
     
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        
        if not catalog_instance.has_quantities(['ra', 'dec', 'extendedness']):
            return TestResult(skipped=True, summary='catalog does not have needed quantities')

        catalog_data = catalog_instance.get_quantities(['ra', 'dec', 'extendedness'], filters=['extendedness == 1'])
        data_map = create_hp_map(catalog_data['ra'], catalog_data['dec'], self.nside) 
        mask = data_map>0 # This is a good approximation if the pixels are big enough
        xmin, xmax = np.percentile(self.validation_data[mask], [5,95])
        data_map /= (3600*hp.nside2pixarea(self.nside, degrees=True)) # To get the density in arcmin^-2
        mean_dens, be, _ = binned_statistic(self.validation_data[mask], data_map[mask], statistic='mean', range=(xmin, xmax))
        std_dens, be, _  = binned_statistic(self.validation_data[mask], data_map[mask], statistic='std', range=(xmin, xmax))
        counts, be, _ = binned_statistic(self.validation_data[mask], data_map[mask], statistic='count', range=(xmin, xmax))
        bin_centers = 0.5*be[1:]+0.5*be[:-1]
        fig, ax = plt.subplots(1,1)
        ax.errorbar(bin_centers, mean_dens, std_dens/np.sqrt(counts), fmt='o')
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Mean density [arcmin$^{-2}$]')     
        fig.savefig(os.path.join(output_dir, '%s_density_vs_extinction.png' % catalog_name))

        return TestResult(inspect_only=True)
