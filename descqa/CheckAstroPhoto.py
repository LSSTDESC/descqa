from __future__ import print_function, division
import os
import numpy as np
from scipy.stats import binned_statistic
from CatalogMatcher.match import spatial_closest  # https://github.com/LSSTDESC/CatalogMatcher
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt
from matplotlib.ticker import NullFormatter

__all__ = ['CheckAstroPhoto']

nullfmt = NullFormatter()

class CheckAstroPhoto(BaseValidationTest):
    """
    Validation test to compare astrometric and photometric results between
    two different datasets.
    """
    
    def __init__(self, **kwargs):
        #pylint: disable=W0231
        self.kwargs = kwargs
        self.min_mag = kwargs['min_mag']  # Minimum magnitude to bin the sample
        self.max_mag = kwargs['max_mag']  # Maximum magnitude to bin the sample
        self.nbins = kwargs['nbins_mag'] # Number of bins
        self.ra=dict() # Here we are going to store the RA for all catalogs
        self.dec=dict() # Here we are going to store the DEC for all catalogs
        self.magnitude=dict() # Here we are going to store the magnitude (in different bands for all catalogs)
        self.selection_cuts = kwargs['selection_cuts'] # Selection cuts to perform on the data sample
        self.bands = kwargs['bands'] # Photometric band(s) to analyze
      
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        #TODO: Use aliases for magnitudes and make this more general
        mags = ['mag_true_%s_lsst' % band for band in self.bands]
        qs = ['ra', 'dec']
        qs.append(mags)
        filters = [GCRQuery(self.selection_cuts)]
        data = catalog_instance.get_quantities(['ra','dec','mag_true_r_lsst'], filters=filters)
        print('Selected %d objects for catalog %s' % (len(data), catalog_name))
        self.ra[catalog_name] = data['ra']
        self.dec[catalog_name] = data['dec']
        
        for band in self.bands:
            self.magnitude[(catalog_name, band)] = data['mag_true_%s_lsst' % band]
        return TestResult(inspect_only=True)

    def scatter_project(self, x, y, xmin, xmax, ymin, ymax, nbins, xlabel, ylabel, savename, bin_stat=False):

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        plt.figure()

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        axScatter.scatter(x, y, s=1., alpha=0.5)
        
        if bin_stat:
            mean_y, be, _ = binned_statistic(x, y, range=(xmin, xmax), bins=nbins)
            std_y, be, _ = binned_statistic(x, y, range=(xmin, xmax), bins=nbins, statistic='std')
            n_y, be, _ = binned_statistic(x, y, range=(xmin, xmax), bins=nbins, statistic='count')
            axScatter.errorbar(0.5*(be[1:]+be[:-1]), mean_y, std_y/np.sqrt(n_y), marker='o', linestyle='none')

        axScatter.set_xlim((xmin, xmax))
        axScatter.set_ylim((ymin, ymax))
        axScatter.set_xlabel(xlabel)
        axScatter.set_ylabel(ylabel)

        axHistx.hist(x, bins=nbins, range=(xmin, xmax))
        axHisty.hist(y, bins=nbins, range=(ymin, ymax), orientation='horizontal')

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        #plt.tight_layout()
        plt.savefig(savename)

    def conclude_test(self, output_dir):
        """
        This function should gather the two catalogs, match them and perform the summary plots
        """

        try:
            assert(len(self.ra.keys())==2) # Making sure that we have *just* two catalogs
        except AssertionError:
            print('The test can compare two catalogs only!')

        cat_names = list(self.ra.keys()) # This is an auxiliary list to easily get the catalogs
        cat_len = [len(self.ra[cat_names[0]]),len(self.ra[cat_names[1]])]
        cat_names = np.array(cat_names)[np.argsort(cat_len)].tolist() # We find the catalog with less objects to build the tree
        # For this test we are going to match using closest neighbor since it is the fastest but it can be easily
        # swapped for any other matching strategy
        
        dist, matched_id, is_matched = spatial_closest(self.ra[cat_names[0]],self.dec[cat_names[0]], 
                                           self.ra[cat_names[1]],self.dec[cat_names[1]],np.arange(cat_len[1]))
        
        delta_ra = self.ra[cat_names[0]]-self.ra[cat_names[1]][matched_id]
        delta_dec = self.dec[cat_names[0]]-self.dec[cat_names[1]][matched_id]
        delta_mag = dict()

        for band in self.bands:
            delta_mag[band] = self.magnitude[(cat_names[0], band)]-self.magnitude[(cat_names[1],band)][matched_id]
        
        # Scatter plot + histogram of RA and Dec (assumed to be in degrees)
        astro_savename = os.path.join(output_dir, 'astrometry_check_%s_%s.png' % (cat_names[0], cat_names[1]))
        self.scatter_project(delta_ra*3600, delta_dec*3600, -1, 1, -1, 1, 100, r'$\Delta$ RA [arcsec]',
                            r'$\Delta$ Dec [arcsec]', astro_savename)
        
        # Scatter plot + histogram of Delta mag vs mag
        for band in self.bands:
            photo_savename = os.path.join(output_dir, 'photometry_check_%s_%s_%s.png' % (cat_names[0], cat_names[1], band))
            self.scatter_project(self.magnitude[(cat_names[0], band)], delta_mag[band], self.min_mag, self.max_mag,
                                -1, 1, self.nbins, '%s' % band, r'$\Delta %s$' % band, photo_savename, bin_stat=True)
 
