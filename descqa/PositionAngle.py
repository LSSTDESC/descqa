import os
import numpy as np
import scipy.stats
from itertools import count
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['PositionAngle']

class PositionAngle(BaseValidationTest):
    """
    validation test to check the slope of the size distribution at small sizes.
    """
    def __init__(self, **kwargs):
        #pylint: disable=W0231
        #validation data: a uniform distribution on the half-circle
        self.uniform_degrees = scipy.stats.uniform(0, 180.).cdf
        self.uniform_radians = scipy.stats.uniform(0, np.pi).cdf
        
        self.acceptable_keys = kwargs['possible_position_angle_fields']
        self.cutoff = kwargs['p_cutoff']

        self._color_iterator = ('C{}'.format(i) for i in count())

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        # check catalog data for required quantities
        key = catalog_instance.first_available(*self.acceptable_keys)
        if not key:
            summary = 'Missing required quantity' + ' or '.join(['{}']*len(self.acceptable_keys))
            return TestResult(skipped=True, summary=summary.format(*self.acceptable_keys))

        # get data
        catalog_data = catalog_instance.get_quantities(key)
        pos_angles = catalog_data[key]
        is_degrees = np.max(pos_angles)> 2*np.pi
        good_data_mask = np.logical_not(np.logical_or(np.isinf(pos_angles), np.isnan(pos_angles)))
        
        if is_degrees:
            ks_results = scipy.stats.kstest(pos_angles, self.uniform_radians)
        else:
            ks_results = scipy.stats.kstest(pos_angles, self.uniform_radians)

        N, _, _ = plt.hist(pos_angles[good_data_mask], bins=20, edgecolor='black')
        if is_degrees:
            plt.xlabel("Angle [deg]")
        else:
            plt.xlabel("Angle [rad]")
        plt.ylabel("N")
        plt.ylim(0,np.max(N)*1.15)
        plt.text(0.95, 0.96,'Uniform distribution: $p={:.3f}$'%ks_results[1], 
                 horizontalalignment='right', verticalalignment='top',
                 transform=plt.gca().transAxes)

        fig.savefig(os.path.join(output_dir, 'position_angle_{}.png'.format(catalog_name)))
        plt.close(fig)
        return TestResult(passed = (ks_results[0]>self.cutoff))


