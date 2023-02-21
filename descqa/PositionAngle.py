import os
from itertools import count
import numpy as np
import scipy.stats
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['PositionAngle']

class PositionAngle(BaseValidationTest):
    """
    validation test to check that the distribution of galaxy position sizes is random.
    """
    def __init__(self, **kwargs):
        #pylint: disable=W0231
        #validation data: a uniform distribution on the half-circle
        self.uniform_degrees = scipy.stats.uniform(0, 180.).cdf
        self.uniform_radians = scipy.stats.uniform(0, np.pi).cdf
       
        self.acceptable_keys = kwargs['possible_position_angle_fields']
        self.cutoff = kwargs['p_cutoff']
        self.max_size = kwargs.get('max_size', 5e6)

        self._color_iterator = ('C{}'.format(i) for i in count())

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        # check catalog data for required quantities
        key = catalog_instance.first_available(*self.acceptable_keys)
        if not key:
            summary = 'Missing required quantity ' + ' or '.join(['{}']*len(self.acceptable_keys))
            return TestResult(skipped=True, summary=summary.format(*self.acceptable_keys))

        # remove ultra-faint synthetics if present in catalog
        if catalog_instance.has_quantity('baseDC2/halo_id'):
            filters = [(lambda z: (z > -20), 'baseDC2/halo_id')]
        elif catalog_instance.has_quantity('base5000/halo_id'):
            filters = [(lambda z: (z > -20), 'base5000/halo_id')]
        else:
            filters = None

        # get data
        catalog_data = catalog_instance.get_quantities(key, filters=filters)
        pos_angles = catalog_data[key]
        is_degrees = np.max(pos_angles) > 2*np.pi
        good_data_mask = np.logical_not(np.logical_or(np.isinf(pos_angles), np.isnan(pos_angles)))

        # downsample data to max_size to get reliable p values
        dlen = len(pos_angles)
        sample=np.random.sample(dlen)
        fraction = min(self.max_size/float(dlen), 1.0)
        index=(sample<fraction)
        test_angles = pos_angles[index]
        print('Downsampling catalog data to {} for p-value statistic'.format(self.max_size))
        
        if is_degrees:
            ks_results = scipy.stats.kstest(test_angles, self.uniform_degrees)
        else:
            ks_results = scipy.stats.kstest(test_angles, self.uniform_radians)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        N, _, _ = ax.hist(pos_angles[good_data_mask], bins=20, color=catalog_color, edgecolor='black')
        if is_degrees:
            ax.set_xlabel("Angle [deg]")
        else:
            ax.set_xlabel("Angle [rad]")
        ax.set_ylabel("N")
        ax.set_ylim(0, np.max(N)*1.15)
        ax.text(0.95, 0.96, 'Uniform distribution: $p={:.3f}$'.format(ks_results[1]),
                horizontalalignment='right', verticalalignment='top',
                transform=plt.gca().transAxes)

        fig.savefig(os.path.join(output_dir, 'position_angle_{}.png'.format(catalog_name)))
        plt.close(fig)
        return TestResult(score=ks_results[1], passed=(ks_results[1]>self.cutoff))

