from __future__ import unicode_literals, absolute_import, division
import os
import re
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['TruthGalaxyVerification']

class TruthGalaxyVerification(BaseValidationTest):
    """
    Verify the galaxy components of the truth catalog.
    Works on a composite catalog that joins the truth and extragalactic catalogs.

    Parameters
    ----------
    to_verify: list of dict
        each dict should have keys `truth` and `extragalactic` that specify the column names
        and also optional keys `atol` and `rtol` that specify tolerance
    check_missing_galaxy_quantities : list of str
        column names in extragalactic catalog to plot the properties of missing galaxies
    """
    def __init__(self, **kwargs):
        to_verify = kwargs.get('to_verify')
        if not to_verify:
            raise ValueError('Nothing to verify!')
        if not all(isinstance(d, dict) for d in to_verify):
            raise ValueError('`to_verify` must be a list of dictionaries')
        if not all('truth' in d and 'extragalactic' in d for d in to_verify):
            raise ValueError('each dict in `to_verify` must have `truth` and `extragalactic`')
        self.to_verify = tuple(to_verify)
        self.check_missing_galaxy_quantities = tuple(kwargs.get('check_missing_galaxy_quantities', []))
        self.bins = int(kwargs.get('bins', 100))
        super(TruthGalaxyVerification, self).__init__(**kwargs)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        passed = []
        failed = []
        masked = None

        for i, to_verify in enumerate(self.to_verify):
            quantities = [
                ('extragalactic', to_verify['extragalactic']),
                ('truth', to_verify['truth']),
            ]
            if not catalog_instance.has_quantities(quantities):
                failed.append(quantities)
                continue

            data = catalog_instance.get_quantities(quantities)
            q1 = data[quantities[0]]
            q2 = data[quantities[1]]
            del data

            if masked is None and np.ma.is_masked(q2):
                masked = q2.mask.copy()

            if to_verify.get('atol') or to_verify.get('rtol'):
                passed_this = np.allclose(q1, q2, **{k: float(to_verify.get(k, 0)) for k in ('atol', 'rtol')})
            else:
                passed_this = (q1 == q2).all()

            if passed_this:
                passed.append(quantities)
            else:
                failed.append(quantities)
                diff = (q1 - q2)
                if np.ma.is_masked(diff):
                    diff = diff.compressed()
                self.plot_hist(diff, '{0[0]}:{0[1]} - {1[0]}:{1[1]}'.format(*quantities), 'diff_{:02d}'.format(i), output_dir, log=True)

        if masked is not None and masked.any() and self.check_missing_galaxy_quantities:
            data = catalog_instance.get_quantities([('extragalactic', q) for q in self.check_missing_galaxy_quantities])
            for i, q in enumerate(self.check_missing_galaxy_quantities):
                self.plot_hist(data[('extragalactic', q)][masked], q, 'missing_{:02d}'.format(i), output_dir, log=True)

        if passed:
            with open(os.path.join(output_dir, 'results_passed.txt'), 'w') as f:
                for q in passed:
                    f.write(str(q) + '\n')

        if failed:
            with open(os.path.join(output_dir, 'results_failed.txt'), 'w') as f:
                for q in failed:
                    f.write(str(q) + '\n')

        return TestResult(score=len(failed), passed=(not failed))

    def plot_hist(self, data, xlabel, filename_prefix, output_dir, **kwargs):
        filename = '{}_{}.png'.format(filename_prefix, re.sub('_+', '_', re.sub(r'\W+', '_', xlabel)).strip('_')).strip('_')
        fig, ax = plt.subplots()
        data = data[np.isfinite(data)]
        if data.size:
            ax.hist(data, self.bins, **kwargs)
        ax.set_xlabel(xlabel)
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

