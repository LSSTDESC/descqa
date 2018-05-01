from __future__ import unicode_literals, absolute_import, division
import os
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['CheckColors']

class CheckColors(BaseValidationTest):
    """
    Inspection test to represent 2D color plots
    """
    def __init__(self, xcolor='ri', ycolor='gr', **kwargs): # pylint: disable=W0231
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'CheckColors')
        self.mag_fields_to_check = ('mag_{}_lsst',
                                    'mag_{}_sdss',
                                    'mag_{}_des',
                                    'mag_{}_stripe82',
                                    'mag_true_{}_lsst',
                                    'mag_true_{}_sdss',
                                    'mag_true_{}_des',
                                    'Mag_true_{}_des_z01',
                                    'Mag_true_{}_sdss_z01',
                                    )

        if len(xcolor) != 2 or len(ycolor) != 2:
            print('Warning: color string is longer than 2 characters. Only first and second bands will be used.')

        self.xcolor = xcolor
        self.ycolor = ycolor
        self.bands = set(xcolor + ycolor)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        has_results = False
        for mag_field in self.mag_fields_to_check:
            mag_labels = [mag_field.format(band) for band in self.bands]
            if not catalog_instance.has_quantities(mag_labels):
                continue

            data = catalog_instance.get_quantities(mag_labels)
            xcolor = data[mag_field.format(self.xcolor[0])] - data[mag_field.format(self.xcolor[1])]
            ycolor = data[mag_field.format(self.ycolor[0])] - data[mag_field.format(self.ycolor[1])]
            has_results = True

            fig, ax = plt.subplots()
            ax.hexbin(xcolor, ycolor, gridsize=(100), cmap='GnBu', mincnt=1, bins='log')
            ax.set_xlabel('{} - {}'.format(mag_field.format(self.xcolor[0]), mag_field.format(self.xcolor[1])))
            ax.set_ylabel('{} - {}'.format(mag_field.format(self.ycolor[0]), mag_field.format(self.ycolor[1])))
            ax.set_title('Color inspection for {}'.format(catalog_name))
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, '{}_{}_{}.png'.format(self.xcolor, self.ycolor, mag_field.replace('_{}_', '_'))))
            plt.close(fig)

        if not has_results:
            return TestResult(skipped=True)

        return TestResult(inspect_only=True)
