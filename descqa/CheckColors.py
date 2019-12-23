from __future__ import print_function, unicode_literals, absolute_import, division
import os
import numpy as np 
from .base import BaseValidationTest, TestResult
import matplotlib.colors as clr
from .base import BaseValidationTest, TestResult
from .plotting import plt
from astropy.table import Table
from scipy.stats import kde
import seaborn as sns

__all__ = ['CheckColors']

class CheckColors(BaseValidationTest):
    """
    Inspection test to represent 2D color plots
    """
    def __init__(self, **kwargs): # pylint: disable=W0231
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'CheckColors')
        #self.mag_fields_to_check = ('mag_{}',
        #                            'mag_{}_cModel',
        #                            'mag_{}_lsst',
        #                            'mag_{}_sdss',
        #                            'mag_{}_des',
        #                            'mag_{}_stripe82',
        #                            'mag_true_{}_lsst',
        #                            'mag_true_{}_sdss',
        #                            'mag_true_{}_des',
        #                            'Mag_true_{}_des_z01',
        #                            'Mag_true_{}_sdss_z01',
        #                            'Mag_true_{}_lsst_z0',
        #                            'Mag_true_{}_sdss_z0',
        #                           )
        self.mag_fields_to_check = kwargs['mag_fields_to_check']
        self.redshift_cut = kwargs['redshift_cut']  
        self.validation_catalog = kwargs['validation_catalog'] 
        self.redshift_cut_val = kwargs['redshift_cut_val'] 
        self.mag_fields_val = kwargs['mag_fields_val']
        self.path_val = kwargs['path_val']

        if len(kwargs['xcolor']) != 2 or len(kwargs['ycolor']) != 2:
            print('Warning: color string is longer than 2 characters. Only first and second bands will be used.')

        self.xcolor = kwargs['xcolor'] 
        self.ycolor = kwargs['ycolor']
        self.bands = set(kwargs['xcolor'] + kwargs['ycolor'])
        
        self.zlo = kwargs['zlo']
        self.zhi = kwargs['zhi']
        self.zbins = kwargs['zbins']
        
        self.levels = kwargs['levels']
        
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        has_results = False
        redshift_bins = np.linspace(self.zlo, self.zhi, num=self.zbins+1)
        catval = Table.read(self.path_val)
        for mag_field in self.mag_fields_to_check:
            print(mag_field,self.bands)
            quantity_list = [mag_field.format(band) for band in self.bands]
            quantity_list.append(self.redshift_cut)
            print(quantity_list)
            if not catalog_instance.has_quantities(quantity_list):
                continue
            dataall = catalog_instance.get_quantities(quantity_list)
            for i,zlo in enumerate(redshift_bins):
                if i == len(redshift_bins)-1:
                    continue
                zhi = redshift_bins[i+1]
                mask = (dataall[self.redshift_cut] > zlo) & (dataall[self.redshift_cut] < zhi)
                xcolor = dataall[mag_field.format(self.xcolor[0])][mask] - dataall[mag_field.format(self.xcolor[1])][mask]
                ycolor = dataall[mag_field.format(self.ycolor[0])][mask] - dataall[mag_field.format(self.ycolor[1])][mask]
                mask_val = (catval[self.redshift_cut_val] > zlo) & (catval[self.redshift_cut_val] < zhi)
                xcolor_val = catval[self.mag_fields_val.format(self.xcolor[0])][mask_val] - catval[self.mag_fields_val.format(self.xcolor[1])][mask_val]
                ycolor_val = catval[self.mag_fields_val.format(self.ycolor[0])][mask_val] - catval[self.mag_fields_val.format(self.ycolor[1])][mask_val]           
                has_results = True
                #print(xcolor_val, ycolor_val)

                # plot hexbin plot for catalog
                fig, ax = plt.subplots()
                ax.hexbin(xcolor, ycolor, gridsize=(100), cmap='GnBu', mincnt=1, bins='log')
                # plot contour plot for validation
                xmin = -0.5
                xmax = 3.0
                ymin = 0.0
                ymax = 3.0
                hrange = [[xmin,xmax],[ymin,ymax]]
                counts,xbins,ybins = np.histogram2d(xcolor_val,ycolor_val,range=hrange,bins=[30,30])
                cntr1 = ax.contour(counts.transpose(), extent=[xmin,xmax,ymin,ymax],
                                   colors='black',linestyles='solid',levels=self.levels)
                h1,_ = cntr1.legend_elements()

                ax.set_xlabel('{} - {}'.format(mag_field.format(self.xcolor[0]), mag_field.format(self.xcolor[1])))
                ax.set_ylabel('{} - {}'.format(mag_field.format(self.ycolor[0]), mag_field.format(self.ycolor[1])))
                title = "%s = %.2f - %.2f" % (self.redshift_cut, zlo, zhi)
                ax.text(0.05, 0.95, title, transform=ax.transAxes, 
                        verticalalignment='top', color='black', fontsize='small')
                ax.set_title('Color inspection for {} vs {}'.format(catalog_name, self.validation_catalog))

                plt.legend([h1[0]], [self.validation_catalog])
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, '{}_{}_{}_{}.png'.format(self.xcolor, self.ycolor,str(i),mag_field.replace('_{}_', '_'))))
                plt.close(fig)

        if not has_results:
            return TestResult(skipped=True)

        return TestResult(inspect_only=True)
