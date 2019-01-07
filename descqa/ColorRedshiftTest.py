from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt
import matplotlib.colors as clr

__all__ = ['ColorRedshiftTest']
class _CatalogDoesNotHaveQuantity(Exception):
        """Raised when the catalog doesn't have a quantity and indicates the
        test should be skipped"""
        def __init__(self, quantity_name):
            self.message = "Catalog does not have {}".format(quantity_name)

class ColorRedshiftTest(BaseValidationTest):
    """
    This test plots various color-redshfit diagnostics 
    """


    def __init__(self, **kwargs):
        # load test config options
        self.kwargs = kwargs
        # self.mag_band = kwargs['MagnitudeBand'], 'option1_default')
        # self.option2 = kwargs.get('option2', 'option2_default')
        # self.test_name = kwargs.get('test_name', 'RepeatingMagnitude')
        # load validation data
        with open(os.path.join(self.data_dir, 'README.md')) as f:
            self.validation_data = f.readline().strip()

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()
        color = kwargs['color']
        assert (len(color) == 3) and (color[1] == '-'), "Color must be defined as 'a-b', where a and b are band names"
        allowed_colors = 'ugrizy'
        self.mag1 = color[0].lower()
        self.mag2 = color[2].lower()
        assert (self.mag1 in allowed_colors) and (self.mag2 in allowed_colors), "only ugrizy colors are allowed"
        self.frame = kwargs['frame']
        assert (self.frame in ['rest', 'obs', 'observed', 'observer']), "Only 'rest', 'obs', and 'observed' frames allowed"
        self.filter = kwargs.get('filter', '').lower()
        assert (self.filter in ['lsst', 'sdss', 'des']), "Only lsst, sdss, or DES filters allowed"
        self.baseDC2 = kwargs.get('baseDC2', False)
        self.central = kwargs.get("central", None)
        self.Mr_cut = kwargs.get("Mr_cut", None)
        self.mr_cut = kwargs.get("mr_cut", None)
        self.stellar_mass_cut = kwargs.get("stellar_mass_cut", None)
        self.halo_mass_cut = kwargs.get("halo_mass_cut", None)
        self.red_sequence_cut = kwargs.get("red_sequence_cut", None)
        self.synthetic_type = kwargs.get("synthetic_type", None)
        self.log_scale = kwargs.get("log_scale", True)
        print("hhhm?")
        print(kwargs.get('test_list', {}))

    def post_process_plot(self, ax):
        pass
        # ax.text(0.05, 0.95, self.validation_data)
        # ax.legend(loc='best')
 

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # check if needed quantities exist
        halo_ids = catalog_instance.first_available('halo_id')
        if self.frame == "rest":
            mag_frame = "Mag_true"
            mag_end = "_z0"
        else:
            mag_frame = "mag"
            mag_end = ""
        mag1_str = "{}_{}_{}{}".format(mag_frame, self.mag1, self.filter, mag_end)
        mag2_str = "{}_{}_{}{}".format(mag_frame, self.mag2, self.filter, mag_end)
        mag1_val = self._get_quantity(catalog_instance, mag1_str)
        mag2_val = self._get_quantity(catalog_instance, mag2_str)
        redshift = self._get_quantity(catalog_instance, 'redshift')
        clr_val = mag1_val - mag2_val
        title = ""
        slct, title = self._get_selection_and_title(catalog_instance, title)
        fig, ax = plt.subplots()
        for ax_this in (ax, self.summary_ax):
            h,xbins,ybins = np.histogram2d(redshift[slct], clr_val[slct], bins=256)
            if self.log_scale:
                pc = ax_this.pcolor(xbins,ybins, h.T+3.0, norm = clr.LogNorm())
                fig.colorbar(pc, ax = ax_this).set_label("Population Density + 3")
            else:
                pc = ax_this.pcolor(xbins,ybins, h.T)
                fig.colorbar(pc, ax = ax_this).set_label("Population Density")
            ax_this.set_ylabel('{} - {}'.format(mag1_str, mag2_str))
            ax_this.set_xlabel('redshift')
            ax_this.text(0.05, 0.95, title, transform=ax.transAxes,
                         verticalalignment='top', color='white',
                         fontsize='small')
            ax_this.set_title(catalog_name)
        # self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close(fig)


        return TestResult(0, inspect_only=True)
    


    def _get_quantity(self, catalog_instance, quantity_name):
        if not catalog_instance.has_quantities([quantity_name]):
            raise _CatalogDoesNotHaveQuantity(quantity_name)
        first_name = catalog_instance.first_available(quantity_name)
        return catalog_instance.get_quantities([first_name], native_filters=['redshift_block_lower == 0'],)[first_name]
                                    
                                    
    def _get_selection_and_title(self, catalog_instance, title):
        # a cheap way to get an array of trues of the correct size
        redshift = catalog_instance.first_available('redshift')
        slct = redshift == redshift
        title_elem_per_line =3 # The number of elements in the title. We want about
        title_elem = 0 # three elements per line. The catalog name is pretty big, so it counts
        # as two elements.
        

        if self.central is not None:
            is_central = self._get_quantity(catalog_instance, 'is_central')
            slct = slct & (is_central == self.central)
            title += "central = {}, ".format(self.central)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if self.Mr_cut is not None:
            Mag_r  = self._get_quantity(catalog_instance, "Mag_true_r_lsst_z0")
            slct = slct & ( Mag_r < self.Mr_cut )
            title += "Mr < {}, ".format(self.Mr_cut)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if self.mr_cut is not None:
            mag_r = self._get_quantity(catalog_instance, "mag_r")
            slct = slct & ( mag_r < self.mr_cut )
            title += "mr < {}, ".format(self.mr_cut)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if self.stellar_mass_cut is not None:
            sm = self._get_quantity(catalog_instance, "stellar_mass")
            slct = slct & ( np.log10(sm) > self.stellar_mass_cut )
            title += "M$_{{*}}$ > {}, ".format(self.stellar_mass_cut)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"
        if self.halo_mass_cut is not None:
            halo_mass = self._get_quantity(catalog_instance, "halo_mass")
            slct = slct & ( np.log10(halo_mass) > self.halo_mass_cut )
            title += "M$_{{halo}}$ > {}, ".format(self.halo_mass_cut)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if self.synthetic_type is not None:
            upid = self._get_quantity(catalog_instance, "baseDC2/upid")
            slct = slct & ( upid == self.synthetic_type )
            title += "synth = {}, ".format(self.synthetic_type)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if self.red_sequence_cut is not None:
            rs = self._get_quantity(catalog_instance, "baseDC2/is_on_red_sequence_gr")
            slct = slct & ( rs == self.red_sequence_cut )
            title += "red seq = {}, ".format(self.red_sequence_cut)
            title_elem +=1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        return slct, title

            
    def _generate_color_z(selection, label, filter_type, frame, mag1, mag2, 
                          plot_type=None):
        pass

    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
