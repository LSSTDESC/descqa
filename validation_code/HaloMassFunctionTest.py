from __future__ import (division, print_function, absolute_import)
import os
import subprocess
from warnings import warn
import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ValidationTest import ValidationTest, TestResult
import CalcStats
from BinnedStellarMassFunctionTest import *

__all__ = ['HaloMassFunctionTest', 'plot_summary']

ShethTormen = 'Sheth-Tormen'
Jenkins = 'Jenkins'
Tinker = 'Tinker'


class HaloMassFunctionTest(BinnedStellarMassFunctionTest):
    """
    validation test class object to compute halo mass function bins
    """
    output_config = dict(\
            catalog_output_file='catalog_hmf.txt',
            validation_output_file='validation_hmf.txt',
            covariance_matrix_file='covariance_hmf.txt',
            log_file='log_hmf.txt',
            plot_file='plot_hmf.png',
            plot_title='Halo Mass Function',
            xaxis_label=r'$\log \, M_{halo}\ (M_\odot)$',
            yaxis_label=r'$dn/dV\, d\logM\ ({\rm Mpc}^{-3})$',
            summary_colormap='rainbow',
            test_range_color='red')

    required_quantities = ('mass', 'parent_halo_id', 'positionX', 'positionY', 'positionZ')

    available_observations = (ShethTormen, Jenkins, Tinker)

    default_kwargs = {
            'observation': Tinker,
            'zlo': 0,
            'zhi': 1000.0,
            'validation_range': (10.0, 15.0),
            'summary_statistic': 'chisq',
            'jackknife_nside': 5,
            'bins': (7.0, 15.0, 25),
    }

    enable_interp_validation = True

    def _init_special(self, kwargs):
        self._import_kwargs(kwargs, 'ztest')
        if self.ztest is None:
            raise ValueError('test argument `ztest` not set')

        #halo mass bins
        self._import_kwargs(kwargs, 'bins', func=lambda b: np.linspace(*b))


    def _prepare_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        generate halo mass function data
        """

        #associate files with observations
        halo_mass_par = {ShethTormen:'ST', Jenkins:'JEN', Tinker:'TINK'}

        #get path to exe
        exe = os.path.join(self.base_data_dir, 'ANALYTIC/amf/amf.exe')
        fn = os.path.join(base_output_dir, 'analytic.dat')
        inputpars = os.path.join(self.base_data_dir, 'ANALYTIC/amf/input.par')

        #get cosmology from galaxy_catalog
        om = galaxy_catalog.cosmology.Om0
        ob = 0.046 # assume ob is included in om
        h  = galaxy_catalog.cosmology.H(self.ztest).value/100.
        s8 = 0.816# from paper
        ns = 0.96 # from paper
        #Delta = 700.0 # default from original halo_mf.py
        delta_c = 1.686
        fitting_f = halo_mass_par[self.observation]

        args = map(str, [exe, "-omega_0", om, "-omega_bar", ob, "-h", h, "-sigma_8", s8, \
                    "-n_s", ns, "-tf", "EH", "-delta_c", delta_c, "-M_min", 1.0e7, "-M_max", 1.0e15, \
                    "-z", 0.0, "-f", fitting_f])

        if getattr(self, '_amf_args', None) != args:
            subprocess.check_call(["cp", inputpars, base_output_dir])
            if os.path.exists(fn):
                os.remove(fn)
            CWD = os.getcwd()
            os.chdir(base_output_dir)
            try:
                with open(os.devnull, 'w') as FNULL:
                    p = subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
            finally:
                os.chdir(CWD)

            MassFunc = np.loadtxt(fn).T
            self.validation_data = {'x':np.log10(MassFunc[2]/h), 'y':MassFunc[3]*h*h*h}
            self._amf_args = args


    def get_mass_and_mask(self, galaxy_catalog):
        """
        get the halo mass and mask from the catalog

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get halo masses from galaxy catalog
        masses = galaxy_catalog.get_quantities("mass", {'zlo': self.zlo, 'zhi': self.zhi})
        parent_halo_id = galaxy_catalog.get_quantities("parent_halo_id", {'zlo': self.zlo, 'zhi': self.zhi})
        x = galaxy_catalog.get_quantities("positionX", {'zlo': self.zlo, 'zhi': self.zhi})

        #remove non-finite or negative numbers and select central halos only
        mask  = np.isfinite(masses)
        mask &= (masses > 0.0)
        mask &= np.isfinite(x)
        mask &= (parent_halo_id == -1)
        masses = masses[mask]
        return masses, mask


    def plot_result(self, result, catalog_name, savepath):
        """
        plot the stellar mass function of the catalog and validation data

        Parameters
        ----------
        result : dictionary
            stellar mass function of galaxy catalog

        catalog_name : string
            name of galaxy catalog

        savepath : string
            file to save plot
        """
        config = self.output_config
        with OnePointFunctionPlot(savepath, title=config['plot_title'], xlabel=config['xaxis_label'], ylabel=config['yaxis_label']) as plot:
            plot.add_line(result, label=catalog_name, color='b')
            plot.add_line(self.validation_data, label=self.observation, color='g')
            plot.add_vband(*self.validation_range, color=self.output_config['test_range_color'], label='Test Region')


def plot_summary(output_file, catalog_list, validation_kwargs):
    """
    make summary plot for validation test

    Parameters
    ----------
    output_file: string
        filename for summary plot

    catalog_list: list of tuple
        list of (catalog, catalog_output_dir) used for each catalog comparison

    validation_kwargs : dict
        keyword arguments used in the validation
    """

    config = HaloMassFunctionTest.output_config
    colors= matplotlib.cm.get_cmap('nipy_spectral')(np.linspace(0, 1, len(catalog_list)+1)[:-1])

    with OnePointFunctionPlot(output_file, title=config['plot_title'], xlabel=config['xaxis_label'], ylabel=config['yaxis_label']) as plot:
        for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
            d = load_file(os.path.join(catalog_dir, config['catalog_output_file']))
            plot.add_line(d, catalog_name, color=color)

        d = load_file(os.path.join(catalog_dir, config['validation_output_file']))
        plot.add_line(d, validation_kwargs['observation'], color='k')

