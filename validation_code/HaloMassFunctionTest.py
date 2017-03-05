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
            plot_validation_as_line=True,
    )

    required_quantities = ('mass', 'parent_halo_id', 'positionX', 'positionY', 'positionZ')

    available_observations = ('Sheth-Tormen', 'Jenkins', 'Tinker')

    default_kwargs = {
            'observation': 'Tinker',
            'zlo': 0,
            'zhi': 1000.0,
            'ztest': 0,
            'summary_statistic': 'chisq',
            'jackknife_nside': 5,
    }

    enable_interp_validation = True


    def _init_follow_up(self, kwargs):
        self._import_kwargs(kwargs, 'ztest', func=float, required=True)


    def load_validation_data(self):
        pass


    def _prepare_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        generate halo mass function data
        """
        #associate files with observations
        halo_mass_par = {'Sheth-Tormen':'ST', 'Jenkins':'JEN', 'Tinker':'TINK'}

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


    def get_quantities_from_catalog(self, galaxy_catalog):
        """
        obtain the masses and mask fom the galaxy catalog

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        #get stellar masses from galaxy catalog
        hm = galaxy_catalog.get_quantities("mass", self._zfilter)
        x = galaxy_catalog.get_quantities("positionX", self._zfilter)
        y = galaxy_catalog.get_quantities("positionY", self._zfilter)
        z = galaxy_catalog.get_quantities("positionZ", self._zfilter)
        pid = galaxy_catalog.get_quantities("parent_halo_id", self._zfilter)

        #remove non-finite or negative numbers
        mask = np.isfinite(hm)
        mask &= (hm > 0)
        mask &= np.isfinite(x)
        mask &= np.isfinite(y)
        mask &= np.isfinite(z)
        mask &= (pid == -1)

        return dict(mass=hm[mask], x=x[mask], y=y[mask], z=z[mask])


plot_summary = HaloMassFunctionTest.plot_summary