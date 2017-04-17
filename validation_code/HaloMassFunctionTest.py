from __future__ import (division, print_function, absolute_import)
import os
import subprocess
from warnings import warn
import numpy as np

from ValidationTest import ValidationTest, TestResult, plt
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
            xaxis_label=r'$\log \, M_{\rm halo} / M_\odot$',
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
        input_par_fn = os.path.join(base_output_dir, 'input.par')

        h = galaxy_catalog.cosmology.H(self.ztest).value/100.
        om = galaxy_catalog.cosmology.Om0
        
        #input.par file contains (in this order!):
        input_par = (
                '{:.6g}'.format(om), #omega_0           -- total matter fraction
                '0.046',             #omega_bar         -- baryon fraction
                '{:.6g}'.format(h),  #h                 -- hubble constant in 100 km/s/Mpc
                '0.816',             #sigma_8           -- variance of the linear density field
                '0.96',              #n_s               -- power spectrum index
                '-1',                #w_0               -- dark energy equation of state parameter
                '0',                 #w_a               -- ... (see the above, w = w_0 + w_a*(1-a)) 
                '1.686',             #delta_c           -- linear overdensity at virialization
                'EH',                #transfer function -- options: CMB, BBKS, EBW, PD, HS, KH or EH
                halo_mass_par[self.observation], #fitting function -- options: PS, ST, JEN, LANL, DELP, REED, REED06 or TINK
                '0.0625',            #redshift          -- z >= 0
                '{:.6g}'.format(97.7/om),        #Delta            -- overdensity of SO halos; used only for Tinker MF
                '1.0E7',             #minimal mass      -- range of masses which output will cover
                '1.0E15',            #maximal mass      -- ... (see the above)
                '50',                #k_max             -- maximum k for calculating sigma(k) integration
        )
        
        if getattr(self, '_amf_args', None) != input_par:
            with open(input_par_fn, 'w') as f:
                f.write('\n'.join(input_par))
                f.write('\n')
            if os.path.exists(fn):
                os.remove(fn)
            CWD = os.getcwd()
            os.chdir(base_output_dir)
            try:
                subprocess.check_call([exe])
            finally:
                os.chdir(CWD)

            MassFunc = np.loadtxt(fn).T
            self.validation_data = {'x':np.log10(MassFunc[2]/h), 'y':MassFunc[3]*h*h*h}
            self._amf_args = tuple(input_par)


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
