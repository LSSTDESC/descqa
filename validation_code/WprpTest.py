from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt

from ValidationTest import ValidationTest

from L2Diff import L2Diff
from helpers.CorrelationFunction import projected_correlation


class WprpTest(ValidationTest):
    """
    validaton test class object to compute project 2-point correlation function wp(rp)
    """
    
    def __init__(self, test_args, data_directory, data_args):
        """
        Initialize a stellar mass function validation test.
        
        Parameters
        ----------
        test_args : dictionary
            dictionary of arguments specifying the parameters of the test
            
        data_directory : string
            path to comparison data directory
        
        data_args : dictionary
            dictionary of arguments specifying the comparison data
        """
        
        super(ValidationTest, self).__init__()
        
        #set validation data information
        self._sdss_wprp = os.path.join(data_directory, data_args['sdss'])
        self._mb2_wprp = os.path.join(data_directory, data_args['mb2'])
        self._sm_cut = test_args['sm_cut']
        self._rbins = np.logspace(*test_args['rbins'])
        self._zmax = test_args['zmax']
        self._njack = test_args['njack']
        self._summary_thres = test_args.get('summary_thres', 10.0)

    def run_validation_test(self, galaxy_catalog, galaxy_catalog_name, output_dict):
        """
        Load galaxy catalog and (re)calculate the stellar mass function.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader
        
        galaxy_catalog_name : string
            name of mock galaxy catalog
        
        output_dict : dictionary
            dictionary of output informaton
        
        Returns
        -------
        test_passed : boolean
        """
        
        #make sure galaxy catalog has appropiate quantities
        required_quantities = ('stellar_mass', 'positionX', 'positionY', 'positionZ', 'velocityZ')

        if not all(q in galaxy_catalog.quantities for q in required_quantities):
            #raise an informative warning
            msg = 'galaxy catalog does not have all the required quantities {}, skipping the rest of the validation test.'.format(', '.join(required_quantities))
            warn(msg)
            with open(output_dict['log'], 'a') as f:
                f.write(msg)
            
            return 2

        #continue with the test
        gc = galaxy_catalog

        try:
            h = gc.cosmology.H0.value/100.0
        except AttributeError:
            h = 0.702
            msg = 'Make sure `cosmology` and `redshift` properties are set. Using default value h=0.702...'
            warn(msg)
            with open(output_dict['log'], 'a') as f:
                f.write(msg)

        # convert arguments
        sm_cut = self._sm_cut/(h*h)
        rbins = self._rbins/h
        zmax = self._zmax/h
        njack = self._njack

        # load catalog
        flag = (gc.get_quantities("stellar_mass", {}) >= sm_cut)
        x = gc.get_quantities("positionX", {})
        flag &= np.isfinite(x)

        x = x[flag]
        y = gc.get_quantities("positionY", {})[flag]
        z = gc.get_quantities("positionZ", {})[flag]
        vz = gc.get_quantities("velocityZ", {})[flag]
    
        vz /= (100.0*h)
        z += vz

        # calc wp(rp)
        points = np.remainder(np.vstack((x,y,z)).T, gc.box_size)
        wp, wp_cov = projected_correlation(points, rbins, zmax, gc.box_size, njack)
        rp = np.sqrt(rbins[1:]*rbins[:-1])
        wp_err = np.sqrt(np.diag(wp_cov))

        fig, ax = plt.subplots()
        self.add_line_on_plot(ax, rp, wp, wp_err, galaxy_catalog_name, output_dict['catalog'])
        d1 = {'x':rp, 'y':wp, 'dy':wp_err}

        # load mb2 wp(rp), use this to validate
        rp, wp, wp_err = np.loadtxt(self._mb2_wprp).T
        self.add_line_on_plot(ax, rp, wp, wp_err, 'MB-II', output_dict['validation'])
        d2 = {'x':rp, 'y':wp, 'dy':wp_err}

        # load sdss wp(rp), just for comparison
        rp, wp, wp_err = np.loadtxt(self._sdss_wprp).T
        self.add_line_on_plot(ax, rp, wp, wp_err, 'SDSS')

        ax.set_xlim(0.1, 50.0)
        ax.set_ylim(1.0, 3.0e3)
        ax.set_xlabel(r'$r_p \; {\rm [Mpc]}$')
        ax.set_ylabel(r'$w_p(r_p) \; {\rm [Mpc]}$')
        ax.set_title(r'Projected correlation function ($M_* > {0:.2E} \, {{\rm M}}_\odot$)'.format(sm_cut))
        ax.legend(loc='best', frameon=False)
        plt.savefig(output_dict['figure'])

        L2, success = L2Diff(d1, d2, self._summary_thres)
        with open(output_dict['summary'], 'a') as f:
            f.write('L2 = {}\n'.format(L2))
            if success:
                f.write('Test passed! L2 < {}\n'.format(self._summary_thres))
            else:
                f.write('Test failed, you need L2 < {}!\n'.format(self._summary_thres))

        return (0 if success else 1)


    def add_line_on_plot(self, ax, rp, wp, wp_err, label, save_output=None):
        if save_output is not None:
            np.savetxt(save_output, np.vstack((rp, wp, wp_err)).T, header='rp wp wp_err')
        l = ax.loglog(rp, wp, label=label, lw=1.5)[0]
        ax.fill_between(rp, wp+wp_err, np.where(wp > wp_err, wp - wp_err, 0.01), alpha=0.2, color=l.get_color(), lw=0)

