import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import spatial
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.constants as cst
from astropy.cosmology import WMAP7
from itertools import count
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['DeltaSigmaTest']

class DeltaSigmaTest(BaseValidationTest):
    """
    This validation test looks at galaxy-shear correlations by comparing Delta
    Sigma to the Singh et al (2015) measurements on the SDSS LOWZ sample.
    """

    def __init__(self, **kwargs):
        #validation data
        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        zmax = kwargs['zmax']

        self.validation_data = np.loadtxt(validation_filepath)
        self._color_iterator = ('C{}'.format(i) for i in count())

        # Create interpolation tables for efficient computation of sigma crit
        z = np.linspace(0,zmax,zmax*100)
        d1 = WMAP7.angular_diameter_distance(z) # in Mpc
        self.angular_diameter_distance = interp1d(z,d1, kind='quadratic')

        d2 = WMAP7.comoving_transverse_distance(z) # in Mpc
        self.comoving_transverse_distance = interp1d(z,d2, kind='quadratic')

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        res = catalog_instance.get_quantities(['redshift_true', 'ra','dec','shear_1', 'shear_2',
                                               'convergence', 'mag_true_i_sdss', 'mag_true_z_sdss',
                                               'mag_true_g_sdss', 'mag_true_r_sdss'])

        # Compute mask for lowz sample
        cperp = (res['mag_true_r_sdss'] - res['mag_true_i_sdss']) - (res['mag_true_g_sdss'] - res['mag_true_r_sdss'])/4.0 - 0.18
        cpar = 0.7*(res['mag_true_g_sdss'] - res['mag_true_r_sdss']) + 1.2*((res['mag_true_r_sdss'] - res['mag_true_i_sdss'])-0.18)

        mask_lowz = np.abs(cperp) < 0.2 # color boundaries
        mask_lowz &= res['mag_true_r_sdss'] < (13.5 + cpar/0.3) # sliding magnitude cut
        mask_lowz &= (res['mag_true_r_sdss'] > 16) &(res['mag_true_r_sdss'] < 19.6)

        #  Additional redshift cuts used in Singh et al. (2015)
        mask_lowz &= (res['redshift_true'] > 0.16) & (res['redshift_true'] < 0.36)

        # Computing mask for source sample, this only serves to keep the number
        # of neighbours manageable
        mask_source = (res['mag_true_i_sdss'] < 26.) & (res['redshift_true'] > 0.4)

        # Create astropy coordinate objects
        coords = SkyCoord(ra=res['ra']*u.degree, dec=res['dec']*u.degree)
        coords_l = coords[mask_lowz]
        coords_s = coords[mask_source]

        # Search for neighbours
        idx1, idx2, sep2d, dist3d= search_around_sky(coords_l, coords_s, 2.*u.deg)

        # Computing sigma crit for each pair
        zl = res['redshift_true'][mask_lowz][idx1]
        zs = res['redshift_true'][mask_source][idx2]

        # Warning: this assumes a flat universe
        # See http://docs.astropy.org/en/v0.3/_modules/astropy/cosmology/core.html#FLRW.angular_diameter_distance_z1z2
        dm1 = self.comoving_transverse_distance(zl)
        dm2 = self.comoving_transverse_distance(zs)
        angular_diameter_distance_z1z2 = u.Quantity((dm2 - dm1)/(1. + zs), u.Mpc)

        sigcrit = cst.c**2/(4.*np.pi*cst.G) * (self.angular_diameter_distance(zs)/((1. + zl)**2. *
                                            angular_diameter_distance_z1z2 *
                                            self.angular_diameter_distance(zl)))
        # Apply unit conversion to obtain sigma crit in h Msol /pc^2
        cms = u.Msun /u.pc**2
        sigcrit = sigcrit*(u.kg/(u.Mpc* u.m)).to(cms)/0.7

        # Computing the projected separation for each pairs, in Mpc/h
        r = sep2d.rad*self.angular_diameter_distance(zl)*(1. + zl) * 0.7

        # Computing the tangential shear
        thetac = np.arctan2((coords_s[idx2].dec.rad - coords_l[idx1].dec.rad)/np.cos((coords_s[idx2].dec.rad + coords_l[idx1].dec.rad) /2.0),
                 coords_s[idx2].ra.rad - coords_l[idx1].ra.rad)
        gammat = -(res['shear_1'][mask_source][idx2] * np.cos(2*thetac) + res['shear_2'][mask_source][idx2] * np.sin(2*thetac))

        # Binning the tangential shear
        bins = np.logspace(np.log10(0.05),1, 17, endpoint=True)
        counts,a = np.histogram(r, bins=bins)
        gt,b = np.histogram(r, bins=bins, weights=gammat*sigcrit)
        rp = 0.5*(b[1:]+b[:-1])

        counts[counts <1] = 1
        gt /= counts

        fig = plt.figure()
        ax = plt.subplot(111)
        plt.loglog(rp, gt, label='LOWZ-like sample from '+catalog_name)
        plt.errorbar(self.validation_data[:,0], self.validation_data[:,1], yerr=self.validation_data[:,2], label='SDSS LOWZ from Singh et al. (2015)' )
        ax.set_xlabel('$r_p$ [Mpc/h]')
        ax.set_ylabel('$\Delta \Sigma [h \ M_\odot / pc^2]$')
        ax.legend()
        ax.set_xlim(0.05,10)
        ax.set_ylim(0.5,100)
        fig.savefig(os.path.join(output_dir, 'delta_sigma_{}.png'.format(catalog_name)))
        plt.close(fig)
        return TestResult(inspect_only=True)
