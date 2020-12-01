from __future__ import unicode_literals, absolute_import, division
import os
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from sklearn.cluster import k_means
import treecorr
import camb
import camb.correlations
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP7 # pylint: disable=no-name-in-module
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt
pars = camb.CAMBparams()

__all__ = ['ShearTest']


class ShearTest(BaseValidationTest):
    """
    Validation test for shear and convergence quantities
    """

    def __init__(self,
                 z='redshift_true',
                 ra='ra',
                 dec='dec',
                 e1='shear_1',
                 e2='shear_2_phosim',
                 mag='Mag_true_r_sdss_z0',
                 maglim=-19.0,
                 kappa='convergence',
                 nbins=20,
                 min_sep=2.5,
                 max_sep=250,
                 sep_units='arcmin',
                 bin_slop=0.1,
                 zlo=0.5,
                 zhi=0.6,
                 ntomo=2,
                 z_range=0.05,
                 do_jackknife=False,
                 N_clust=10,
                 **kwargs):
        #pylint: disable=W0231

        plt.rcParams['font.size'] = 9

        self.z = z
        #sep-bounds and binning
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbins = nbins
        self.sep_bins = np.linspace(min_sep, max_sep, nbins + 1)
        self.sep_units = sep_units
        self.bin_slop = bin_slop
        self.ra = ra
        self.dec = dec
        self.mag = mag
        self.maglim = maglim
        self.e1 = e1
        self.e2 = e2
        self.kappa = kappa
        self.N_clust = N_clust
        self.do_jackknife = do_jackknife
        # cut in redshift
        self.filters = [(lambda z: (z > zlo) & (z < zhi), self.z)]
        self.summary_fig, self.summary_ax = plt.subplots(nrows=2, ncols=ntomo, sharex=True, squeeze=False, figsize=(ntomo*5, 5))
        self.ntomo = ntomo
        self.z_range = z_range
        self.zlo = zlo
        self.zhi = zhi

    def compute_nz(self, n_z):
        '''create interpolated n(z) distribution'''
        z_bins = np.linspace(self.zlo, self.zhi, 301)
        n = np.histogram(n_z, bins=z_bins)[0]
        z = (z_bins[1:] - z_bins[:-1]) / 2. + z_bins[:-1]
        n2 = interp1d(z, n, bounds_error=False, fill_value=0.0, kind='cubic')
        n2_sum = quad(n2, self.zlo, self.zhi)[0]
        n2 = interp1d(z, n / n2_sum, bounds_error=False, fill_value=0.0, kind='cubic')
        return n2

    @staticmethod
    def integrand_w(x, n, chi, chi_int, cosmo):
        ''' This is the inner bit of GWL lensing kernel - z is related to x, not chi'''
        z = chi_int(x)
        H_z = cosmo.H(z).to(1./u.s).value
        dchidz = const.c.to(u.Mpc/u.s).value/H_z  # pylint: disable=no-member
        return n(z) / dchidz * (x - chi) / x

    def galaxy_W(self, z, n, chi_int, cosmo, chi_max):
        ''' galaxy window function'''
        #pylint: disable=E1101
        chi = cosmo.comoving_distance(z).value  # can be array
        cst = 3. / 2. * cosmo.H(0).to(1. / u.s)**2 / const.c.to(u.Mpc / u.s)**2 * cosmo.Om(
            0)
        prefactor = cst * chi * (1. + z) * u.Mpc
        val_array = []
        for chi_this in chi:
            val_array.append(quad(self.integrand_w, chi_this, chi_max, args=(n, chi_this, chi_int, cosmo))[0])
        W = np.array(val_array) * prefactor * (u.Mpc)  # now unitless
        return W

    @staticmethod
    def integrand_lensing_limber(chi, l, galaxy_W_int, chi_int, p):
        '''return overall integrand for one value of l'''
        z = chi_int(chi)
        k = (l + 0.5) / chi
        integrand = p(z, k, grid=False) * galaxy_W_int(z)**2 / chi**2
        return integrand

    def phi(self, lmax, n_z, cosmo, p, chi_max):
        z_array = np.logspace(-3, np.log10(self.zhi+1.0), 200)
        chi_array = cosmo.comoving_distance(z_array).value
        chi_int = interp1d(chi_array, z_array, bounds_error=False, fill_value=0.0)
        n = self.compute_nz(n_z)
        galaxy_W_int = interp1d(z_array, self.galaxy_W(z_array, n, chi_int, cosmo, chi_max), bounds_error=False, fill_value=0.0)
        phi_array = []
        l = range(0, lmax, 1)
        l = np.array(l)
        for i in l:
            a = quad(
                self.integrand_lensing_limber, 1.e-10, chi_max, args=(i, galaxy_W_int, chi_int, p), epsrel=1.e-6)[0]
            phi_array.append(a)
        phi_array = np.array(phi_array)
        #NOTE: comments here allow for small corrections on large and small scales, these can be used to check impact of pixelization on lensing maps and flat sky approximations in theory.
        prefactor = 1.0  #(l+2)*(l+1)*l*(l-1)  / (l+0.5)**4
        #import healpy as hp
        #pixwin = hp.pixwin(1024)[:lmax]
        return l, phi_array * prefactor#*pixwin**2

    def theory_corr(self, n_z2, xvals, lmax2, cosmo, p, chi_max):
        ll, pp = self.phi(lmax=lmax2, n_z=n_z2, cosmo=cosmo, p=p, chi_max=chi_max)
        pp3_2 = np.zeros((lmax2, 4))
        pp3_2[:, 1] = pp[:] * (ll * (ll + 1.)) / (2. * np.pi)
        cxvals = np.cos(xvals / (60.) / (180. / np.pi))
        vals = camb.correlations.cl2corr(pp3_2, cxvals)
        return xvals, vals[:, 1], vals[:, 2]

    def get_score(self, measured, theory, cov, opt='diagonal'):
        if opt == 'cov':
            cov = np.matrix(cov).I
            print("inverse covariance matrix")
            print(cov)
            chi2 = np.matrix(measured - theory) * cov * np.matrix(measured - theory).T
        elif opt == 'diagonal':
            chi2 = np.sum([(measured[i] - theory[i])**2 / cov[i][i] for i in range(len(measured))])
        else:
            chi2 = np.sum([(measured[i] - theory[i])**2 / theory[i]**2 for i in range(len(measured))])
        diff = chi2 / float(len(measured))
        return diff

    def jackknife(self, catalog_data, xip, xim, mask):
        " computing jack-knife covariance matrix using K-means clustering"
        #k-means clustering to define areas
        #NOTE: This is somewhat deprecated, the jack-knifing takes too much effort to find appropriately accurate covariance matrices.
        # If you want to use this, do a quick convergence check and some timing tests on small N_clust values (~5 to start) first.
        # note also that this is comparing against the (low) variance in the catalog which might not be a great comparison -no shape noise
        N_clust = self.N_clust
        nn = np.stack((catalog_data[self.ra][mask], catalog_data[self.dec][mask]), axis=1)
        _, labs, _ = k_means(
            n_clusters=N_clust, random_state=0, X=nn, n_jobs=-1)  # check random state, n_jobs is in debugging mode
        print("computing jack-knife errors")
        time_jack = time.time()
        # jack-knife code
        xip_jack = []
        xim_jack = []
        gg = treecorr.GGCorrelation(
            nbins=self.nbins,
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            sep_units='arcmin',
            bin_slop=self.bin_slop,
            verbose=True)
        for i in range(N_clust):
            ##### shear computation excluding each jack-knife region
            mask_jack = (labs != i)
            cat_s = treecorr.Catalog(
                ra=catalog_data[self.ra][mask][mask_jack],
                dec=catalog_data[self.dec][mask][mask_jack],
                g1=catalog_data[self.e1][mask][mask_jack] - np.mean(catalog_data[self.e1][mask][mask_jack]),
                g2=-(catalog_data[self.e2][mask][mask_jack] - np.mean(catalog_data[self.e2][mask][mask_jack])),
                ra_units='deg',
                dec_units='deg')
            gg.process(cat_s)

            xip_jack.append(gg.xip)
            xim_jack.append(gg.xim)
            ## debugging outputs
            print("xip_jack")
            print(i)
            print(gg.xip)
            print("time = " + str(time.time() - time_jack))


        ### assign covariance matrix - loop is poor python syntax but compared to the time taken for the rest of the test doesn't really matter
        cp_xip = np.zeros((self.nbins, self.nbins))
        for i in range(self.nbins):
            for j in range(self.nbins):
                for k in range(N_clust):
                    cp_xip[i][j] += N_clust/(N_clust - 1.)  * (xip[i] - xip_jack[k][i] * 1.e6) * (
                        xip[j] - xip_jack[k][j] * 1.e6)

        cp_xim = np.zeros((self.nbins, self.nbins))
        for i in range(self.nbins):
            for j in range(self.nbins):
                for k in range(N_clust):
                    cp_xim[i][j] += N_clust/(N_clust - 1.)  * (xim[i] - xim_jack[k][i] * 1.e6) * (
                        xim[j] - xim_jack[k][j] * 1.e6)
        return cp_xip, cp_xim

    @staticmethod
    def get_catalog_data(gc, quantities, filters=None):
        '''
        Get quantities from catalog
        '''
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True, summary='Missing requested quantities')

        data = gc.get_quantities(quantities, filters=filters)
        #make sure data entries are all finite
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data

    # define theory from within this class

    def post_process_plot(self, ax):
        '''
        Post-processing routines on plot
        '''
        zmeans = np.linspace(self.zlo, self.zhi, self.ntomo+2)[1:-1]

        # vmin and vmax are very rough DES-like limits (maximum and minimum scales)
        for i in range(self.ntomo):
            for ax_this, vmin, vmax in zip(ax[:, i], (2.5, 35), (200, 200)):
                ax_this.set_xscale('log')
                ax_this.axvline(vmin, ls='--', c='k')
                ax_this.axvline(vmax, ls='--', c='k')
            ax[-1][i].set_xlabel(r'$\theta \; {\rm (arcmin)}$')
            ax[0][i].set_title('z = '+str(zmeans[i]))
            ax[0][i].legend()
            ax[-1][i].legend()
        ax[0][0].set_ylabel(r'$\chi_{{{}}} \; (10^{{-6}})$'.format('+'))
        ax[-1][0].set_ylabel(r'$\chi_{{{}}} \; (10^{{-6}})$'.format('-'))



    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        run test on a single catalog
        '''
        # check if needed quantities exist
        if not catalog_instance.has_quantities([self.z, self.ra, self.dec]):
            return TestResult(skipped=True, summary='do not have needed location quantities')
        if not catalog_instance.has_quantities([self.e1, self.e2, self.kappa]):
            return TestResult(skipped=True, summary='do not have needed shear quantities')
        if not catalog_instance.has_quantities([self.mag]):
            return TestResult(skipped=True, summary='do not have required magnitude quantities for cuts')
        catalog_data = self.get_catalog_data(
            catalog_instance, [self.z, self.ra, self.dec, self.e1, self.e2, self.kappa, self.mag], filters=self.filters)

        cosmo = getattr(catalog_instance, 'cosmology', WMAP7)
        ns = getattr(cosmo, 'n_s', 0.963)
        s8 = getattr(cosmo, 'sigma8', 0.8)
        print(cosmo)

        pars.set_cosmology(H0=cosmo.H0.value, ombh2=cosmo.Ob0*cosmo.h**2, omch2=(cosmo.Om0-cosmo.Ob0)*cosmo.h**2)
        pars.InitPower.set_params(ns=ns, As=2.168e-9*(s8/0.8)**2)
        camb.set_halofit_version(version='takahashi') # pylint: disable=no-member
        p = camb.get_matter_power_interpolator(pars, nonlinear=True, k_hunit=False, hubble_units=False, kmax=100., zmax=self.zhi+1., k_per_logint=False).P
        z_max = np.max(catalog_data[self.z])
        if self.zhi>z_max:
            print("updating zhi to "+ str(z_max)+ " from "+ str(self.zhi))
            self.zhi = z_max
            zhi = z_max
        else:
            zhi = self.zhi
        chi_max = cosmo.comoving_distance(self.zhi+1.0).value
        mask_mag = (catalog_data[self.mag][:]<self.maglim)

        # read in shear values and check limits
        e1 = catalog_data[self.e1]
        max_e1 = np.max(e1)
        min_e1 = np.min(e1)
        e2 = catalog_data[self.e2]
        max_e2 = np.max(e2)
        min_e2 = np.min(e2)
        if ((min_e1 < (-1.)) or (max_e1 > 1.0)):
            return TestResult(skipped=True, summary='e1 values out of range [-1,+1]')
        if ((min_e2 < (-1.)) or (max_e2 > 1.0)):
            return TestResult(skipped=True, summary='e2 values out of range [-1,+1]')
        ntomo = self.ntomo
        fig, ax = plt.subplots(nrows=2, ncols=ntomo, sharex=True, squeeze=False, figsize=(ntomo*5, 5))
        zmeans = np.linspace(self.zlo, zhi, ntomo+2)[1:-1]
        for ii in range(ntomo):
            z_mean = zmeans[ii]
            zlo2 = z_mean - self.z_range
            zhi2 = z_mean + self.z_range
            print(zlo2, zhi2)
            zmask = (catalog_data[self.z] < zhi2) & (catalog_data[self.z] > zlo2)
            mask = zmask & mask_mag
            # compute shear auto-correlation
            cat_s = treecorr.Catalog(
                ra=catalog_data[self.ra][mask],
                dec=catalog_data[self.dec][mask],
                g1=catalog_data[self.e1][mask] - np.mean(catalog_data[self.e1][mask]),
                g2=-(catalog_data[self.e2][mask] - np.mean(catalog_data[self.e2][mask])),
                ra_units='deg',
                dec_units='deg')
            gg = treecorr.GGCorrelation(
                nbins=self.nbins,
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                sep_units='arcmin',
                bin_slop=self.bin_slop,
                verbose=True)
            gg.process(cat_s)
            r = np.exp(gg.meanlogr)

            #NOTE: We are computing 10^6 x correlation function for easier comparison
            xip = gg.xip * 1.e6
            xim = gg.xim * 1.e6

            print("npairs  = ")
            print(gg.npairs)
	    #sig = np.sqrt(gg.varxi)  # this is shape noise only - should be very low for simulation data

            do_jackknife = self.do_jackknife
	    # Diagonal covariances for error bars on the plots. Use full covariance matrix for chi2 testing.

            if do_jackknife:
                cp_xip, cp_xim = self.jackknife(catalog_data, xip, xim, mask)
                print(cp_xip)
                sig_jack = np.zeros((self.nbins))
                sigm_jack = np.zeros((self.nbins))
                for i in range(self.nbins):
                    sig_jack[i] = np.sqrt(cp_xip[i][i])
                    sigm_jack[i] = np.sqrt(cp_xim[i][i])
            else:
                sig_jack = np.zeros((self.nbins))
                sigm_jack = np.zeros((self.nbins))
                for i in range(self.nbins):
                    # pylint: disable=no-member # TODO: check whether varxi should be varxip or varxim
                    sig_jack[i] = np.sqrt(gg.varxi[i])*1.e6
                    sigm_jack[i] = np.sqrt(gg.varxi[i])*1.e6

            n_z = catalog_data[self.z][mask]
            xvals, theory_plus, theory_minus = self.theory_corr(n_z, r, 15000, cosmo, p, chi_max)
            theory_plus = theory_plus * 1.e6
            theory_minus = theory_minus * 1.e6

            if do_jackknife:
                chi2_dof_1 = self.get_score(xip, theory_plus, cp_xip, opt='diagonal')  #NOTE: correct this to opt=cov if you want full covariance matrix
            else:
                chi2_dof_1 = self.get_score(xip, theory_plus, 0, opt='nojack')  # correct this

            print(theory_plus)
            print(theory_minus)
            print(xip)
            print(xim)

	    #The following are further treecorr correlation functions that could be added in later to extend the test
	    #treecorr.NNCorrelation(nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin')
	    #treecorr.NGCorrelation(nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin')  # count-shear  (i.e. <gamma_t>(R))
	    #treecorr.NKCorrelation(nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin')  # count-kappa  (i.e. <kappa>(R))
	    #treecorr.KKCorrelation(nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin')  # count-kappa  (i.e. <kappa>(R))

            for ax_this in (ax, self.summary_ax):
                ax_this[0, ii].errorbar(r, xip, sig_jack, lw=0.6, marker='o', ls='', color="#3f9b0b", label=r'$\chi_{+}$')
                ax_this[0, ii].plot(xvals, theory_plus, 'o', color="#9a0eea", label=r'$\chi_{+}$' + " theory")
                ax_this[1, ii].errorbar(r, xim, sigm_jack, lw=0.6, marker='o', ls='', color="#3f9b0b", label=r'$\chi_{-}$')
                ax_this[1, ii].plot(xvals, theory_minus, 'o', color="#9a0eea", label=r'$\chi_{-}$' + " theory")

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close(fig)

        score = chi2_dof_1  #calculate your summary statistics

        #TODO: This criteria for the score is effectively a placeholder if jackknifing isn't used and assumes a diagonal covariance if it is
        # Proper validation criteria need to be assigned to this test
        if score < 2:
            return TestResult(score, inspect_only=True)
        else:
            return TestResult(score, passed=False)

    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
