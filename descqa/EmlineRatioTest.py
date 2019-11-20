# pylint: disable=E1101,E0611,W0231,W0201
# E1101 throws errors on my setattr() stuff and astropy.units.W and astropy.units.Hz
# E0611 throws an error when importing astropy.cosmology.Planck15
# W0231 gives a warning because __init__() is not called for BaseValidationTest
# W0201 gives a warning when defining attributes outside of __init__()
from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from GCR import GCRQuery
from pandas import read_csv
from descqa import BaseValidationTest, TestResult

emline_names = {'ha': r'H$\alpha$', 'hb': r'H$\beta$', 'oii': '[OII]', 'oiii': '[OIII]'}

__all__ = ['EmlineRatioTest']

class EmlineRatioTest(BaseValidationTest):
    """
    Validation test for the relaive luminosity of emission lines in a catalog

    Parameters
    ----------
    emline_ratio1: str, optional, (default: 'oii/oiii')
        The emission line luminosity ratio to be plotted on the x-axis
    emline_ratio2: str, optional, (default: 'hb/oiii')
        The emission line luminosity ratio to be plotted on the y-axis
    sdss_file: str, optional, (default: 'sdss_emission_lines/sdss_query_snr10_ew.csv')
        Location of the SDSS data file that will be passed into the sdsscat class.  Looks
        in the 'data/' folder.
    mag_u_cut: float, optional, (default: 26.3)
        u-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_g_cut: float, optional, (default: 27.5)
        g-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_r_cut: float, optional, (default: 27.7)
        r-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_i_cut: float, optional, (default: 27.0)
        i-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_z_cut: float, optional, (default: 26.2)
        z-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_y_cut: float, optional, (default: 24.9)
        y-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    sdss_drawnum: int, optional, (default: 30000)
        The number of galaxies to draw from the SDSS data file to perform the comparison.
        The default number is chosen to (hopefully) not make the 2-D KS test too stringent.
    sim_drawnum: int, optional, (default: 30000)
        The number of galaxies to draw from the simulated data to perform the comparison.
        The default number is chosen to (hopefully) not make the 2-D KS test too stringent.
    """
    def __init__(self, **kwargs):

        np.random.seed(0)

        # load test config options
        self.kwargs = kwargs
        self.emline_ratio1 = kwargs.get('emline_ratio1', 'oii/oiii') # Currently does not support other emission line ratios
        self.emline_ratio2 = kwargs.get('emline_ratio2', 'hb/oiii') # Currently does not support other emission line ratios
        sdss_file = kwargs.get('sdss_file', 'sdss_emission_lines/sdss_query_snr10_ew.csv')
        # self.sdsscat = sdsscat(self.data_dir + '/' + sdss_file)
        self.sdsscat = sdsscat('descqa/data/' + sdss_file)

        # The magnitude cuts for galaxies pulled from the catalog.  These numbers correspond to
        # a 5-sigma cut based on https://arxiv.org/pdf/0912.0201.pdf

        self.mag_u_cut = kwargs.get('mag_u_cut', 26.3)
        self.mag_g_cut = kwargs.get('mag_g_cut', 27.5)
        self.mag_r_cut = kwargs.get('mag_r_cut', 27.7)
        self.mag_i_cut = kwargs.get('mag_i_cut', 27.0)
        self.mag_z_cut = kwargs.get('mag_z_cut', 26.2)
        self.mag_y_cut = kwargs.get('mag_y_cut', 24.9)

        # These numbers dictate how large the two samples will be.  I have found that
        # if the numbers get much larger than this, the 2-D KS test becomes more discriminatory
        # than desired, but they can be changed if necessary

        self.sdss_drawnum = kwargs.get('sdss_drawnum', 30000)
        self.sim_drawnum = kwargs.get('sim_drawnum', 30000)
        
        self.figlist = []
        self.runcat_name = []


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        #=========================================
        # Begin Reading in Data
        #=========================================

        # check if needed quantities exist
        if not catalog_instance.has_quantities(['mag_u_lsst',
                                                'mag_g_lsst',
                                                'mag_r_lsst',
                                                'mag_i_lsst',
                                                'mag_z_lsst',
                                                'mag_y_lsst',
                                                'galaxyID',
                                                'emissionLines/totalLineLuminosity:oxygenII3726',
                                                'emissionLines/totalLineLuminosity:oxygenII3729',
                                                'redshift',
                                                'emissionLines/totalLineLuminosity:balmerAlpha6563',
                                                'emissionLines/totalLineLuminosity:balmerBeta4861',
                                                'emissionLines/totalLineLuminosity:nitrogenII6584',
                                                'emissionLines/totalLineLuminosity:oxygenIII4959',
                                                'emissionLines/totalLineLuminosity:oxygenIII5007',
                                                'emissionLines/totalLineLuminosity:sulfurII6716',
                                                'emissionLines/totalLineLuminosity:sulfurII6731']):
            return TestResult(skipped=True, summary='Necessary quantities are not present')

        uband_maglim = GCRQuery((np.isfinite, 'mag_u_lsst'), 'mag_u_lsst < %.1f' % self.mag_u_cut)
        gband_maglim = GCRQuery((np.isfinite, 'mag_g_lsst'), 'mag_g_lsst < %.1f' % self.mag_g_cut)
        rband_maglim = GCRQuery((np.isfinite, 'mag_r_lsst'), 'mag_r_lsst < %.1f' % self.mag_r_cut)
        iband_maglim = GCRQuery((np.isfinite, 'mag_i_lsst'), 'mag_i_lsst < %.1f' % self.mag_i_cut)
        zband_maglim = GCRQuery((np.isfinite, 'mag_z_lsst'), 'mag_z_lsst < %.1f' % self.mag_z_cut)
        yband_maglim = GCRQuery((np.isfinite, 'mag_y_lsst'), 'mag_y_lsst < %.1f' % self.mag_y_cut)


        data = catalog_instance.get_quantities(['galaxyID',
                                                'emissionLines/totalLineLuminosity:oxygenII3726',
                                                'emissionLines/totalLineLuminosity:oxygenII3729',
                                                'redshift',
                                                'emissionLines/totalLineLuminosity:balmerAlpha6563',
                                                'emissionLines/totalLineLuminosity:balmerBeta4861',
                                                'emissionLines/totalLineLuminosity:nitrogenII6584',
                                                'emissionLines/totalLineLuminosity:oxygenIII4959',
                                                'emissionLines/totalLineLuminosity:oxygenIII5007',
                                                'emissionLines/totalLineLuminosity:sulfurII6716',
                                                'emissionLines/totalLineLuminosity:sulfurII6731',
                                                'mag_u_lsst',
                                                'mag_g_lsst',
                                                'mag_r_lsst',
                                                'mag_i_lsst',
                                                'mag_z_lsst',
                                                'mag_y_lsst'], filters=(uband_maglim | gband_maglim | rband_maglim | iband_maglim | zband_maglim | yband_maglim)) 
        sz = data['redshift']
        galaxyID = data['galaxyID']
        Halpha = data['emissionLines/totalLineLuminosity:balmerAlpha6563']* 4.4659e13*u.W/u.Hz
        Hbeta = data['emissionLines/totalLineLuminosity:balmerBeta4861']* 4.4659e13*u.W/u.Hz
        NII6584 = data['emissionLines/totalLineLuminosity:nitrogenII6584']* 4.4659e13*u.W/u.Hz
        OIII5007 = data['emissionLines/totalLineLuminosity:oxygenIII5007']* 4.4659e13*u.W/u.Hz
        OIII4959 = data['emissionLines/totalLineLuminosity:oxygenIII4959']* 4.4659e13*u.W/u.Hz
        OII3726 = data['emissionLines/totalLineLuminosity:oxygenII3726']* 4.4659e13*u.W/u.Hz
        OII3729 = data['emissionLines/totalLineLuminosity:oxygenII3729']* 4.4659e13*u.W/u.Hz
        SII6716 = data['emissionLines/totalLineLuminosity:sulfurII6716']* 4.4659e13*u.W/u.Hz
        SII6731 = data['emissionLines/totalLineLuminosity:sulfurII6731']* 4.4659e13*u.W/u.Hz
        SIItot = SII6716 + SII6731
        OIIItot = OIII5007 + OIII4959
        OIItot = OII3726 + OII3729

        # Reduce the sample size by drawing self.sim_drawnum galaxies 

        indices = np.random.choice(np.arange(len(Halpha)), size=self.sim_drawnum, replace=False)

        sz_small = sz[indices]
        galaxyID_small = galaxyID[indices]

        lumdist_small = cosmo.luminosity_distance(sz_small)

        property_list = [Halpha, Hbeta, NII6584, OIII5007, OIII4959, OII3726, OII3729,
                        SII6716, SII6731, SIItot, OIIItot, OIItot]

        # This loop needs to be formatted in this way (rather than using 'for thisproperty in property_list')
        # so that the changes persist outside of the loop

        for x in range(len(property_list)):

            property_list[x] = (property_list[x][indices]/(4*np.pi*lumdist_small**2)).to('erg/s/cm**2/Hz').value

        Halpha_small, Hbeta_small, NII6584_small, OIII5007_small, OIII4959_small, OII3726_small, OII3729_small, SII6716_small, SII6731_small, SIItot_small, OIIItot_small, OIItot_small = property_list

        self.id = galaxyID_small
        self.ha = Halpha_small
        self.hb = Hbeta_small
        self.oii = OIItot_small
        self.oiii = OIIItot_small
        self.nii6584 = NII6584_small
        self.oiii5007 = OIII5007_small
        self.oiii4959 = OIII4959_small
        self.oii3726 = OII3726_small
        self.oii3729 = OII3729_small
        self.sii6716 = SII6716_small
        self.sii6731 = SII6731_small
        self.siitot = SIItot_small


        #=========================================
        # End Reading in Data
        #=========================================

        #=========================================
        # Perform the Test and Return Results
        #=========================================


        thisfig, pvalue, medianshift = self.makeplot(catalog_name)
        self.figlist.append(thisfig)
        self.runcat_name.append(catalog_name)


        if np.log10(pvalue) >= -4. and np.linalg.norm(medianshift) <= 0.25:
            return TestResult(pvalue, passed=True)
        elif np.linalg.norm(medianshift) <= 0.25:
            return TestResult(pvalue, passed=False, summary='P-value must exceed 1e-4.')
        elif np.log10(pvalue) >= -4.:
            return TestResult(pvalue, passed=False, summary='Total median shift must be less than or equal to 0.25 dex.')
        else:
            return TestResult(pvalue, passed=False, summary='P-value must exceed 1e-4 and total median shift must be less than or equal to 0.25 dex.')


    def makeplot(self, catalog_name):
        """
        Make a summary plot of the test results
        """
        #=========================================
        # Begin Test and Plotting
        #=========================================

        fig = plt.figure(figsize = (16, 8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        dist1 = [[], []]
        dist2 = [[], []]

        xlabel = ''
        ylabel = ''

        # Generate each distribution
        # dist1 is SDSS data
        # dist2 is simulation data

        for cat, dist in [[self.sdsscat, dist1], [self, dist2]]:

            emline1 = getattr(cat, self.emline_ratio1.split('/')[0])
            emline2 = getattr(cat, self.emline_ratio1.split('/')[1])

            er1 = np.log10(emline1/emline2)

            emline1 = getattr(cat, self.emline_ratio2.split('/')[0])
            emline2 = getattr(cat, self.emline_ratio2.split('/')[1])

            er2 = np.log10(emline1/emline2)

            good_inds = np.where(np.isfinite(er1) & np.isfinite(er2))

            dist[0] = er1[good_inds]
            dist[1] = er2[good_inds]

        xlabel = r'$\log_{10}$(' + emline_names[self.emline_ratio1.split('/')[0]] + '/' + emline_names[self.emline_ratio1.split('/')[1]] + ')'
        ylabel = r'$\log_{10}$(' + emline_names[self.emline_ratio2.split('/')[0]] + '/' + emline_names[self.emline_ratio2.split('/')[1]] + ')'

        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        sp1.hist2d(*dist1, bins=50, range=[[-1.2, 1.2], [-1.25, 1]], norm=LogNorm(), cmap='plasma_r')
        sp2.hist2d(*dist2, bins=50, range=[[-1.2, 1.2], [-1.25, 1]], norm=LogNorm(), cmap='plasma_r')

        # Draw a number of SDSS galaxies equal to self.sdss_drawnum

        sdss_draw_inds = np.random.choice(np.arange(len(dist1[0])), size=self.sdss_drawnum)
        dist1 = dist1[:, sdss_draw_inds]

        # Shift the median of the simulated galaxies to match that of the SDSS galaxies 
        # before performing the comparison

        medianshift = np.nanmedian(dist1, axis=1).reshape(2, 1) - np.nanmedian(dist2, axis=1).reshape(2, 1)

        medianmatch_dist2 = dist2 + medianshift

        pvalue, KSstat = kstest_2d(dist1, medianmatch_dist2)

        # Plotting stuff

        sp1.set_xlabel(xlabel, fontsize=20)
        sp1.set_ylabel(ylabel, fontsize=20)
        sp2.set_xlabel(xlabel, fontsize=20)
        sp1.set_xlim(-1.2, 1.2)
        sp1.set_ylim(-1.25, 1)
        sp2.set_xlim(-1.2, 1.2)
        sp2.set_ylim(-1.25, 1)

        sp2.set_yticklabels([])

        plt.subplots_adjust(wspace=0.0)

        sp2.text(0.02, 0.98, 'log p = %.2f\nD$_\mathrm{KS}$ = %.2f\nMed Shift = [%.2f, %.2f]' % (np.log10(pvalue), KSstat, *medianshift.T[0]), fontsize=14, transform=sp2.transAxes, ha='left', va='top', bbox = dict(boxstyle='round', facecolor='white', alpha=0.8))

        sp1.text(0.98, 0.02, 'SDSS', fontsize=24, ha='right', va='bottom', transform=sp1.transAxes)
        sp2.text(0.98, 0.02, catalog_name, fontsize=24, ha='right', va='bottom', transform=sp2.transAxes)

        return fig, pvalue, medianshift


    def summary_file(self, output_dir):
        """
        Saves a summary file with information about the cuts performed on the data in order to
        perform the test
        """

        with open(os.path.join(output_dir, 'Emline_Lum_Ratio_Summary.txt'), 'w') as writefile:
            writefile.write('Simulation Galaxies Drawn: %i\n' % self.sim_drawnum)
            writefile.write('SDSS Galaxies Drawn: %i\n' % self.sdss_drawnum)
            for thisband in ['u', 'g', 'r', 'i', 'z', 'y']:
                writefile.write(thisband + '-band magnitude cut: %.1f\n' % getattr(self, 'mag_' + thisband + '_cut'))
            writefile.write('\n')
            writefile.write('=================\n')
            writefile.write(' Catalogs Tested \n')
            writefile.write('=================\n')


            for thiscat in self.runcat_name:

                writefile.write(thiscat + '\n')






    def conclude_test(self, output_dir):

        # Save a summary file with the details of the test

        self.summary_file(output_dir)

        # Save all of the summary plots into output_dir

        for thisfig, thiscat in zip(self.figlist, self.runcat_name):
            thisfig.savefig(os.path.join(output_dir, thiscat + '_emline_ratios.png'), bbox_inches='tight')
            plt.close(thisfig)





def fhCounts(x,edge):
    # computes local CDF at a given point considering all possible axis orderings
    
    templist = [np.sum((x[0, 0:] >= edge[0]) & (x[1, 0:] >= edge[1])),
                np.sum((x[0, 0:] <= edge[0]) & (x[1, 0:] >= edge[1])),
                np.sum((x[0, 0:] <= edge[0]) & (x[1, 0:] <= edge[1])),
                np.sum((x[0, 0:] >= edge[0]) & (x[1, 0:] <= edge[1]))]
    return templist

def kstest_2d(dist1, dist2):
    """
    Perform the 2-D KS-test on dist1 and dist2.
    """
    num1 = dist1.shape[1]
    num2 = dist2.shape[1]

    KSstat = -np.inf

    for iX in (np.arange(0, num1+num2)):

        if iX < num1:
            edge = dist1[0:, iX]
        else:
            edge = dist2[0:, iX-num1]

        vfCDF1 = np.array(fhCounts(dist1, edge)) / num1
        vfCDF2 = np.array(fhCounts(dist2, edge)) / num2

        vfThisKSTS = np.abs(vfCDF1 - vfCDF2)
        fKSTS = np.amax(vfThisKSTS)

        if (fKSTS > KSstat):
            KSstat = fKSTS

    # Peacock Z calculation and P estimation

    n = num1 * num2 /(num1 + num2)
    Zn = np.sqrt(n) * KSstat
    Zinf = Zn / (1 - 0.53 * n**(-0.9))
    pValue = 2 *np.exp(-2 * (Zinf - 0.5)**2)

    # Clip invalid values for P
    if pValue > 1.0:
        pValue = 1.0

#     H = (pValue <= alpha)

    return pValue, KSstat




class sdsscat:
    """
    This class holds the SDSS data in an easily accessible form, and also dust corrects
    the emission lines using the Balmer Decrement.
    """


    def __init__(self, infile):

        self.Calzetti2000 = np.vectorize(self.Calzetti2000_novec)

        data = read_csv(infile)

        usecols = ['z', 'z_err', 'oii_flux', 'oii_flux_err', 'oiii_flux', 'oiii_flux_err',
                   'h_alpha_flux', 'h_alpha_flux_err', 'h_beta_flux', 'h_beta_flux_err',
                   'lgm_tot_p50', 'lgm_tot_p16', 'lgm_tot_p84', 'sfr_tot_p50', 'sfr_tot_p16', 'sfr_tot_p84',
                   'oh_p50', 'h_alpha_eqw', 'oiii_4959_eqw', 'oiii_5007_eqw', 'oii_3726_eqw', 'oii_3729_eqw', 'h_beta_eqw',
                   'h_alpha_eqw_err', 'oiii_4959_eqw_err', 'oiii_5007_eqw_err', 'oii_3726_eqw_err', 'oii_3729_eqw_err', 'h_beta_eqw_err']
        newnames = ['z', 'z_err', 'oii_uncorr', 'oii_err_uncorr', 'oiii_uncorr', 'oiii_err_uncorr', 'ha_uncorr', 'ha_err_uncorr', 'hb_uncorr', 'hb_err_uncorr',
                    'logmstar', 'logmstar_lo', 'logmstar_hi', 'sfr', 'sfr_lo', 'sfr_hi', 'o_abundance', 'ha_ew_uncorr', 'oiii4959_ew_uncorr', 'oiii5007_ew_uncorr', 'oii3726_ew_uncorr', 'oii3729_ew_uncorr', 'hb_ew_uncorr',
                    'ha_ew_err', 'oiii4959_ew_err_uncorr', 'oiii5007_ew_err_uncorr', 'oii3726_ew_err_uncorr', 'oii3729_ew_err_uncorr', 'hb_ew_err_uncorr']

        for col, name in zip(usecols, newnames):
            setattr(self, name, data[col].values)

        for x, colname in enumerate(newnames):
            if 'flux' in usecols[x]:
                setattr(self, colname, getattr(self, colname)/10**17) # Units are 10**-17 erg/s/cm^2

        # Dust correction
        # E(B-V) = log_{10}(ha_uncorr/(hb_uncorr*2.86)) *(-0.44/0.4) / (k(lam_ha) - k(lam_hb))

        self.EBV = np.log10(self.ha_uncorr/(self.hb_uncorr*2.86)) * (-.44/0.4) / (self.Calzetti2000(6563.) - self.Calzetti2000(4863.))

        # A_oiii = self.Calzetti2000(4980.) * self.EBV / 0.44
        # A_oii = self.Calzetti2000(3727.) * self.EBV / 0.44
        # A_ha = self.Calzetti2000(6563.) * self.EBV / 0.44
        # A_hb = self.Calzetti2000(4863.) * self.EBV / 0.44

        for x, colname in enumerate(newnames):

            if 'ha_' in colname:
                wave = 6563.
            elif 'hb_' in colname:
                wave = 4863.
            elif 'oii_' in colname:
                wave = 3727.
            elif 'oiii_' in colname:
                wave = 4980.
            elif 'oii3726_' in colname:
                wave = 3726.
            elif 'oii3729_' in colname:
                wave = 3729.
            elif 'oiii4959_' in colname:
                wave = 4969.
            elif 'oiii5007_' in colname:
                wave = 5007.

            if 'uncorr' in colname and 'ew' not in colname:

                A_line = self.Calzetti2000(wave) * self.EBV / 0.44

                newflux = getattr(self, colname) * np.power(10, 0.4*A_line)
                setattr(self, colname[:-7], newflux)

            elif 'uncorr' in colname and 'ew' in colname:

                multiplier = np.power(10, 0.4 * self.Calzetti2000(wave) * self.EBV * ((1./.44) - 1.))
                setattr(self, colname[:-7], getattr(self, colname)*multiplier)

        self.ha_lum = self.ha * 4 * np.pi * (cosmo.luminosity_distance(self.z).to('cm').value)**2

        goodind = np.where(np.log10(self.ha_lum) < 45)[0]

        for x, colname in enumerate(list(self.__dict__.keys())):
            if colname != 'Calzetti2000':
                setattr(self, colname, getattr(self, colname)[goodind])

        self.oiii_ew = self.oiii4959_ew + self.oiii5007_ew
        self.oii_ew = self.oii3726_ew + self.oii3729_ew
        self.oiii_ew_err = np.sqrt(self.oiii4959_ew_err**2. + self.oiii5007_ew_err**2.)
        self.oii_ew_err = np.sqrt(self.oii3726_ew_err**2. + self.oii3729_ew_err**2.)





    def Calzetti2000_novec(self, lam):

        # Plug in lam in angstroms
        # From Calzetti2000
        # Returns k(lam)

        lam = lam * 0.0001 # Convert angstroms to microns

        # Rprime_v = 4.88 # pm 0.98 from Calzetti 1997b
        Rprime_v = 4.05

        if 0.1200 < lam and 0.6300 > lam:

            return 2.659 * (-2.156 + (1.509/lam) - (0.198/(lam**2.)) + (0.011/(lam**3.))) + Rprime_v

        elif 0.6300 < lam and 2.2000 > lam:

            return 2.659 * (-1.857 + (1.04/lam)) + Rprime_v

        else:

            return np.NaN