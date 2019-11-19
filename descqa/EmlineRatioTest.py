from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from descqa import BaseValidationTest, TestResult
from matplotlib import pyplot as plt
import GCRCatalogs
from GCRCatalogs import GCRQuery
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import ks_2samp
from pandas import read_csv
from matplotlib.colors import LogNorm


__all__ = ['EmlineRatioTest']

class EmlineRatioTest(BaseValidationTest):
    """
    An example validation test
    """
    def __init__(self, **kwargs):

        np.random.seed(0)

        # load test config options
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'emline_ratio_test')
        self.emline_ratio1 = kwargs.get('emline_ratio1', 'oii/oiii') # Currently does not support other emission line ratios
        self.emline_ratio2 = kwargs.get('emline_ratio2', 'hb/oiii') # Currently does not support other emission line ratios
        sdss_file = kwargs.get('sdss_file', 'sdss_emission_lines/sdss_query_snr10_ew.csv')
        # self.sdsscat = sdsscat(self.data_dir + '/' + sdss_file)
        self.sdsscat = sdsscat('data/' + sdss_file)

        self.sdss_drawnum = kwargs.get('sdss_drawnum', 30000)
        self.sim_drawnum = kwargs.get('sim_drawnum', 30000)
        self.fig = plt.figure(figsize = (16,8))
        self.sp1 = self.fig.add_subplot(121)
        self.sp2 = self.fig.add_subplot(122)


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
            return TestResult(skipped=True, summary='do not have needed quantities')

        # gc = GCRCatalogs.load_catalog(catname)

        uband_maglim = GCRQuery((np.isfinite, 'mag_u_lsst'), 'mag_u_lsst < 26.3')
        gband_maglim = GCRQuery((np.isfinite, 'mag_g_lsst'), 'mag_g_lsst < 27.5')
        rband_maglim = GCRQuery((np.isfinite, 'mag_r_lsst'), 'mag_r_lsst < 27.7')
        iband_maglim = GCRQuery((np.isfinite, 'mag_i_lsst'), 'mag_i_lsst < 27.0')
        zband_maglim = GCRQuery((np.isfinite, 'mag_z_lsst'), 'mag_z_lsst < 26.2')
        yband_maglim = GCRQuery((np.isfinite, 'mag_y_lsst'), 'mag_y_lsst < 24.9')


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
                                  'mag_y_lsst'], filters = (uband_maglim | gband_maglim | rband_maglim | iband_maglim | zband_maglim | yband_maglim)) 
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

        indices = np.random.choice(np.arange(len(Halpha)), size = self.sim_drawnum, replace = False)

        sz_small = sz[indices]
        galaxyID_small = galaxyID[indices]

        lumdist_small = cosmo.luminosity_distance(sz_small)

        property_list = [Halpha, Hbeta, NII6584, OIII5007, OIII4959, OII3726, OII3729,
                        SII6716, SII6731, SIItot, OIIItot, OIItot]

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


        #=========================================
        # End Reading in Data
        #=========================================


        pvalue, KSstat, medianshift = self.makeplot()


        if np.log10(pvalue) >= -4. and np.linalg.norm(medianshift) <= 0.25:
            return TestResult(pvalue, passed = True)
        elif np.linalg.norm(medianshift) <= 0.25:
            return TestResult(pvalue, passed = False, summary='P-value must exceed 1e-4.')
        elif np.log10(pvalue) >= -4.:
            return TestResult(pvalue, passed = False, summary = 'Total median shift must be less than or equal to 0.25 dex.')
        else:
            return TestResult(pvalue, passed = False, summary = 'P-value must exceed 1e-4 and total median shift must be less than or equal to 0.25 dex.')


    def makeplot(self):

        #=========================================
        # Begin Test and Plotting
        #=========================================


        dist1 = [[],[]]
        dist2 = [[],[]]

        for subplot, cat, dist in [[self.sp1, self, dist1],[self.sp2, self.sdsscat, dist2]]:

            emline1 = getattr(cat, self.emline_ratio1.split('/')[0])
            emline2 = getattr(cat, self.emline_ratio1.split('/')[1])

            er1 = np.log10(emline1/emline2)

            emline1 = getattr(cat, self.emline_ratio2.split('/')[0])
            emline2 = getattr(cat, self.emline_ratio2.split('/')[1])

            er2 = np.log10(emline1/emline2)

            good_inds = np.where(np.isfinite(er1) & np.isfinite(er2))

            dist[0] = er1[good_inds]
            dist[1] = er2[good_inds]

        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        self.sp1.hist2d(*dist1, bins = 50, range = [[-1.2, 1.2],[-1.25, 1]], norm = LogNorm(), cmap = 'plasma_r')
        self.sp2.hist2d(*dist2, bins = 50, range = [[-1.2, 1.2],[-1.25, 1]], norm = LogNorm(), cmap = 'plasma_r')

        sdss_draw_inds = np.random.choice(np.arange(len(dist2[0])), size = self.sdss_drawnum)
        dist2 = dist2[:,sdss_draw_inds]

        medianshift = np.nanmedian(dist1, axis = 1).reshape(2,1) - np.nanmedian(dist2, axis = 1).reshape(2,1)

        medianmatch_dist2 = dist2 + medianshift


        pvalue, KSstat = kstest_2d(dist1, medianmatch_dist2)

        self.sp1.set_xlabel('log(' + self.emline_ratio1 + ')')
        self.sp1.set_ylabel('log(' + self.emline_ratio2 + ')')
        self.sp2.set_xlabel('log(' + self.emline_ratio1 + ')')
        self.sp1.set_xlim(-1.2, 1.2)
        self.sp1.set_ylim(-1.25, 1)
        self.sp2.set_xlim(-1.2, 1.2)
        self.sp2.set_ylim(-1.25, 1)

        self.sp2.set_yticklabels([])

        plt.subplots_adjust(wspace = 0.0)

        self.sp1.text(0.02, 0.98, 'log p = %.2f\nD = %.2f\nMed Shift = [%.2f, %.2f]' % (np.log10(pvalue), KSstat, *medianshift.T[0]), fontsize = 14, transform = self.sp1.transAxes, ha = 'left', va = 'top')

        return pvalue, KSstat, medianshift



    def conclude_test(self, output_dir):
        self.fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.fig)





def fhCounts(x,edge):
    # computes local CDF at a given point considering all possible axis orderings
    
    templist = [np.sum((x[0,0:] >= edge[0]) & (x[1,0:] >= edge[1])), 
                np.sum((x[0,0:] <= edge[0]) & (x[1,0:] >= edge[1])), 
                np.sum((x[0,0:] <= edge[0]) & (x[1,0:] <= edge[1])), 
                np.sum((x[0,0:] >= edge[0]) & (x[1,0:] <= edge[1]))]
    return templist

def kstest_2d(dist1, dist2, alpha = 0.05):
    
    num1 = dist1.shape[1]
    num2 = dist2.shape[1]
    
    KSstat = -np.inf
    
    for iX in (np.arange(0,num1+num2)):
        
        if iX < num1:
            edge = dist1[0:,iX]
        else:
            edge = dist2[0:,iX-num1]
        
#         vfCDF1 = np.sum(fhCounts(dist1, edge)) / num1
#         vfCDF2 = np.sum(fhCounts(dist2, edge)) / num2

        vfCDF1 = np.array(fhCounts(dist1, edge)) / num1
        vfCDF2 = np.array(fhCounts(dist2, edge)) / num2
        
        vfThisKSTS = np.abs(vfCDF1 - vfCDF2)
        fKSTS = np.amax(vfThisKSTS)
        
        if (fKSTS > KSstat):
            KSstat = fKSTS
            #print(KSstat, vfCDF1, vfCDF2)

    # Peacock Z calculation and P estimation

    n =  num1 * num2 /(num1 + num2)
    Zn = np.sqrt(n) * KSstat;
    Zinf = Zn / (1 - 0.53 * n**(-0.9));
    pValue = 2 *np.exp(-2 * (Zinf - 0.5)**2);

    # Clip invalid values for P
    if pValue > 1.0:
        pValue = 1.0
        
#     H = (pValue <= alpha)
    
    return pValue, KSstat







class sdssdist:

    def __init__(self, component_file = 'data/oii-oiii_hb-oiii_components.dat'):

        self.weights, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.cov = np.loadtxt('components.dat', unpack = True)
            


    def draw_components(self, size):
        component_choices, counts = np.unique(np.random.choice(np.arange(len(self.weights)), p = self.weights, size = size), return_counts = True)
        data_x = []
        data_y = []
        for thischoice, thiscount in zip(component_choices, counts):
            cov = [[self.sigma_x[thischoice], self.cov[thischoice]],[self.cov[thischoice], self.sigma_y[thischoice]]]
            tempx, tempy = np.random.multivariate_normal([self.mu_x[thischoice], self.mu_y[thischoice]], cov, thiscount).T
            data_x = data_x + list(tempx)
            data_y = data_y + list(tempy)

        return np.array([data_x, data_y])





class sdsscat:

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
        'ha_ew_err','oiii4959_ew_err_uncorr', 'oiii5007_ew_err_uncorr', 'oii3726_ew_err_uncorr', 'oii3729_ew_err_uncorr', 'hb_ew_err_uncorr']

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