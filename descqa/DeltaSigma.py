import os
import numpy as np
import treecorr
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.constants as cst
from astropy.cosmology import WMAP7 # pylint: disable=no-name-in-module
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['DeltaSigma']

class DeltaSigma(BaseValidationTest):
    """
    This validation test looks at galaxy-shear correlations by comparing DeltaSigma.
    """

    def __init__(self, **kwargs):
        # pylint: disable=super-init-not-called

        # validation data
        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])

        self.data = kwargs['data']
        self.zmin_l = kwargs['zmin_l']
        self.zmax_l = kwargs['zmax_l']
        self.zmin_s = kwargs['zmin_s']
        self.zmax_s = kwargs['zmax_s']
        self.max_background_galaxies = int(float(kwargs['max_background_galaxies']))
        self.zmax = kwargs['zmax']
        self.Rmin = kwargs['Rmin']
        self.Rmax = kwargs['Rmax']
        self.nR = kwargs['nR']

        self.validation_data = np.loadtxt(validation_filepath)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # pylint: disable=no-member

        # Try to read cosmology from catalog, otherwise defualts to WMAP7
        try:
            cosmo = catalog_instance.cosmology
        except AttributeError:
            cosmo = WMAP7

        # Create interpolation tables for efficient computation of sigma crit
        z = np.linspace(0, self.zmax, self.zmax*100)
        d1 = cosmo.angular_diameter_distance(z) # in Mpc
        angular_diameter_distance = interp1d(z, d1, kind='quadratic')
        d2 = cosmo.comoving_transverse_distance(z) # in Mpc
        comoving_transverse_distance = interp1d(z, d2, kind='quadratic')

	# Now figure out the lenses, for the validation data available, 
        # each have slightly non-trivial cuts, so we do them separately... not totally ideal

        if self.data == 'sdss_lowz':

            # Singh et al (2015) (http://adsabs.harvard.edu/abs/2015MNRAS.450.2195S) measurements on the SDSS LOWZ sample.
		
            res = catalog_instance.get_quantities(['redshift_true', 'ra', 'dec', 'shear_1', 'shear_2',
                    'mag_true_i_sdss', 'mag_true_z_sdss','mag_true_g_sdss', 'mag_true_r_sdss'])

            # Compute mask for lowz sample
            # These cuts are defined in section 3 of https://arxiv.org/pdf/1509.06529.pdf
            # and summarised here: http://www.sdss.org/dr14/algorithms/boss_galaxy_ts/#TheBOSSLOWZGalaxySample
            # Definition of auxiliary colors:
            cperp = (res['mag_true_r_sdss'] - res['mag_true_i_sdss']) - (res['mag_true_g_sdss'] - res['mag_true_r_sdss'])/4.0 - 0.18
            cpar = 0.7*(res['mag_true_g_sdss'] - res['mag_true_r_sdss']) + 1.2*((res['mag_true_r_sdss'] - res['mag_true_i_sdss'])-0.18)
            # LOWZ selection cuts:
            mask_lens = np.abs(cperp) < 0.2 # color boundaries
            mask_lens &= res['mag_true_r_sdss'] < (13.5 + cpar/0.3) # sliding magnitude cut
            mask_lens &= (res['mag_true_r_sdss'] > 16) &(res['mag_true_r_sdss'] < 19.6)

            #  Additional redshift cuts used in Singh et al. (2015)
            mask_lens &= (res['redshift_true'] > self.zmin_l) & (res['redshift_true'] < self.zmax_l)
            Mask_lens = [mask_lens]
            
            fig = plt.figure()

        if self.data == 'cfhtlens':

            res = catalog_instance.get_quantities(['redshift_true', 'ra', 'dec', 'shear_1', 'shear_2',
                    'Mag_true_g_lsst_z0', 'Mag_true_r_lsst_z0'])
                
            Mr_min = np.arange(8)*(-0.5)-21.0
            Mr_max = np.arange(8)*(-0.5)-20.0
            blue_frac = np.array([0.7,0.45,0.32,0.2,0.11,0.05,0.03,0.09])*100

            gr = res['Mag_true_g_lsst_z0'] - res['Mag_true_r_lsst_z0'] # larger number means redder

            Mask_lens = []
            for i in range(8):
                mask_lens = (res['redshift_true']>self.zmin_l) & (res['redshift_true']<self.zmax_l) & (res['Mag_true_r_lsst_z0']>Mr_min[i]) & (res['Mag_true_r_lsst_z0']<Mr_max[i])
                gr_threshold = np.percentile(gr[mask_lens], blue_frac[i])
                Mask_lens.append(mask_lens & (gr>gr_threshold))
                Mask_lens.append(mask_lens & (gr<gr_threshold))
            
            fig = plt.figure(figsize=(12,5))            

    
        # Computing mask for source sample, this only serves to keep the number of galaxies managable
        mask_source = (res['redshift_true'] > self.zmin_s) & (res['redshift_true'] < self.zmax_s)
        inds = np.where(mask_source)[0]
        if len(inds) > int(self.max_background_galaxies):
               mask_source[inds[np.random.choice(len(inds),
               size=len(inds) - int(self.max_background_galaxies),
               replace=False)]] = False

        ra_s = res['ra'][mask_source]
        dec_s = res['dec'][mask_source]
        coords = SkyCoord(ra=res['ra']*u.degree, dec=res['dec']*u.degree)
        coords_s = coords[mask_source]
        g1 = res['shear_1'][mask_source]
        g2 = res['shear_2'][mask_source]


        # run gammat in thin redshift bins, loop over lens bins of different stellar mass and colors
        for i in range(len(Mask_lens)):
	        
            nlens = len(np.where(Mask_lens[i])[0]) / catalog_instance.sky_area
            with open(os.path.join(output_dir, 'galaxy_density_'+str(self.data)+'.dat'), 'a') as f:
                        f.write('{} \n'.format(nlens))

            # Create astropy coordinate objects
            coords_l = coords[Mask_lens[i]]

            # Search for neighbours
            idx1, idx2, sep2d, _ = search_around_sky(coords_l, coords_s, 2.*u.deg)

            # Computing sigma crit for each pair
            zl = res['redshift_true'][Mask_lens[i]][idx1]
            zs = res['redshift_true'][mask_source][idx2]

            # Warning: this assumes a flat universe
            # See http://docs.astropy.org/en/v0.3/_modules/astropy/cosmology/core.html#FLRW.angular_diameter_distance_z1z2
            dm1 = comoving_transverse_distance(zl)
            dm2 = comoving_transverse_distance(zs)
            angular_diameter_distance_z1z2 = u.Quantity((dm2 - dm1)/(1. + zs), u.Mpc)

            sigcrit = cst.c**2 / (4.*np.pi*cst.G) * angular_diameter_distance(zs) / \
                ((1. + zl)**2. * angular_diameter_distance_z1z2 * angular_diameter_distance(zl))

            # NOTE: the validation data is in comoving coordinates, the next few
            # lines take care of proper unit conversions
            # Apply unit conversion to obtain sigma crit in h Msol /pc^2 (comoving)
            cms = u.Msun / u.pc**2
            sigcrit = sigcrit*(u.kg/(u.Mpc* u.m)).to(cms) / cosmo.h
            # Computing the projected separation for each pairs, in Mpc/h (comoving)
            r = sep2d.rad*angular_diameter_distance(zl)*(1. + zl) * cosmo.h

            # Computing the tangential shear
            thetac = np.arctan2((coords_s[idx2].dec.rad - coords_l[idx1].dec.rad) / np.cos((coords_s[idx2].dec.rad + coords_l[idx1].dec.rad) / 2.0),coords_s[idx2].ra.rad - coords_l[idx1].ra.rad)
            gammat = -(res['shear_1'][mask_source][idx2] * np.cos(2*thetac) - res['shear_2'][mask_source][idx2] * np.sin(2*thetac))

            # Binning the tangential shear
            bins = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), self.nR, endpoint=True)
            counts = np.histogram(r, bins=bins)[0]
            gt, b = np.histogram(r, bins=bins, weights=gammat*sigcrit)
            rp = 0.5*(b[1:]+b[:-1])
            gt = gt/counts

            outfile = os.path.join(output_dir, 'DS_'+str(self.data)+'_'+str(i)+'.dat')
            np.savetxt(outfile, np.vstack((rp, gt)).T)

            
            if self.data == 'sdss_lowz':
                ax = plt.subplot(111)
                plt.errorbar(self.validation_data[:,0], self.validation_data[:,1], yerr=self.validation_data[:,2], label='SDSS LOWZ from Singh et al. (2015)')
                plt.loglog(rp, gt, label=catalog_name)

           
            if self.data == 'cfhtlens':
                ii = np.mod(i,2)
                ax = plt.subplot(1,2,ii+1)
                plt.loglog(rp, gt, label='['+str(Mr_min[int(i/2)])+', '+str(Mr_max[int(i/2)])+']')

                if ii==0:
                    plt.title('red')
                else:
                    plt.title('blue')

        #plt.title('Number density {}/deg$^2$ vs 57/deg$^2$ for LOWZ'.format(nlens))
        ax.plt.subplot(121)
        ax.set_xlabel('$r_p$ [Mpc/h]')
        ax.set_ylabel(r'$\Delta \Sigma [h \ M_\odot / pc^2]$')
        ax.set_xlim(self.Rmin*0.7, self.Rmax*1.3)
        ax.set_ylim(0.5, 100)

        ax.plt.subplot(122)
        ax.legend()
        ax.set_xlabel('$r_p$ [Mpc/h]')
        ax.set_ylabel(r'$\Delta \Sigma [h \ M_\odot / pc^2]$')
        ax.set_xlim(self.Rmin*0.7, self.Rmax*1.3)
        ax.set_ylim(0.5, 100)

        fig.savefig(os.path.join(output_dir, 'delta_sigma_'+str(catalog_name)+'.png'))
        plt.close(fig)


        return TestResult(inspect_only=True)


"""
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.loglog(rp, gt, label='LOWZ-like sample from '+catalog_name)
        plt.errorbar(self.validation_data[:,0], self.validation_data[:,1], yerr=self.validation_data[:,2], label='SDSS LOWZ from Singh et al. (2015)')
        plt.title('Number density {}/deg$^2$ vs 57/deg$^2$ for LOWZ'.format(nlens))
        ax.set_xlabel('$r_p$ [Mpc/h]')
        ax.set_ylabel(r'$\Delta \Sigma [h \ M_\odot / pc^2]$')
        ax.legend()
        ax.set_xlim(0.05, 10)
        ax.set_ylim(0.5, 100)
        fig.savefig(os.path.join(output_dir, 'delta_sigma_{}.png'.format(catalog_name)))
        plt.close(fig)
        return TestResult(inspect_only=True)
"""

