from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt
from scipy import fftpack
import astropy.table
import sqlite3
from sqlite3 import Error

__all__ = ['ImagePkTest','QuickBkgTest']

def compute_bkg(image):
    """
    Routine to give an estimate of the mean, median and std 
    of the background level from  a given image

    Args:
    -----
    image : np.array

    Returns:
    --------
    mean_bkg : Mean background level
    median_bkg : Median background level
    bkg_noise: Background noise level
    """
    image = image.flatten()
    q95 = np.percentile(image,95) # This is kind of arbitrary but it works fine
    q5 = np.percentile(image,5) # Same as above -> can be substituted by 10 and 90
    mask = (image>q5) & (image<q95)
    median_bkg = np.median(image[mask])
    mean_bkg = np.mean(image[mask])
    bkg_noise = np.std(image[mask])
    return mean_bkg, median_bkg, bkg_noise

def get_predicted_bkg(visit,validation_dataset,db_file,band):
    if validation_dataset == 'Opsim':
        return get_opsim_bkg(visit,db_file,band)
    # TODO add imSim option
    #if validation_dataset == 'imSim':
    #    return get_imsim_bkg(visit,band)

def compute_sky_counts(mag,band,nsnap):
    # Data from https://github.com/lsst-pst/syseng_throughputs/blob/master/plots/table2
    if band == 'u':
        mag0 = 22.95
        counts0 = 50.2
    if band == 'g':
        mag0 = 22.24
        counts0 = 384.6
    if band == 'r':
        mag0 = 21.20
        counts0 = 796.2
    if band == 'i':
        mag0 = 20.47
        counts0 = 1108.1
    if band == 'z':
        mag0 = 19.60
        counts0 = 1687.9
    if band == 'y':
        mag0 = 18.63
        counts0 = 2140.8
    return nsnap*counts0*10**(-0.4*(mag-mag0))

def get_airmass_raw_seeing(visit,db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT airmass, filtSkyBrightness, finSeeing, rawSeeing, visitExpTime, fiveSigmaDepth FROM ObsHistory WHERE obsHistID==%d" %(visit))
    rows = cur.fetchall()
    return rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[0][4], rows[0][5]

def get_opsim_bkg(visit,db_file,band):
    skybrightness = get_airmass_raw_seeing(visit,db_file)[1]
    mean_bkg = compute_sky_counts(skybrightness,band,1)
    median_bkg = mean_bkg
    bkg_noise = np.sqrt(mean_bkg)
    return mean_bkg, median_bkg, bkg_noise


class ImagePkTest(BaseValidationTest):
    """
    Validation test that computes the power spectrum
    of a given raft image
    """
    def __init__(self,input_path,val_label,raft):
        self.input_path = input_path
        self.validation_data = astropy.table.Table.read(self.input_path)
        self.label = val_label
        self.raft = raft
    def post_process_plot(self, ax):
        ax.text(0.05, 0.95, self.input_path)
        ax.plot(self.validation_data['k'],self.validation_data['Pk'],
            label=self.label)
        ax.legend()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # The catalog instance is a focal plane
        test_raft = list(catalog_instance.focal_plane.rafts.values())[self.raft]
        if len(test_raft.sensors) != 9:
            return TestResult(skipped=True, summary='Raft is not complete')
        xdim, ydim = list(test_raft.sensors.values())[0].get_data().shape
        total_data = np.zeros((xdim*3,ydim*3))
        
        # Assemble the 3 x 3 raft's image: Need to use LSST's software to
        # handle the edges properly
        for i in range(0,3):
            for j in range(0,3):
                total_data[xdim*i:xdim*(i+1),ydim*j:ydim*(j+1)] = \ 
                    list(test_raft.sensors.values())[3*i+j].get_data()

        # FFT of the density contrast
        F1 = fftpack.fft2((total_data/np.mean(total_data)-1)) 
        F2 = fftpack.fftshift( F1 )
        psd2D = np.abs( F2 )**2 # 2D power
        pix_scale = 0.2/60*self.rebinning #pixel scale in arcmin
        kx = 1./pix_scale*np.arange(-F2.shape[0]/2,F2.shape[0]/2)*1./F2.shape[0]
        ky = 1./pix_scale*np.arange(-F2.shape[1]/2,F2.shape[1]/2)*1./F2.shape[1]
        kxx, kyy = np.meshgrid(kx,ky)
        rad = np.sqrt(kxx**2+kyy**2)
        bins = 1./pix_scale*np.arange(0,F2.shape[0]/2)*1./F2.shape[0]
        bin_space = bins[1]-bins[0]
        ps1d = np.zeros(len(bins))
        for i,b in enumerate(bins):
            ps1d[i] = np.mean(psd2D.T[(rad>b-0.5*bin_space) & (rad<b+0.5*bin_space)])/(F2.shape[0]*F2.shape[1]) 
        
        fig, ax = plt.subplots(2,1)
        for i in range(0,9):
            ax[0].hist(image[i].flatten(),histtype='step',label='Image: %d' % i)
     
        ax[1].plot(bins,ps1d,label=catalog_instace.sensor_raft)
        ax[1].set_xlabel('k [arcmin$^{-1}]')
        ax[1].set_ylabel('P(k)') 
        self.post_process_plot(ax[1])
        fig.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close(fig)
        # Check if the k binning/rebinning is the same before checking chi-sq
        if all(bins==self.validation_data['Pk']):
            score = (ps1d-self.validation_data['Pk'])
        # Check criteria to pass or fail (images in the edges of the focal plane 
        # will have way more power than the ones in the center if they are not
        # flattened
        return TestResult(score, passed=True)
 
class QuickBkgTest(BaseValidationTest):
    """
    Check of mean, median and standard deviation of the image background.
    We compare to expeted values by OpSim or imSim.
    """
    def __init__(self,label,bkg_validation_dataset,visit,band):
        self.validation_data =  get_predicted_bkg(visit,bkg_validation_dataset,band)
        self.label = label
        self.visit = visit
        self.band = band
    def post_process_plot(self, ax):
        ymin, ymax = ax[0].get_ylim()
        ax[0].plot(np.ones(3)*self.validation_data[0], np.linspace(ymin, ymax, 3),
            label='{}-Mean'.format(self.bkg_validation_dataset))
        ax[0].plot(np.ones(3)*self.validation_data[1], np.linspace(ymin, ymax, 3),
            label='{}-Median'.format(self.bkg_validation_dataset))
        ax[0].legend()
        ymin, ymax = ax[1].get_ylim()
        ax[1].plot(np.ones(3)*self.validation_data[2], np.linspace(ymin, ymax, 3),
            label='{}'.format(self.bkg_validation_dataset))
        ax[1].legend()             
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # Pass one focal plane and analyze sensor by sensor
        rafts = catalog_instance.focal_plane.rafts
        nsensors = len(catalog_instance._filelist)
        median_bkg = {}
        mean_bkg = {}
        bkg_noise = {}

        for rname, r in rafts.iteritems():
            for sname, s in r.sensors.iteritems():
                aux1, aux2, aux3 = compute_bkg(s.get_data())
                mean_bkg.update({'%s-%s' % (rname, sname) : aux1})
                median_bkg.update({'%s-%s' % (rname, sname) : aux2})
                bkg_noise.update({'%s-%s' % (rname, sname) : aux3})
        
        fig, ax = plt.subplots(1,2)
        ax[0].hist(list(mean_bkg.values()), histtype='step', label='Mean')
        ax[0].hist(list(median_bkg.values()), histtype='step', label='Median')
        ax[0].set_xlabel('{} [ADU]'.format(label))
        ax[0].set_ylabel('Number of sensors')
        ax[1].hist(list(bkg_noise.values()), histtype='step')
        ax[1].set_xlabel('{} noise [ADU]'.format(label))
        ax[1].set_ylabel('Number of sensors')      
