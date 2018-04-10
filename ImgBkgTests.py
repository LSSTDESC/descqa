from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt
from scipy import fftpack
import astropy.table

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

class ImagePkTest(BaseValidationTest):
    """
    Validation test that computes the power spectrum
    of a given raft image
    """
    def __init__(self,input_path,val_label):
        self.input_path = input_path
        self.validation_data = astropy.table.Table.read(self.input_path)
        self.label = val_label
    def post_process_plot(self, ax):
        ax.text(0.05, 0.95, self.input_path)
        ax.plot(self.validation_data['k'],self.validation_data['Pk'],
            label=self.label)
        ax.legend()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # Pass a catalog instance that is a focal plane and read one raft?
        if not len(catalog_instance.data_path)==9:
            return TestResult(skipped=True, summary='Raft is not complete')
        
        total_data = np.zeros((images[0].shape[0]*3,images[0].shape[1]*3))
        
        # Assemble the 3 x 3 raft's image: Need to use LSST's software to
        # handle the edges properly
        for i in range(0,3):
            for j in range(0,3):
                total_data[images[0].shape[0]*i:images[0].shape[0]*(i+1), \ 
                    images[0].shape[1]*j:images[0].shape[1]*(j+1)] = \ 
                        images[3*i+j]

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
    def __init__(self,label,check_imsim=False):
        self.check_imsim = check_imsim
        self.validation_data = 
        self.label = label
    def post_process_plot(self, ax):
        ax.text(0.05, 0.95, self.input_path)
        ymin, ymax = ax.get_ylim()
        ax.plot(np.ones(3)*self.validation_data, np.linspace(ymin, ymax, 3),
            label=self.label)
        ax.legend()
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # Pass one focal plane and analyze sensor by sensor
         
