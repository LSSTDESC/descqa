from __future__ import unicode_literals, absolute_import, division
import os
import sqlite3
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['QuickBkgTest']


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

    q_low, q_high = np.percentile(image, [5, 95]) # This is kind of arbitrary but it works fine
    image = image[(image > q_low) & (image < q_high)] 
    return np.mean(image), np.median(image), np.std(image)

def get_predicted_bkg(visit, validation_dataset, db_file, band):
    if validation_dataset.lower() == 'opsim':
        return get_opsim_bkg(visit, db_file, band)
    else:
        raise NotImplementedError('only "opsim" is currently supported')
    # TODO add imSim option
    #if validation_dataset == 'imSim':
    #    return get_imsim_bkg(visit,band)


def compute_sky_counts(mag, band, nsnap):
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
    return nsnap * counts0 * 10**(-0.4 * (mag - mag0))


def get_airmass_raw_seeing(visit, db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        "SELECT airmass, filtSkyBrightness, finSeeing, rawSeeing, visitExpTime, fiveSigmaDepth FROM ObsHistory WHERE obsHistID==%d"
        % (visit))
    rows = cur.fetchall()
    return rows[0]

def get_opsim_bkg(visit,db_file,band):
    skybrightness = get_airmass_raw_seeing(int(visit),db_file)[1]
    # We are going to compute the background counts given OpSim's sky-brightness
    mean_bkg = compute_sky_counts(skybrightness,band,1)
    median_bkg = mean_bkg # We assume that the background is completely homogeneous
    bkg_noise = np.sqrt(mean_bkg) # We assume Poisson noise
    return mean_bkg, median_bkg, bkg_noise

class QuickBkgTest(BaseValidationTest):
    """
    Check of mean, median and standard deviation of the image background.
    We compare to expeted values by OpSim or imSim.
   
    Args:
    -----
     
    label (str): x-label for the validation plots
    visit (int): Visit numbr to analyze
    band (str): Filter/band to analyze
    bkg_validation_dataset (str): Name of the validation data to which compare, for now,
        only opsim is available.
    """

    def __init__(self, label, bkg_validation_dataset, visit, band, db_file, **kwargs):
        # pylint: disable=W0231
        self.validation_data = get_predicted_bkg(visit, bkg_validation_dataset, db_file, band)
        self.label = label
        self.visit = visit
        self.band = band
        self.bkg_validation_dataset = bkg_validation_dataset

    def post_process_plot(self, ax):
        ymin, ymax = ax[0].get_ylim()
        ax[0].plot(
            np.ones(3) * self.validation_data[0],
            np.linspace(ymin, ymax, 3),
            label='{}-Mean'.format(self.bkg_validation_dataset))
        ax[0].plot(
            np.ones(3) * self.validation_data[1],
            np.linspace(ymin, ymax, 3),
            label='{}-Median'.format(self.bkg_validation_dataset))
        ax[0].legend()
        ymin, ymax = ax[1].get_ylim()
        ax[1].plot(
            np.ones(3) * self.validation_data[2],
            np.linspace(ymin, ymax, 3),
            label='{}'.format(self.bkg_validation_dataset))
        ax[1].legend()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # Pass one focal plane and analyze sensor by sensor
        rafts = catalog_instance.focal_plane.rafts
        median_bkg = {}
        mean_bkg = {}
        bkg_noise = {}

        for rname, r in rafts.items():
            for sname, s in r.sensors.items():
                aux1, aux2, aux3 = compute_bkg(s.get_data())
                mean_bkg.update({'%s-%s' % (rname, sname): aux1})
                median_bkg.update({'%s-%s' % (rname, sname): aux2})
                bkg_noise.update({'%s-%s' % (rname, sname): aux3})

        fig, ax = plt.subplots(2, 1)
        ax[0].hist(list(mean_bkg.values()), histtype='step', label='Mean')
        ax[0].hist(list(median_bkg.values()), histtype='step', label='Median')
        ax[0].set_xlabel('{} [ADU]'.format(self.label))
        ax[0].set_ylabel('Number of sensors')
        ax[1].hist(list(bkg_noise.values()), histtype='step')
        ax[1].set_xlabel('{} noise [ADU]'.format(self.label))
        ax[1].set_ylabel('Number of sensors') 
        score = sum(median_bkg.values()) / len(median_bkg) / self.validation_data[0] - 1.
        score = np.fabs(score)
        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'plot_png'))
        plt.close(fig)
        return TestResult(score, passed=score < 0.2)

