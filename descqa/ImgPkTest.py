from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from scipy.stats import binned_statistic, chi2
from astropy.table import Table
from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import first, is_string_like

__all__ = ['ImgPkTest']

class ImgPkTest(BaseValidationTest):
    """
    Validation test that computes the power spectrum
    of a given raft image

    Args:
    -----
    raft: str, list of str, or None
        Raft number to analyze (e.g., 'R01', 'R10', 'R22').
    rebinning: int, or None
        rebinning image by this factor
    validation_data_path: str, or None
        path to validation data
    validation_data_label: str
        label of validation data
    pixel_scale : float
        pixel scale  in arcmin
    """

    def __init__(self, raft=None, rebinning=None, validation_data_path=None,
                 validation_data_label=None, pixel_scale=(0.2/60.0),
                 **kwargs):
        # pylint: disable=W0231
        self.raft = raft
        self.rebinning = rebinning
        if validation_data_path is None:
            self.validation_data = None
        else:
            self.validation_data = Table.read(validation_data_path)
        self.validation_data_label = validation_data_label
        self.pixel_scale = pixel_scale

    def get_rebinning(self, raft):
        if self.rebinning is None:
            return first(raft.sensors.values()).default_rebinning
        return self.rebinning

    def calc_psd(self, raft, bins=200):
        rebinning = self.get_rebinning(raft)

        # Assemble the 3 x 3 raft's image
        # TODO: Need to use LSST's software to handle the gaps properly
        total_data = np.array([raft.sensors['S%d%d'%(i,j)].get_data(rebinning=rebinning) for i in range(3) for j in range(3)])
        xdim, ydim = total_data.shape[1:]
        total_data = total_data.reshape(3, 3, xdim, ydim).swapaxes(1, 2).reshape(3*xdim, 3*ydim)

        # FFT of the density contrast
        FT = np.fft.fft2(total_data / total_data.mean() - 1)
        n_kx, n_ky = FT.shape
        psd = np.square(np.abs(FT)).ravel()
        spacing = self.pixel_scale * rebinning
        k_rad = np.hypot(*np.meshgrid(np.fft.fftfreq(n_kx, spacing), np.fft.fftfreq(n_ky, spacing), indexing='ij')).ravel()

        k_rad /= (2.0 * np.pi)
        psd *= (spacing / n_kx) * (spacing / n_ky)

        return binned_statistic(k_rad, [k_rad, psd], bins=bins)[0]

    def plot_hist(self, ax, raft):
        rebinning = self.get_rebinning(raft)
        for key, image in raft.sensors.items():
            ax.hist(image.get_data(rebinning=rebinning).ravel(),
                    histtype='step', range=(200, 2000), bins=200, label=key, log=True)
        ax.set_xlabel('Background level [ADU]')
        ax.set_ylabel('Number of pixels')
        ax.legend(loc='best', ncol=2)
        return ax

    def plot_psd(self, ax, k, psd, label):
        ax.loglog(k, psd, label=label)
        if self.validation_data is not None:
            ax.loglog(self.validation_data['k'], self.validation_data['Pk'], label=self.validation_data_label)
        ax.set_xlabel('k [arcmin$^{-1}$]')
        ax.set_ylabel('P(k)')
        ax.set_xlim(0.005, 2)
        ax.set_ylim(1.0e-4, 2)
        ax.legend(loc='best')
        return ax

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # The catalog instance is a focal plane
        rafts = catalog_instance.focal_plane.rafts

        if self.raft is None:
            raft_names = list(rafts)
        elif is_string_like(self.raft):
            raft_names = [self.raft]
        else:
            raft_names = list(self.raft)

        if not all(raft_name in rafts for raft_name in raft_names):
            return TestResult(skipped=True, summary='Not all rafts exist!')

        total_chi2 = 0
        count = 0
        for raft_name in raft_names:
            raft = rafts[raft_name]
            fig, ax = plt.subplots(2, 1, figsize=(7, 7))
            self.plot_hist(ax[0], raft)
            if len(raft.sensors) == 9:
                k, psd = self.calc_psd(raft)
                self.plot_psd(ax[1], k, psd, label=raft_name)
                if self.validation_data is not None:
                    psd_log_interp = np.interp(self.validation_data['k'], k, np.log(psd), left=-np.inf, right=-np.inf)
                    count += 1
                    total_chi2 += np.square((psd_log_interp - np.log(self.validation_data['Pk']))).sum()
            else:
                msg = 'Raft {} is not complete!'.format(raft_name)
                ax[1].text(0.05, 0.5, msg, transform=ax[1].transAxes)
                print('[WARNING]', msg)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'plot_{}.png'.format(raft_name)))
            plt.close(fig)
        if count:
            total_chi2 /= count
        ndof = len(self.validation_data['k']) - 1
        score = 1 - chi2.cdf(total_chi2, ndof)
        # Check criteria to pass or fail (images in the edges of the focal plane
        # will have way more power than the ones in the center if they are not
        # flattened, we require the power to be within 2-sigma ( p < 0.95)
        return TestResult(score=score, passed=(score < 0.95))
