from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from scipy.stats import binned_statistic
import astropy.table
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['ImgPkTest']


def first(iterable, default=None):
    return next(iter(iterable), default)


class ImgPkTest(BaseValidationTest):
    """
    Validation test that computes the power spectrum
    of a given raft image

    Args:
    -----
    input_path: (str) Directory where the raw e-images live.
    val_label: (str) Label of the horizontal axis for the validation plots.
    raft: (str) Raft number to analyze (e.g., 'R01', 'R10', 'R22').
    """

    def __init__(self, input_path, val_label, raft, **kwargs):
        # pylint: disable=W0231
        self.input_path = input_path
        self.validation_data = astropy.table.Table.read(self.input_path)
        self.label = val_label
        self.raft = raft

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # The catalog instance is a focal plane
        test_raft = catalog_instance.focal_plane.rafts[self.raft]
        rebinning = first(test_raft.sensors.values()).rebinning
        if not rebinning or rebinning < 0:
            return TestResult(skipped=True, summary='invalid rebinning value: {}'.format(rebinning))
        if len(test_raft.sensors) != 9:
            return TestResult(skipped=True, summary='Raft is not complete')

        # Assemble the 3 x 3 raft's image
        # TODO: Need to use LSST's software to handle the gaps properly
        total_data = np.array([test_raft.sensors['S%d%d'%(i, j)].get_data() for i in range (3) for j in range(3)])
        xdim, ydim = total_data.shape[1:]
        total_data = total_data.reshape(3, 3, xdim, ydim).swapaxes(1, 2).reshape(3*xdim, 3*ydim)

        # FFT of the density contrast
        FT = np.fft.fft2(total_data / total_data.mean() - 1)
        n_kx, n_ky = FT.shape
        psd2D = np.square(np.abs(FT)).ravel() # 2D power, flattened
        psd2D /= (n_kx * n_ky)
        pix_scale = 0.2 / 60.0 #pixel scale in arcmin
        spacing = pix_scale * rebinning
        rad = np.hypot(*np.meshgrid(np.fft.fftfreq(n_kx, spacing), np.fft.fftfreq(n_ky, spacing), indexing='ij')).ravel()
        rad /= (2.0 * np.pi)

        psd1D, bin_edges, _ = binned_statistic(rad, psd2D, bins=self.validation_data['k'])
        bin_center = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        score = np.square(psd1D/self.validation_data['Pk'][:-1] - 1.0).sum()

        # make plot
        fig, ax = plt.subplots(2, 1, figsize=(7, 7))
        for key, image in test_raft.sensors.items():
            ax[0].hist(image.get_data().ravel(), histtype='step', range=(200, 2000), bins=200, label=key, log=True)
        ax[0].set_xlabel('Background level [ADU]')
        ax[0].set_ylabel('Number of pixels')
        ax[0].legend(loc='best')
        ax[1].plot(bin_center, psd1D, label=self.raft)
        ax[1].plot(self.validation_data['k'], self.validation_data['Pk'], label='validation')
        ax[1].set_xlabel('k [arcmin$^{-1}$]')
        ax[1].set_ylabel('P(k)')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_ylim(1, 1000)
        ax[1].legend(loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close(fig)

        # Check criteria to pass or fail (images in the edges of the focal plane
        # will have way more power than the ones in the center if they are not
        # flattened
        return TestResult(score=score, inspect_only=True)
