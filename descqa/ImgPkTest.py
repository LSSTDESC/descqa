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

    def calc_psd(self, image_data, rebinning=1, bins=200):
        FT = np.fft.fft2(image_data / image_data.mean() - 1)
        n_kx, n_ky = FT.shape
        psd = np.square(np.abs(FT)).ravel()
        spacing = self.pixel_scale * rebinning
        k_rad = np.hypot(*np.meshgrid(np.fft.fftfreq(n_kx, spacing), np.fft.fftfreq(n_ky, spacing), indexing='ij')).ravel()
        k_rad /= (2.0 * np.pi)
        psd *= (spacing / n_kx) * (spacing / n_ky)
        return binned_statistic(k_rad, [k_rad, psd], bins=bins)[0]

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        if hasattr(catalog_instance, 'focal_plane'):
            focal_plane = catalog_instance.focal_plane
        elif hasattr(catalog_instance, 'focal_planes'):
            focal_plane = first(catalog_instance.focal_planes.values())
        else:
            return TestResult(skipped=True, summary='Not an e-image!')

        rafts = focal_plane.rafts

        if self.rebinning is None:
            rebinning = catalog_instance.default_rebinning
        else:
            rebinning = self.rebinning

        if self.raft is None:
            raft_names = list(rafts)
        elif is_string_like(self.raft):
            raft_names = [self.raft]
        else:
            raft_names = list(self.raft)

        if not all(raft_name in rafts for raft_name in raft_names):
            return TestResult(skipped=True, summary='Not all rafts exist!')

        sensor_names = ['S%d%d'%(i,j) for i in range(3) for j in range(3)]

        total_chi2 = 0
        total_dof = 0

        for raft_name in raft_names:
            raft = rafts[raft_name]
            data = [raft.sensors[name].get_data(rebinning) if name in raft.sensors else None for name in sensor_names]

            fig, ax = plt.subplots(2, 1, figsize=(7, 8))
            for sensor, data_this in zip(sensor_names, data):
                if data_this is None:
                    continue

                ax[0].hist(data_this.ravel(), np.linspace(200, 2000, 181),
                           histtype='step', log=True, label=sensor)
                ax[1].loglog(*self.calc_psd(data_this, rebinning), label=sensor, alpha=0.8)

            if sum((1 for data_this in data if data_this is not None)) == 9:
                data = np.array(data)
                xdim, ydim = data.shape[1:]
                data = data.reshape(3, 3, xdim, ydim).swapaxes(1, 2).reshape(3*xdim, 3*ydim)
                k, psd = self.calc_psd(data, rebinning)
                ax[1].loglog(k, psd, label='all', c='k')
            else:
                psd = None

            if self.validation_data is not None:
                ax[1].loglog(self.validation_data['k'], self.validation_data['Pk'], label=self.validation_data_label, c='r', ls=':')

            if self.validation_data is not None and psd is not None:
                psd_log_interp = np.interp(self.validation_data['k'], k, np.log(psd), left=np.nan, right=np.nan)
                mask = np.isfinite(psd_log_interp)
                total_dof += np.count_nonzero(mask)
                total_chi2 += np.square((psd_log_interp[mask] - np.log(self.validation_data['Pk'][mask]))).sum()

            ax[0].legend(ncol=3)
            ax[1].legend(ncol=3)

            ax[0].set_title('{} - {}'.format(raft_name, catalog_name))
            ax[0].set_xlabel('Background level [ADU]')
            ax[0].set_ylabel('Number of pixels')
            ax[0].set_ylim(None, 1e5)
            ax[1].set_xlabel('k [arcmin$^{-1}$]')
            ax[1].set_ylabel('P(k)')
            ax[1].set_xlim(0.005, 2)
            ax[1].set_ylim(1.0e-4, 2)

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'plot_{}.png'.format(raft_name)))
            plt.close(fig)

        score = chi2.cdf(total_chi2, total_dof)

        # Check criteria to pass or fail (images in the edges of the focal plane
        # will have way more power than the ones in the center if they are not
        # flattened, we require the power to be within 2-sigma ( p < 0.95)
        return TestResult(score=score, passed=(score < 0.95))
