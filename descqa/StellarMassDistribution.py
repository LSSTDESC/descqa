import os

import numpy as np
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ["StellarMassTest"]


class StellarMassTest(BaseValidationTest):

    """
    This validation test looks at stellar mass distribution
    of DC2 catalogs to make sure it matches the distribution
    of CMASS galaxies which have constraints on both
    magnitude and color og galaxies and also checks the
    number density of galaxies per square degree as the
    score to pass the test.
    """

    def __init__(self, **kwargs):
        # load validation data
        path = os.path.join(self.data_dir, "stellar_mass_dist", "CMASS_data.txt")
        self.validation_data = np.loadtxt(path)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.legend_location = kwargs.get('legend_location', 'upper right')
        self.font_size = kwargs.get('font_size', 22)
        self.text_size = kwargs.get('text_size', 20)
        self.legend_size = kwargs.get('legend_size', 18)
        self.Mlo = kwargs.get('Mlo', 10.)
        self.Mhi = kwargs.get('Mhi', 12.5)
        
    @staticmethod
    def get_smass(catalog_instance):
        """
        Parameters
        ----------
        catalog_instance = Catalogue to use

        Returns
        -------
        - log10 of stellar mass with CMASS color and magnitude cuts applied
        - number density of galaxies (galaxies per square degree)
        """
        gc = catalog_instance
        sky_area = float(gc.sky_area)

        cols = {
            "smass": gc.first_available("stellar_mass"),
            "g": gc.first_available("mag_true_g_lsst"),
            "r": gc.first_available("mag_true_r_lsst"),
            "i": gc.first_available("mag_true_i_lsst"),
        }
        if not all(cols.values()):
            raise KeyError("Not all needed quantities exist!!")

        valid_smass = GCRQuery("{smass} > 0".format(**cols))
        cmass_cuts = GCRQuery(
            "({r} - {i}) - ({g} - {r}) / 8 > 0.55".format(**cols),
            "{i} < 19.86 + 1.6 * (({r} - {i}) - ({g} - {r}) / 8 - 0.8)".format(**cols),
            "{i} < 19.9".format(**cols),
            "{i} > 17.5".format(**cols),
            "{r} - {i} < 2".format(**cols),
        )

        log_smass_cmass = np.log10(gc.get_quantities([cols["smass"]], filters=[valid_smass, cmass_cuts])[cols["smass"]])

        print()
        print("minimum cmass-cut = ", np.min(log_smass_cmass))
        print("maximum cmass-cut = ", np.max(log_smass_cmass))
        print()

        numDen = len(log_smass_cmass) / sky_area
        return log_smass_cmass, numDen

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        log_smass_cmass, numDen = self.get_smass(catalog_instance)

        if self.truncate_cat_name:
            catalog_name = catalog_name.partition("_")[0]

        print(catalog_name)
        x = self.validation_data[:, 0]
        y = self.validation_data[:, 1]

        fig = plt.figure(1, figsize=(12, 6))
        plt.hist(x, bins=x, weights=y, histtype="step", color="teal", density=True, linewidth=2, label="CMASS")
        plt.hist(log_smass_cmass, bins=np.linspace(self.Mlo, self.Mhi, 50), color="orange", linewidth=2, density=True,
                 histtype="step", label=catalog_name)
        text = '{}: {:.1f} gals/sq. deg.\nCMASS: 101 gals/sq. deg.'.format(catalog_name, numDen)
        #plt.title(f"n[{catalog_name} = {numDen:.1f} , CMASS = 101] gals/sq deg")
        ax = plt.gca()
        plt.text(0.01, 0.86, text, fontsize=self.text_size, transform=ax.transAxes)
        plt.xlabel(r"$\log_{10}(M^*/M_{\odot})$", fontsize=self.font_size)
        plt.ylabel("$N$", fontsize=self.font_size)
        plt.xlim(self.Mlo + 0.3, self.Mhi - 0.3)
        plt.legend(loc=self.legend_location, fontsize=self.legend_size)
        plt.show()
        plt.savefig(os.path.join(output_dir, "Mstellar_distribution.png"))
        plt.close()

        # CMASS stellar mass mean
        log_cmass_mean = 11.25

        # score is defined as error away from CMASS stellar mass mean
        score = (np.mean(log_smass_cmass) - log_cmass_mean) / log_cmass_mean

        return TestResult(score=score, passed=(score < 1.0))
