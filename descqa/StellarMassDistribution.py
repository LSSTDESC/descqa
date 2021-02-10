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

        cols = {
            "smass": gc.first_available("stellar_mass"),
            "g": gc.first_available("mag_true_g_lsst"),
            "r": gc.first_available("mag_true_r_lsst"),
            "i": gc.first_available("mag_true_i_lsst"),
        }

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

        numDen = len(log_smass_cmass) / float(gc.sky_area)
        return log_smass_cmass, numDen

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        log_smass_cmass, numDen = self.get_smass(catalog_instance)

        x = self.validation_data[:, 0]
        y = self.validation_data[:, 1]

        plt.figure(1, figsize=(12, 6))
        plt.hist(x, bins=x, weights=y, histtype="step", color="orange", density=True, linewidth=2, label="CMASS")
        plt.hist(log_smass_cmass, bins=np.linspace(10, 12.5, 50), color="teal", linewidth=2, density=True, histtype="step", label=catalog_name)
        plt.title(f"n[{catalog_name} = {numDen:.1f} , CMASS = 101] gals/sq deg")
        plt.xlabel(r"$\log(M_{\star}/M_{\odot})$", fontsize=20)
        plt.ylabel("N", fontsize=20)
        plt.legend(loc="best")
        plt.show()
        plt.savefig(os.path.join(output_dir, "Mstellar_distribution.png"))
        plt.close()

        # CMASS stellar mass mean
        log_cmass_mean = 11.25

        # score is defined as error away from CMASS stellar mass mean
        score = (np.mean(log_smass_cmass) - log_cmass_mean) / log_cmass_mean

        return TestResult(score=score, passed=(score < 1.0))
