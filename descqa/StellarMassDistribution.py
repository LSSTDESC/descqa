import os

import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value

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
        Parameter
        -----------
        catalog_instance = Catalogue to use

        Return
        -----------
        - log10 of total stellar mass with no cuts applied
        - log10 of stellar mass in DCa2 catalogue with CMASS color and magnitude cuts applied
        - redshift
        - number density of galaxies (galaxies per square degree)
        """
        # pylint: disable=no-member

        gc = catalog_instance
        catSize = float(gc.sky_area)
        data = gc.get_quantities(["stellar_mass", "mag_true_i_lsst", "mag_true_r_lsst", "mag_true_g_lsst", "x", "y", "z"])
        smass = data["stellar_mass"]
        x, y, z = data["x"], data["y"], data["z"]
        log10smass = np.log10(smass)

        # calculating the reshifts from comoving distance
        com_dist = np.sqrt((x ** 2) + (y ** 2) + (z ** 2)) * u.Mpc  # units of Mpc
        min_indx = np.argmin(com_dist)
        max_indx = np.argmax(com_dist)

        cosmology = gc.cosmology
        zmin = z_at_value(cosmology.comoving_distance, com_dist[min_indx])
        zmax = z_at_value(cosmology.comoving_distance, com_dist[max_indx])
        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)

        CDgrid = cosmology.comoving_distance(zgrid) * (cosmology.H0 / (100.0 * u.km / u.s / u.Mpc))  # This has units of Mpc

        #  use interpolation to get redshifts for satellites only
        new_redshifts = np.interp(com_dist, CDgrid, zgrid)

        r = data["mag_true_r_lsst"]
        i = data["mag_true_i_lsst"]
        g = data["mag_true_g_lsst"]

        # applying CMASS cuts
        dperp = (r - i) - (g - r) / 8.0
        cond1 = dperp > 0.55
        cond2 = i < (19.86 + 1.6 * (dperp - 0.8))
        cond3 = (i < 19.9) & (i > 17.5)
        cond4 = (r - i) < 2
        cond5 = i < 21.5

        # applying the cuts to stellar mass
        smass_cmass_cut = smass[(cond1 & cond2 & cond3 & cond4 & cond5)]
        log10smass_cmass_cut = np.log10(smass_cmass_cut)

        print()
        print("minimum cmass-cut = ", np.min(log10smass_cmass_cut))
        print("maximum cmass-cut = ", np.max(log10smass_cmass_cut))
        print()

        numDen = len(smass_cmass_cut) / catSize
        return log10smass, log10smass_cmass_cut, new_redshifts, numDen

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        _, log_smass_cmass_DC2, _, numDen = self.get_smass(catalog_instance)

        cmass = self.validation_data

        x = cmass[:, 0]
        y = cmass[:, 1]

        bins = x

        plt.figure(1, figsize=(12, 6))
        plt.hist(x, bins=bins, weights=y, histtype="step", color="orange", density=True, linewidth=2, label="CMASS")
        plt.hist(log_smass_cmass_DC2, bins=np.linspace(10, 12.5, 50), color="teal", linewidth=2, density=True, histtype="step", label=catalog_name)
        plt.title(f"n[{catalog_name} = {numDen:.1f} , CMASS = 101] gals/sq deg")
        plt.xlabel(r"$\log(M_{\star}/M_{\odot})$", fontsize=20)
        plt.ylabel("N", fontsize=20)
        plt.legend(loc="best")
        plt.show()
        plt.savefig(output_dir + "Mstellar_distribution.png")
        plt.close()

        # CMASS stellar mass mean
        log_cmass_mean = 11.25

        # score is defined as error away from CMASS stellar mass mean
        score = (np.mean(log_smass_cmass_DC2) - log_cmass_mean) / log_cmass_mean

        return TestResult(score=score, passed=(score < 1.0))
