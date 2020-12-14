import os
import pickle
import numpy as np
import treecorr
from .kcorrect_wrapper import kcorrect
from .base import BaseValidationTest, TestResult
from .plotting import plt
import warnings


__all__ = ["ConditionalLuminosityFunction_redmapper"]


def cluster_Lcount(lumbins, limmag):
    # count number of cluster for mag bins
    nclusters_lum = np.zeros_like(lumbins)
    binlist = np.zeros_like(limmag).astype(int)
    dlum = lumbins[1] - lumbins[0]
    minlum = lumbins[0] - dlum / 2.0
    nlumbins = len(lumbins)
    p = np.zeros_like(limmag) + 1

    for i, limmag_in in enumerate(limmag):
        mybin = int(np.floor((limmag_in - minlum) / dlum))
        binlist[i] = 0
        if mybin > nlumbins:
            nclusters_lum = nclusters_lum + p[i]
            binlist[i] = nlumbins
        if (mybin > 0) & (mybin <= nlumbins):
            nclusters_lum[:mybin] = nclusters_lum[:mybin] + p[i]
            binlist[i] = mybin
    return nclusters_lum, binlist


def count_galaxies_p_cen(cenmag, lumbins, p_cen):
    # counting central galaxies
    nlum = len(lumbins)
    dlum = lumbins[1] - lumbins[0]
    minlum = lumbins[0] - dlum / 2.0
    chto_countArray = np.zeros([len(cenmag), nlum])
    mybin = np.floor((cenmag[:, :] - minlum) / dlum).astype(np.int)
    p_cen = p_cen.reshape(-1, 1)
    ncen = np.zeros(p_cen.shape)
    ncen = np.hstack(((np.ones(p_cen.shape[0])).reshape(-1, 1), ncen[:, :-1])).astype(
        np.int
    )
    weight = p_cen

    ys = np.outer(np.ones(p_cen.shape[0]), np.arange(p_cen.shape[1]))
    mask = ((mybin >= 0) & (mybin < nlum) & (ys < ncen)).astype(np.float64)
    for i in range(nlum):
        masknew = (mybin == i).astype(np.float64) * mask
        newNewbin = np.sum(masknew * weight, axis=1)
        chto_countArray[:, i] = newNewbin[:]
    return chto_countArray


def count_galaxies_p(c_mem_id, scaleval, g_mem_id, p, mag, lumbins):
    # counting all galaxies
    nclusters = len(c_mem_id)
    nlum = len(lumbins)
    dlum = lumbins[1] - lumbins[0]
    minlum = lumbins[0] - dlum / 2.0
    maxlum = lumbins[-1] + dlum / 2.0
    count_arr = np.zeros([nclusters, nlum])

    max_id = np.max(c_mem_id)
    index = np.zeros(max_id + 1) - 100
    index[c_mem_id] = range(len(c_mem_id))
    mylist = np.where((mag <= maxlum) & (mag >= minlum))[0]
    if mylist.size == 0:
        print("WARNING:  No galaxies found in range {0}, {1}".format(minlum, maxlum))
        return count_arr

    for gal in mylist:
        if g_mem_id[gal] > max_id:
            continue
        mycluster = int(index[g_mem_id[gal]])
        mybin = np.floor((mag[gal] - minlum) / dlum)
        mybin = mybin.astype(int)
        if (mybin < 0) | (mybin >= nlum) | (mycluster == -100):
            continue
        count_arr[mycluster, mybin] += p[gal] * scaleval[mycluster]

    return count_arr


class ConditionalLuminosityFunction_redmapper(BaseValidationTest):
    old_lambd_bins = None

    def __init__(self, **kwargs):  # pylint: disable=W0231
        self.band = kwargs.get("band1", "i")
        self.band_kcorrect = kwargs.get("band_kcorrect", "u, g, r, i, z")
        self.band_kcorrect = [x.strip() for x in self.band_kcorrect.split(",")]
        self.possible_mag_fields = ("mag_{0}_lsst",)
        self.possible_magerr_fields = ("magerr_{0}_lsst",)
        self.bandshift = kwargs.get("bandshift", 0.3)
        self.njack = kwargs.get("njack", 20)
        self.z_bins = np.array(kwargs.get("z_bins", (0.1, 0.3, 1.0)))
        self.data_z_mins = kwargs.get("data_z_mins", (0.2, 0.2))
        self.data_z_maxs = kwargs.get("data_z_maxs", (0.3, 0.3))
        self.n_z_bins = len(self.z_bins) - 1
        defaultbins = np.array([5.0, 10.0, 15.0, 100.0]).reshape(-1, 1)
        self.is_abundance_matching = kwargs.get("abundance_matching", False)
        self.lambd_bins = np.tile(defaultbins, (1, self.n_z_bins)).T
        self.magnitude_bins = np.linspace(*kwargs.get("magnitude_bins", (-26, -18, 29)))
        self.n_magnitude_bins = len(self.magnitude_bins) - 1
        self.nlambd_bins = len(self.lambd_bins[0]) - 1
        self.dmag = self.magnitude_bins[1:] - self.magnitude_bins[:-1]
        self.lambd_center = (self.lambd_bins[1:] + self.lambd_bins[:-1]) * 0.5
        self.mag_center = (self.magnitude_bins[1:] + self.magnitude_bins[:-1]) * 0.5
        self.compared_survey = kwargs.get("survey", "SDSS")
        self.filters = kwargs.get("filters", "buzzard_filters.dat")
        self._other_kwargs = kwargs

    def prepare_galaxy_catalog(self, gc):
        # list all quantities that is needed
        quantities_needed = {
            "cluster_id_member",
            "cluster_id",
            "ra",
            "dec",
            "ra_cluster",
            "dec_cluster",
            "richness",
            "redshift_true",
            "lim_limmag_dered",
            "p_mem",
            "scaleval",
            "p_cen",
            "redshift_cluster",
        }
        try:
            magnitude_fields = []
            magnitude_err_fields = []
            for band in self.band_kcorrect:
                possible_mag_fields = [
                    magfield.format(band) for magfield in self.possible_mag_fields
                ]
                possible_magerr_fields = [
                    magfield.format(band) for magfield in self.possible_magerr_fields
                ]
                magnitude_fields.append(gc.first_available(*possible_mag_fields))
                magnitude_err_fields.append(gc.first_available(*possible_magerr_fields))
        except ValueError:
            return
        for field in zip(magnitude_fields, magnitude_err_fields):
            quantities_needed.add(field[0])
            quantities_needed.add(field[1])
        if not gc.has_quantities(quantities_needed):
            print(quantities_needed)
            return
        return magnitude_fields, magnitude_err_fields, quantities_needed

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        # loop over all catalog
        prepared = self.prepare_galaxy_catalog(catalog_instance)
        if prepared is None:
            TestResult(skipped=True)
        magnitude_fields, magnitude_err_fields, quantities_needed = prepared
        quant = catalog_instance.get_quantities(quantities_needed)
        # Abundance matching
        if self.is_abundance_matching:
            self.abundance_matching(
                quant["richness"], catalog_instance.get_catalog_info()["sky_area"]
            )
        # get magnitude
        mag = []
        magerr = []
        for magzip in zip(magnitude_fields, magnitude_err_fields):
            mag.append(quant[magzip[0]])
            magerr.append(quant[magzip[1]])
        mag = np.array(mag).T
        magerr = np.array(magerr).T
        # get k correction
        z = quant["redshift_true"]
        kcorrect_path = self.data_dir + "/clf/kcorrect/" + catalog_name + "_kcorr.cache"
        if not os.path.exists(kcorrect_path):
            kcorr = kcorrect(mag, magerr, z, self.bandshift, filters=self.filters)
            if kcorr is not None:
                np.savetxt(kcorrect_path, kcorr)
        else:
            kcorr = np.loadtxt(kcorrect_path)

        # Preprocess for all quantity
        # get analysis band and do kcorrection
        analindex = self.band_kcorrect.index(self.band)
        Mag = (
            mag[:, analindex]  # pylint: disable=unsubscriptable-object
            - catalog_instance.cosmology.distmod(z).value
        )
        if kcorr is not None :
            Mag -= kcorr[:, analindex]
        # Mask for central galaxy
        mask = quant["richness"] > 1
        limmag = quant["lim_limmag_dered"][mask]
        limmag = (
            limmag
            - catalog_instance.cosmology.distmod(quant["redshift_cluster"]).value[mask]
        )
        cenMag, cengalindex = self.get_central_mag_id(
            quant["cluster_id"][mask],
            quant["cluster_id_member"],
            quant["ra_cluster"][mask],
            quant["dec_cluster"][mask],
            quant["ra"],
            quant["dec"],
            Mag,
        )
        np.save(
            "cenMag.npy",
            cenMag,
        )
        np.save(
            "Mag.npy",
            Mag,
        )

        # For halo run pcen are 1
        pcen_all = np.zeros(len(quant["ra"]))
        pcen_all[cengalindex.flatten().astype(int)] = quant["p_cen"][mask]

        # Prepare for jackknife
        jackList = self.make_jack_samples_simple(
            quant["ra_cluster"][mask], quant["dec_cluster"][mask]
        )

        match_index_jack = self.getjackgal(
            jackList[0], quant["cluster_id"][mask], quant["cluster_id_member"]
        )

        # calculating clf
        cenclf, satclf, covar_cen, covar_sat = self.redm_clf(
            Mag,
            cenMag,
            pcen_all,
            jackList,
            quant["cluster_id"][mask],
            quant["cluster_id_member"],
            match_index=match_index_jack,
            limmag=limmag,
            scaleval=quant["scaleval"][mask],
            pmem=quant["p_mem"],
            pcen=quant["p_cen"][mask],
            cluster_lm=quant["richness"][mask],
            cluster_z=quant["redshift_cluster"][mask],
        )
        #Read compare data
        data = self.read_compared_data(catalog_instance.cosmology.h, kcorr is None)
        scores_shift = []
        scores_scatter = []
        for i in range(self.n_z_bins):
            for j in range(self.nlambd_bins):
                # calculate the relative brightness of centrals and satellites
                meancen_ref = np.average(
                    data["centrals"][i, j, :, 0], weights=data["centrals"][i, j, :, 1]
                )
                meansat_ref = np.average(
                    data["satellites"][i, j, :, 0],
                    weights=data["satellites"][i, j, :, 1],
                )
                meancen = np.average(self.mag_center, weights=cenclf[i, j])
                meansat = np.average(self.mag_center, weights=satclf[i, j])
                scores_shift.append(
                    np.abs(meancen - meansat - (meancen_ref - meansat_ref))
                )

                # calculate the std of central luminosity given richness
                std_cen_ref = np.sqrt(
                    np.average(
                        (data["centrals"][i, j, :, 0] - meancen_ref) ** 2,
                        weights=data["centrals"][i, j, :, 1],
                    )
                )
                std_cen = np.sqrt(
                    np.average((self.mag_center - meancen) ** 2, weights=cenclf[i, j])
                )
                scores_scatter.append(np.abs(std_cen - std_cen_ref))
        scores = [np.max(scores_shift), np.max(scores_scatter)]
        clf = {"satellites": satclf, "centrals": cenclf}
        covar = {"satellites": covar_sat, "centrals": covar_cen}
        if kcorr is None:
            name = catalog_name + " no kcorrect"
        else:
            name = catalog_name + " kcorrect z={0}".format(self.bandshift)
        self.make_plot(
            clf,
            covar,
            data,
            name,
            os.path.join(output_dir, "clf_redmapper.png"),
        )
        if (scores[0] < 0.5) & (scores[1] < 0.5):
            return TestResult(np.max(scores), passed=True)
        else:
            return TestResult(np.max(scores), passed=False)

    def read_compared_data(self, h, nokcorr):
        #check whether kcorrect is available
        if nokcorr:
            warnings.warn("no kcorrection availabel, comparing to non-kcorrected data")
            kcorrectfield = "_nokcorrected"
        else:
            kcorrectfield = ""
        # read the data to be compared to
        zmin = np.array(self.data_z_mins)
        zmax = np.array(self.data_z_maxs)
        galaxytype = ["centrals", "satellites"]
        data = {}
        for galtype in galaxytype:
            data[galtype] = []
        for i, zrange in enumerate(zip(zmin, zmax)):
            lambds = self.lambd_bins[i]
            for lambdlow, lambdhigh in zip(lambds[:-1], lambds[1:]):
                for galtype in galaxytype:
                    if galtype == "centrals":
                        name = "cen"
                    else:
                        name = "sat"
                    try:
                        loaded_data = np.loadtxt(
                            self.data_dir + "/clf/{5}/clf_{4}_z_{0}_{1}_lm_{2}_{3}{6}.dat".format(
                                zrange[0],
                                zrange[1],
                                lambdlow,
                                lambdhigh,
                                name,
                                self.compared_survey, kcorrectfield
                            )
                        )
                        loaded_data[:, 0] += 5 * np.log10(
                            h
                        )  # kcorrect distmodule assume h=1, so we need to adjust it to match h in simulation
                        data[galtype].append(loaded_data)
                    except IOError:
                        data[galtype].append(None)
        newdata = {}
        for item in data.keys():
            newdata[item] = np.array(data[item]).reshape((self.n_z_bins, self.nlambd_bins, -1, 3))
        return newdata

    def abundance_matching(self, richness, area):
        # do abundance matching for =richness
        sortedrichness = np.sort(richness)[::-1]
        newlambda_bins = []
        zbin_low = self.z_bins[:-1]
        zbin_high = self.z_bins[1:]
        with open(
            self.data_dir
            + "/clf/{0}/abundance_cluster_perdegsq_matching.pkl".format(
                self.compared_survey
            ),
            "rb",
        ) as f:
            sdssabundance = pickle.load(f)
        for i, zrange in enumerate(zip(zbin_low, zbin_high)):
            richness_temp = []
            for lambd in self.lambd_bins[i]:
                abundance_sdss = sdssabundance[
                    "z_{0:.2f}_{1:.2f}_lambdgt_{2:.10f}".format(
                        zrange[0], zrange[1], lambd
                    )
                ]
                richness_temp.append(sortedrichness[round(abundance_sdss * area)])
            newlambda_bins.append(richness_temp)
        self.old_lambd_bins = self.lambd_bins
        self.lambd_bins = newlambda_bins
        return

    def make_plot(self, clf, covar, data, name, save_to):
        # plot the result
        fig, ax = plt.subplots(
            self.nlambd_bins,
            self.n_z_bins,
            sharex=True,
            sharey=True,
            figsize=(12, 10),
            dpi=100,
        )
        if len(ax.shape) == 1:
            ax = ax.reshape(-1, 1)
        for i in range(self.n_z_bins):
            for j in range(self.nlambd_bins):
                ax_this = ax[j, i]
                for k, fmt in zip(("satellites", "centrals"), ("^", "o")):
                    ax_this.errorbar(
                        self.mag_center,
                        clf[k][i, j],
                        yerr=np.sqrt(np.diag(covar[k][i, j])),
                        label=k,
                        fmt=fmt,
                    )
                    newdata = data[k][i, j]
                    if newdata is not None:
                        ax_this.errorbar(
                            newdata[:, 0],
                            newdata[:, 1],
                            yerr=newdata[:, 2],
                            label=k + "_" + self.compared_survey,
                            fmt=fmt,
                        )
                ax_this.set_ylim(0.05, 50)
                if self.old_lambd_bins is not None:
                    bins = (
                        self.old_lambd_bins[i][j],
                        self.old_lambd_bins[i][j + 1],
                        self.z_bins[i],
                        self.z_bins[i + 1],
                        self.data_z_mins[i],
                        self.data_z_maxs[i],
                    )
                else:
                    bins = (
                        self.lambd_bins[i][j],
                        self.lambd_bins[i][j + 1],
                        self.z_bins[i],
                        self.z_bins[i + 1],
                        self.data_z_mins[i],
                        self.data_z_maxs[i],
                    )
                ax_this.text(
                    -25,
                    10,
                    (
                        r"${:.1E} \leq \lambda <{:.1E}$"
                        + "\n"
                        + r"${:g} \leq z<{:g}$"
                        + "\n"
                        + self.compared_survey
                        + r": ${:g} \leq z<{:g}$"
                    ).format(*bins),
                )
                ax_this.set_yscale("log")
        ax_this.legend(loc="lower right", frameon=False, fontsize="medium")

        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(
            labelcolor="none",
            top=False,
            bottom=False,
            left=False,
            right=False,
            direction="in",
        )
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.grid(False)
        ax.set_ylabel(
            r"$\phi(M_{{{}}}\,|\,\lambda,z)\quad[{{\rm Mag}}^{{-1}}]$".format(
                self.band
            ),
            labelpad=30,
        )
        ax.set_xlabel(r"$M_{{{}}}\quad[{{\rm Mag}}]$".format(self.band), labelpad=30)
        ax.set_title(name)

        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)

    def get_central_mag_id(
        self, cat_mem_match_id, mem_mem_match_id, ra_cluster, dec_cluster, ra, dec, mag
    ):
        # get the central galaxy's magnitude
        ncluster = len(cat_mem_match_id)
        ngals = len(mem_mem_match_id)
        cenmag = np.zeros([ncluster, 1])
        cengalindex = np.zeros([ncluster, 1]) - 1
        count_lo = 0
        count_hi = 0
        ncent = 1
        for i in range(ncluster):
            idList = []
            while cat_mem_match_id[i] != mem_mem_match_id[count_lo]:
                if cat_mem_match_id[i] > mem_mem_match_id[count_lo]:
                    idList.append(mem_mem_match_id[count_lo])
                count_lo = count_lo + 1
                count_hi = count_lo
            while cat_mem_match_id[i] == mem_mem_match_id[count_hi]:
                count_hi += 1
                if count_hi >= ngals:
                    break
            for j in range(ncent):
                if len(ra_cluster.shape) == 1:
                    place = np.where(
                        (np.abs(ra_cluster[i] - ra[count_lo:count_hi]) < 1e-5)
                        & (np.abs(dec_cluster[i] - dec[count_lo:count_hi]) < 1e-5)
                    )[0]
                if len(place) == 0:
                    print("WARNING:  Possible ID issue in get_central_mag")
                    cenmag[i][j] = 0
                    cengalindex[i][j] = -1
                    continue
                cenmag[i][j] = mag[count_lo + place[0]]
                cengalindex[i][j] = count_lo + place[0]

            count_lo = count_hi
        return cenmag, cengalindex

    def make_jack_samples_simple(self, RA, DEC):
        # do jackknife
        cat = treecorr.Catalog(ra=RA.flatten(), dec=DEC.flatten(), ra_units='deg', dec_units='deg')
        labels, _ = cat.getNField().run_kmeans(self.njack)
        uniquelabel = np.unique(labels)
        jacklist = np.empty(self.njack, dtype=np.object)
        for i in range(self.njack):
            jacklist[i] = np.where(labels != uniquelabel[i])[0]
        return jacklist

    def getjackgal(self, jackclusterList, c_mem_id, g_mem_id, match_index=None):
        # get galaxy mask for jackknife
        nclusters = len(c_mem_id)
        ngals = len(g_mem_id)

        if match_index is None:
            ulist = np.zeros(nclusters).astype(int)
            place = 0
            for i in range(ngals):
                if place == 0:
                    ulist[place] = 0
                    place = place + 1
                    continue
                if g_mem_id[i] == g_mem_id[i - 1]:
                    continue
                ulist[place] = i
                place = place + 1
                if place >= nclusters:
                    break
            match_index = np.zeros(np.max(c_mem_id) + 1).astype(int) - 1
            match_index[c_mem_id] = ulist
            return match_index
        max_id = np.max(c_mem_id)
        galpos = np.arange(len(g_mem_id))
        gboot_single = []
        for index in jackclusterList:
            mygal = match_index[c_mem_id[index]]
            if c_mem_id[index] == max_id:
                endgal = len(g_mem_id)
            else:
                endgal = match_index[c_mem_id[index + 1]]
            if endgal < mygal:
                print("Something has gone wrong, ")
            glist = galpos[mygal:endgal]
            if glist.size != 0:
                gboot_single.extend(glist)
            else:
                gboot_single.extend([])

        return gboot_single

    def make_single_clf(
        self, lm, z, lumbins, count_arr, lm_min, lm_max, zmin, zmax, limmag=None
    ):
        # calculate clf for single bins
        dlum = lumbins[1] - lumbins[0]
        clf = np.zeros_like(lumbins).astype(int)

        clist = np.where((z >= zmin) & (z < zmax) & (lm >= lm_min) & (lm < lm_max))[0]

        if clist.size == 0:
            print(
                "WARNING: no clusters found for limits of: {0} {1} {2} {3}".format(
                    lm_min, lm_max, zmin, zmax
                )
            )
            return clf

        [nclusters_lum, binlist] = cluster_Lcount(lumbins, limmag[clist])
        for i, c in enumerate(clist):
            clf[: binlist[i]] = clf[: binlist[i]] + count_arr[c, : binlist[i]]

        clf = clf / nclusters_lum / dlum

        return clf

    def redm_clf(
        self,
        Mag,
        cenMag,
        pcen_all,
        jacklist,
        cluster_id,
        cluster_id_member,
        match_index,
        limmag,
        scaleval,
        pmem,
        pcen,
        cluster_lm,
        cluster_z,
    ):
        # Main method of calculating clf
        lumbins = self.magnitude_bins[1:]
        nlum = len(lumbins)
        zmin = self.z_bins[:-1]
        zmax = self.z_bins[1:]
        lm_min = np.array([lambdbins[:-1] for lambdbins in self.lambd_bins])
        lm_max = np.array([lambdbins[1:] for lambdbins in self.lambd_bins])

        nlambda = len(lm_min[0])
        nz = len(zmin)
        njack = len(jacklist)
        cenclf = np.zeros([nz, nlambda, nlum])
        satclf = np.zeros([nz, nlambda, nlum])

        sat_count_arr = count_galaxies_p(
            cluster_id, scaleval, cluster_id_member, pmem * (1 - pcen_all), Mag, lumbins
        )
        cen_count_arr = count_galaxies_p_cen(cenMag, lumbins, pcen)

        for i in range(nz):
            for j in range(nlambda):
                cenclf[i, j] = self.make_single_clf(
                    cluster_lm,
                    cluster_z,
                    lumbins,
                    cen_count_arr,
                    lm_min[i, j],
                    lm_max[i, j],
                    zmin[i],
                    zmax[i],
                    limmag=limmag,
                )
                satclf[i, j] = self.make_single_clf(
                    cluster_lm,
                    cluster_z,
                    lumbins,
                    sat_count_arr,
                    lm_min[i, j],
                    lm_max[i, j],
                    zmin[i],
                    zmax[i],
                    limmag=limmag,
                )
        njack = len(jacklist)
        cenclf_jack = np.zeros([njack, nz, nlambda, nlum])
        satclf_jack = np.zeros([njack, nz, nlambda, nlum])

        for i, jack in enumerate(jacklist):
            gjack = self.getjackgal(jack, cluster_id, cluster_id_member, match_index)
            sat_count_arr_b = count_galaxies_p(
                cluster_id[jack],
                scaleval[jack],
                cluster_id_member[gjack],
                (pmem * (1 - pcen_all))[gjack],
                Mag[gjack],
                lumbins,
            )
            cen_count_arr_b = count_galaxies_p_cen(cenMag[jack], lumbins, pcen[jack])
            my_limmag = limmag[jack]
            for j in range(nz):
                for k in range(nlambda):
                    cenclf_jack[i, j, k, :] = self.make_single_clf(
                        cluster_lm[jack],
                        cluster_z[jack],
                        lumbins,
                        cen_count_arr_b,
                        lm_min[j, k],
                        lm_max[j, k],
                        zmin[j],
                        zmax[j],
                        limmag=my_limmag,
                    )
                    satclf_jack[i, j, k, :] = self.make_single_clf(
                        cluster_lm[jack],
                        cluster_z[jack],
                        lumbins,
                        sat_count_arr_b,
                        lm_min[j, k],
                        lm_max[j, k],
                        zmin[j],
                        zmax[j],
                        limmag=my_limmag,
                    )
        covar_cen = np.zeros([nz, nlambda, nlum, nlum])
        covar_sat = np.zeros([nz, nlambda, nlum, nlum])
        meanjack_cen = np.zeros([nz, nlambda, nlum])
        meanjack_sat = np.zeros([nz, nlambda, nlum])
        for i in range(nz):
            for j in range(nlambda):
                for k in range(nlum):
                    for l in range(nlum):
                        mean_jack_cen_k = np.mean(cenclf_jack[:, i, j, k])
                        mean_jack_cen_l = np.mean(cenclf_jack[:, i, j, l])
                        mean_jack_sat_k = np.mean(satclf_jack[:, i, j, k])
                        mean_jack_sat_l = np.mean(satclf_jack[:, i, j, l])
                        meanjack_cen[i, j, k] = mean_jack_cen_k
                        meanjack_sat[i, j, k] = mean_jack_sat_k
                        covar_cen[i, j, k, l] = (
                            np.sum(
                                (cenclf_jack[:, i, j, k] - mean_jack_cen_k)
                                * (cenclf_jack[:, i, j, l] - mean_jack_cen_l)
                            )
                            / (njack)
                            * (njack - 1)
                        )
                        covar_sat[i, j, k, l] = (
                            np.sum(
                                (satclf_jack[:, i, j, k] - mean_jack_sat_k)
                                * (satclf_jack[:, i, j, l] - mean_jack_sat_l)
                            )
                            / (njack)
                            * (njack - 1)
                        )
        return cenclf, satclf, covar_cen, covar_sat
