# Test GCRCatalogs vs. direct DataFrame

import h5py
import healpy as hp
import GCRCatalogs

catalog_name = "cosmoDC2_v1.1.4_small"
instance = GCRCatalogs.load_catalog(catalog_name)

files = instance._get_file_list()

ra_col = "ra_true"
dec_col = "dec_true"


def calc_healpix_set_from_iterator(
    data_iterator,
    nside,
    ra_col,
    dec_col
):
    """

    Parameters
    ----------
    data_iterator: Iterator
        That yields objects that can produce RA, Dec with
        data_iterator[ra_col], data_iterator[dec_col]
    nside: int
        HealPix NSIDE
    ra_col: str
        Column name of RA data in data_iterator
    dec_col: str
        Column name of Dec data in data_iterator

    Calculating the healpixel for all of the data is the I/O intensive step.
    so we separate out this here into its own function.
    """
    pixels = set()
    for d in data_iterator:
        pixels.update(hp.ang2pix(nside, d[ra_col], d[dec_col], lonlat=True))

    return pixels


def raw_iterator(files, quantities):
    groups = ["galaxyProperties"]
    for file in files:
        fh = h5py.File(file, 'r')
        for group in groups:
            df = {}
            for quantity in quantities
                df[quantity] = fh["{}/{}".format(group, quantity)]

            yield df
            fh.close()


raw_data_iterator = raw_iterator(files, quantities=[ra_col, dec_col])
gcr_data_iterator = instance.get_quantities([ra_col, dec_col], return_iterator=True)

pixels_raw = calc_healpix_set_from_iterator(raw_data_iterator)
pixels_gcr = calc_healpix_set_from_iterator(gcr_data_iterator)
