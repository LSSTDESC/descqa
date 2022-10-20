from __future__ import division
import numpy as np
from scipy.stats import chi2
import healpy as hp
from descqa.utils import *


def check_ra_dec_basic(ra, dec, n):
    assert ra.size == n
    assert dec.size == n
    assert (ra <= 360.0).all()
    assert (ra >= 0.0).all()
    assert (dec <= 90.0).all()
    assert (dec >= -90.0).all()


def check_ra_dec_uniform(ra, dec, nside=2, footprint=None):
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True)
    npix = hp.nside2npix(nside) if footprint is None else footprint.size
    pixels, counts = np.unique(pixels, return_counts=True)
    assert pixels.size == npix
    mean = ra.size / npix
    assert chi2.sf(((counts - mean)**2.0 / mean).sum(), df=npix-1) > 1e-5


def test_generate_uniform_random_ra_dec_basic():
    n = 10000
    ra, dec = generate_uniform_random_ra_dec(n)
    check_ra_dec_basic(ra, dec, n)
    check_ra_dec_uniform(ra, dec)


def test_generate_uniform_random_ra_dec_footprint():
    n = 10000
    nside = 2
    npix = hp.nside2npix(nside)
    footprint = np.arange(npix)[np.random.randint(2, size=npix).astype(bool)]

    ra, dec = generate_uniform_random_ra_dec_footprint(n, footprint, nside)
    check_ra_dec_basic(ra, dec, n)
    check_ra_dec_uniform(ra, dec, nside, footprint)

    assert (get_healpixel_footprint(ra, dec, nside) == footprint).all()
