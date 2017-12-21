"""
utility functions for descqa
"""
from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
import healpy as hp


__all__ = [
    'get_sky_volume',
    'get_opt_binpoints',
    'get_healpixel_footprint',
    'generate_uniform_random_ra_dec',
    'generate_uniform_random_ra_dec_footprint',
]


def get_sky_volume(sky_area, zlo, zhi, cosmology):
    """
    Parameters
    ----------
    sky_area : float
        sky area in sq. deg.
    zlo : float
        lower redshift
    zhi : float
        upper redshift
    cosmology : astropy.Cosmology

    Returns
    -------
    sky_volume : float
        in unit of Mpc**3.0
    """
    dhi = cosmology.comoving_distance(zhi).to('Mpc').value if zhi > 0 else 0.0
    dlo = cosmology.comoving_distance(zlo).to('Mpc').value if zlo > 0 else 0.0
    sky_area_rad = np.deg2rad(np.deg2rad(sky_area))
    return (dhi**3.0 - dlo**3.0) * sky_area_rad / 3.0


def get_opt_binpoints(N, sumM, sumM2, bins):
    """
    compute optimal values at which to plot bin counts
    optimal point corresponds to location where function describing data
    equals value of N_i given by (Taylor expansion of integral of over bin)/bin-width:
       f(c_i) + offset*fprime(c_i) = f(c_i) + bin-width**2*fdblprime(c_i)/24
    uses N (counts per bin)
         sumM (first moment of points per bin)
         sumM2 (second moment of points per bin)
    """
    centers = (bins[1:]+bins[:-1])/2
    Delta = -bins[:-1]+bins[1:] #bin widths

    moment0 = N/Delta       #(integrals over bins)/Delta = f(c_i) + Delta**2*fdblprime(c_i)/24
    moment1 = N*(sumM/N - centers)/Delta  #(first moments about bin centers)/Delta = Delta**2*fprime(c_i)/12
    moment2 = N*(sumM2/N - 2*centers*sumM/N + centers**2)/Delta  #(second moments about bin centers)/Delta = Delta**2(f(c_i)/12 + *2*fdblprime(c_i)/160)
    fprime = 12.*moment1/Delta**2  #first derivative of function at bin center
    fdblprime = 360*(moment2 - moment0*Delta**2/12)/Delta**4 #second derivative of function at bin center
    offset = Delta**2*fdblprime/fprime/24  # offset*fprime(c_i) = Delta**2*fdblprime(c_i)/24
    return centers + offset


def get_healpixel_footprint(ra, dec, nside, nest=False, count_threshold=None):
    """
    Parameters
    ----------
    ra : ndarray
        RA in degrees
    dec : ndarray
        Dec in degrees
    nside : int
        number of healpixel nside, must be 2**k
    nest : bool, optional
        using healpixel nest or ring ordering
    count_threshold : None or int (optional)
        minimal number of points within a healpixel to count as part of the footprint

    Returns
    -------
    pixels : ndarray
        1d array that contains healpixel IDs
    """
    pixels = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True)
    if count_threshold and count_threshold > 1:
        pixels, counts = np.unique(pixels, return_counts=True)
        return pixels[counts >= count_threshold]
    return np.unique(pixels)


def generate_uniform_random_ra_dec(n):
    """
    Parameters
    ----------
    n : int
        number of random points needed

    Returns
    -------
    ra : ndarray
        1d array of length n that contains RA in degrees
    dec : ndarray
        1d array of length n that contains Dec in degrees
    """
    # ra = 360 * (U - 0.5)
    ra = np.random.rand(n)
    ra -= 0.5
    ra *= 360.0
    # dec = arccos(2*U - 1) * (180/pi) - 90
    dec = np.random.rand(n)
    dec -= 0.5
    dec *= 2.0
    dec = np.arccos(dec, out=dec)
    dec = np.rad2deg(dec, out=dec)
    dec -= 90.0
    return ra, dec


def generate_uniform_random_ra_dec_footprint(n, footprint=None, nside=None, nest=False, max_chunk=100000):
    """
    Parameters
    ----------
    n : int
        number of random points needed
    footprint : 1d array, optional
        unique healpixel IDs
    nside : int, optional
        number of healpixel nside as used in footprint, must be 2**k
    nest : bool, optional
        using healpixel nest or ring ordering
    max_chunk : int, optional
        maximal number of random to generate in each iteration

    Returns
    -------
    ra : ndarray
        1d array of length n that contains RA in degrees
    dec : ndarray
        1d array of length n that contains Dec in degrees
    """
    if footprint is not None:
        assert hp.isnsideok(nside), '`healpixel_nside` is not valid'
        scale = hp.nside2npix(nside) / len(footprint)

    if footprint is None or scale == 1.0:
        return generate_uniform_random_ra_dec(n)

    ra = np.empty(n)
    dec = np.empty_like(ra)
    n_needed = n

    while n_needed > 0:
        n_create = int(min((n_needed + np.ceil(np.sqrt(n_needed)*3.0))*scale, max_chunk))
        ra_this, dec_this = generate_uniform_random_ra_dec(n_create)
        mask = np.where(np.in1d(hp.ang2pix(nside, ra_this, dec_this, nest=nest, lonlat=True), footprint))[0]
        count_this = mask.size
        if n_needed - count_this < 0:
            count_this = n_needed
            mask = mask[:n_needed]

        s = slice(-n_needed, -n_needed+count_this if -n_needed+count_this < 0 else None)
        ra[s] = ra_this[mask]
        dec[s] = dec_this[mask]
        n_needed -= count_this

    return ra, dec


def generate_uniform_random_dist(n, dlo, dhi):
    """
    Parameters
    ----------
    n : int
        number of random points needed
    dlo : float
        lower distance
    dhi : float
        upper distance

    Returns
    -------
    dist : ndarray
        1d array of length n that contains distance
    """
    d = np.random.rand(n)
    d *= (dhi**3.0 - dlo**3.0)
    d += dlo**3.0
    d **= 1.0/3.0
    return d
