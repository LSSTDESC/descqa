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
    'first',
    'is_string_like',
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

def get_sky_area(catalog_instance, nside=1024):
    """
    Parameters
    ----------
    catalog_instance: GCRCatalogs intance
    nside: nside parameter for healpy
    Returns
    -------
    sky_area : float
        in units of deg**2.0
    """
    possible_area_qs = (('ra_true', 'ra'), ('dec_true', 'dec'))
    area_qs = [catalog_instance.first_available(*a) for a in possible_area_qs]

    pixels = set()
    for d in catalog_instance.get_quantities(area_qs, return_iterator=True):
        pixels.update(hp.ang2pix(nside, d[area_qs[0]], d[area_qs[1]], lonlat=True))
            
    frac = len(pixels) / hp.nside2npix(nside)
    sky_area = frac * np.rad2deg(np.rad2deg(4.0*np.pi))

    return sky_area
    
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


def generate_uniform_random_ra_dec_min_max(n, ra_min, ra_max, dec_min, dec_max):
    """
    Parameters
    ----------
    n : int
        number of random points needed
    ra_min, ra_max, dec_min, dec_max: float
        min and max of ra and dec

    Returns
    -------
    ra : ndarray
        1d array of length n that contains RA in degrees
    dec : ndarray
        1d array of length n that contains Dec in degrees
    """
    ra = np.random.uniform(ra_min, ra_max, size=n)
    dec = np.random.uniform(np.sin(np.deg2rad(dec_min)), np.sin(np.deg2rad(dec_max)), size=n)
    dec = np.arcsin(dec, out=dec)
    dec = np.rad2deg(dec, out=dec)
    return ra, dec


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
    return generate_uniform_random_ra_dec_min_max(n, 0, 360.0, -90.0, 90.0)


def generate_uniform_random_ra_dec_healpixel(n, pix, nside, nest=False):
    """
    Parameters
    ----------
    n : int
        number of random points needed
    pix : int
        healpixel ID
    nside : int
        number of healpixel nside, must be 2**k
    nest : bool, optional
        using healpixel nest or ring ordering

    Returns
    -------
    ra : ndarray
        1d array of length n that contains RA in degrees
    dec : ndarray
        1d array of length n that contains Dec in degrees
    """

    ra, dec = hp.vec2ang(hp.boundaries(nside, pix, 1, nest=nest).T, lonlat=True)
    ra_dec_min_max = ra.min(), ra.max(), dec.min(), dec.max()

    ra = np.empty(n)
    dec = np.empty_like(ra)
    n_needed = n

    while n_needed > 0:
        ra_this, dec_this = generate_uniform_random_ra_dec_min_max(n_needed*2, *ra_dec_min_max)
        mask = np.where(hp.ang2pix(nside, ra_this, dec_this, nest=nest, lonlat=True) == pix)[0]
        count_this = mask.size
        if n_needed - count_this < 0:
            count_this = n_needed
            mask = mask[:n_needed]

        s = slice(-n_needed, -n_needed+count_this if -n_needed+count_this < 0 else None)
        ra[s] = ra_this[mask]
        dec[s] = dec_this[mask]
        n_needed -= count_this

    return ra, dec


def generate_uniform_random_ra_dec_footprint(n, footprint=None, nside=None, nest=False):
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

    Returns
    -------
    ra : ndarray
        1d array of length n that contains RA in degrees
    dec : ndarray
        1d array of length n that contains Dec in degrees
    """
    if footprint is None or hp.nside2npix(nside) == len(footprint):
        return generate_uniform_random_ra_dec(n)

    n_per_pix_all = np.histogram(np.random.rand(n), np.linspace(0, 1, len(footprint)+1))[0]

    ra = np.empty(n)
    dec = np.empty_like(ra)
    count = 0

    for n_per_pix, pix in zip(n_per_pix_all, footprint):
        ra_this, dec_this = generate_uniform_random_ra_dec_healpixel(n_per_pix, pix, nside, nest)
        s = slice(count, count+n_per_pix)
        ra[s] = ra_this
        dec[s] = dec_this
        count += n_per_pix

    assert count == n

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


def first(iterable, default=None):
    """
    returns the first element of `iterable`
    """
    return next(iter(iterable), default)


def is_string_like(obj):
    """
    test if `obj` is string like
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True
