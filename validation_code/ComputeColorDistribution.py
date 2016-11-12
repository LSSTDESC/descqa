from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def load_DEEP2(colors, zlo, zhi):
    """
    Compute the CDF of DEEP2 colors for some redshift range. 

    Parameters
    ----------
        colors : list of string, required
            list of colors to be tested
            e.g ['u-g','g-r','r-i','i-z']

        zlo : float, requred
            minimum redshift of the validation catalog
        
        zhi : float, requred
            maximum redshift of the validation catalog
    """


    # MAG_APERCOR or MAG_AUTO
    mag_apercor_q = True

    cat = fits.getdata('/project/projectdirs/lsst/rongpu/descqa/DEEP2_uniq_Terapix_Subaru.fits')

    mask = (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)
    mask = (cat['zquality']>=3) & (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)

    # Both CFHTLS Deep and Wide
    mask = (cat['zquality']>=3) & (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)
    # CFHTLS Deep only
    # mask = (cat['zquality']>=3) & (cat['cfhtls_source']==0) & (cat['r_radius_arcsec']!=99)
    cat = cat[mask]

    if mag_apercor_q:
        translate = {'u':'u_apercor', 'g':'g_apercor', 'r':'r_apercor', 'i':'i_apercor', 'z':'z_apercor'}
    else:
        translate = {'u':'u', 'g':'g', 'r':'r', 'i':'i', 'z':'z'}
        
    mask_redshift = (cat['zhelio']>zlo) & (cat['zhelio']<zhi)

    vsummary = []
    # PDF with small bins for calculating CDF
    for index in range(len(colors)):
        color = colors[index]
        band1 = translate[color[0]]
        band2 = translate[color[2]]

        mask = mask_redshift & (np.abs(cat[band1])>0) & (np.abs(cat[band1])<50) & (np.abs(cat[band2])>0) & (np.abs(cat[band2])<50)
        bins = np.linspace(-1, 4, 2000)
        hist, bin_edges = np.histogram((cat[band1]-cat[band2])[mask], bins=bins)
        hist = hist/np.sum(hist)
        binctr = (bin_edges[1:] + bin_edges[:-1])/2.

        vsummary.append((binctr, hist))

    return vsummary

def load_SDSS(colors, zlo, zhi):
    """
    Compute the CDF of SDSS colors for some redshift range. 

    Parameters
    ----------
        colors : list of string, required
            list of colors to be tested
            e.g ['u-g','g-r','r-i','i-z']

        zlo : float, requred
            minimum redshift of the validation catalog
        
        zhi : float, requred
            maximum redshift of the validation catalog
    """
    
    cat = fits.getdata('/project/projectdirs/lsst/rongpu/descqa/SpecPhoto_sdss_extinction_corrected_trimmed.fit')

    translate = {'u':'modelMag_u', 'g':'modelMag_g', 'r':'modelMag_r', 'i':'modelMag_i', 'z':'modelMag_z'}
        
    mask_redshift = (cat['z']>zlo) & (cat['z']<zhi)

    vsummary = []
    # PDF with small bins for calculating CDF
    for index in range(len(colors)):
        color = colors[index]
        band1 = translate[color[0]]
        band2 = translate[color[2]]

        mask = mask_redshift & (np.abs(cat[band1])>0) & (np.abs(cat[band1])<50) & (np.abs(cat[band2])>0) & (np.abs(cat[band2])<50)
        bins = np.linspace(-1, 4, 2000)
        hist, bin_edges = np.histogram((cat[band1]-cat[band2])[mask], bins=bins)
        hist = hist/np.sum(hist)
        binctr = (bin_edges[1:] + bin_edges[:-1])/2.

        vsummary.append((binctr, hist))

    return vsummary
