from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import kcorrect
from astropy.cosmology import FlatLambdaCDM

# def load_DEEP2(filename, colors, zlo, zhi, limiting_band, limiting_mag):
#     """
#     Compute the CDF of DEEP2 colors for some redshift range. 

#     Parameters
#     ----------
#         colors : list of string, required
#             list of colors to be tested
#             e.g ['u-g','g-r','r-i','i-z']

#         zlo : float, requred
#             minimum redshift of the validation catalog
        
#         zhi : float, requred
#             maximum redshift of the validation catalog
#     """


#     # MAG_APERCOR or MAG_AUTO
#     mag_apercor_q = True

#     cat = fits.getdata(filename)

#     mask = (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)
#     mask = (cat['zquality']>=3) & (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)

#     # Both CFHTLS Deep and Wide
#     mask = (cat['zquality']>=3) & (cat['cfhtls_source']>=0) & (cat['r_radius_arcsec']!=99)
#     # CFHTLS Deep only
#     # mask = (cat['zquality']>=3) & (cat['cfhtls_source']==0) & (cat['r_radius_arcsec']!=99)
#     cat = cat[mask]

#     if mag_apercor_q:
#         translate = {'u':'u_apercor', 'g':'g_apercor', 'r':'r_apercor', 'i':'i_apercor', 'z':'z_apercor'}
#     else:
#         translate = {'u':'u', 'g':'g', 'r':'r', 'i':'i', 'z':'z'}

#     limiting_band_name = translate[limiting_band]
#     mask_all = (cat['zhelio']>zlo) & (cat['zhelio']<zhi) & (cat[limiting_band_name]<limiting_mag)

#     vsummary = []
#     # PDF with small bins for calculating CDF
#     for index in range(len(colors)):
#         color = colors[index]
#         band1 = translate[color[0]]
#         band2 = translate[color[2]]

#         mask = mask_all & (np.abs(cat[band1])>0) & (np.abs(cat[band1])<50) & (np.abs(cat[band2])>0) & (np.abs(cat[band2])<50)
#         bins = np.linspace(-1, 4, 2000)
#         hist, bin_edges = np.histogram((cat[band1]-cat[band2])[mask], bins=bins)
#         hist = hist/np.sum(hist)
#         binctr = (bin_edges[1:] + bin_edges[:-1])/2.

#         vsummary.append((binctr, hist))

#     return vsummary
                                   
def load_SDSS(filename, colors, SDSS_kcorrection_z):
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
    
    translate = {'u':'M_u', 'g':'M_g', 'r':'M_r', 'i':'M_i', 'z':'M_z'}
    
    # limiting_band_name = translate[limiting_band]
    # mask_all = (cat['z']>zlo) & (cat['z']<zhi) & (cat[limiting_band_name]<limiting_mag)

    data_dir = os.path.dirname(filename)
    kcorrect_magnitudes_path = os.path.join(data_dir, 'sdss_k_corrected_magnitudes_z_0.06_0.09_z_{:.3f}.fits'.format(SDSS_kcorrection_z))

    if not os.path.exists(kcorrect_magnitudes_path):

        kcorrect_maggies_path = os.path.join(data_dir, 'sdss_k_corrected_maggies_z_0.06_0.09_z_{:.3f}.dat'.format(SDSS_kcorrection_z))

        # Load kcorrect templates and filters
        kcorrect.load_templates()
        kcorrect.load_filters()

        kcorrect.reconstruct_maggies_from_file(filename, redshift=SDSS_kcorrection_z, outfile=kcorrect_maggies_path)

        #----------Convert kcorrected maggies to magnitudes----------------
        cat = Table.read(os.path.join(data_dir, kcorrect_maggies_path), format='ascii.no_header', names=('redshift', 'maggies_u', 'maggies_g', 'maggies_r', 'maggies_i', 'maggies_z'))

        cat0 = Table.read(filename, format='ascii.no_header')

        redshifts = cat0['col1']
        u = -2.5*np.log10(cat['maggies_u'])
        g = -2.5*np.log10(cat['maggies_g'])
        r = -2.5*np.log10(cat['maggies_r'])
        i = -2.5*np.log10(cat['maggies_i'])
        z = -2.5*np.log10(cat['maggies_z'])

        cat1 = Table()
        cat1['redshift'] = redshifts
        cat1['u'] = u
        cat1['g'] = g
        cat1['r'] = r
        cat1['i'] = i
        cat1['z'] = z
        cat1.write(kcorrect_magnitudes_path)
        cat = cat1.copy()
    else:
        cat = Table.read(kcorrect_magnitudes_path)

    # distance modulus
    ##########################################
    cosmo = FlatLambdaCDM(H0=70.2, Om0=0.275)
    ##########################################
    dm = np.array(cosmo.distmod(cat['redshift']))
    cat['M_u'] = cat['u'] - dm
    cat['M_g'] = cat['g'] - dm
    cat['M_r'] = cat['r'] - dm
    cat['M_i'] = cat['i'] - dm
    cat['M_z'] = cat['z'] - dm

    # Calculate the aboluste magnitude cut
    mask = (cat['redshift']>0.089) & (cat['redshift']<0.090)
    mr = cat['M_r'][mask]
    mr_sort = np.sort(mr)
    mrmax = mr_sort[int(len(mr)*0.85)]

    # Apply r-band absolute magnitude
    mask = (cat['M_r'] < mrmax)
    cat = cat[mask]

    vsummary = []

    # PDF with small bins for calculating CDF
    for index in range(len(colors)):
        color = colors[index]
        band1 = translate[color[0]]
        band2 = translate[color[2]]

        bins = np.linspace(-1, 4, 2000)
        hist, bin_edges = np.histogram((cat[band1]-cat[band2]), bins=bins)
        hist = hist/np.sum(hist)
        binctr = (bin_edges[1:] + bin_edges[:-1])/2.

        vsummary.append((len(cat), binctr, hist))

    return vsummary
