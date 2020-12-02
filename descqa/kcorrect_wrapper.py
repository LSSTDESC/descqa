import numpy as np

try:
    import kcorrect as kc
except ImportError
    import warnings
    warnings.warn("Cannot import kcorrect!")
    kc = None

__all__ = ["kcorrect"]

def kcorrect(mags, magerrs, redshifts, bandshift, filters="des_filters.dat"):
    try:
        kc.load_templates()
        kc.load_filters(f=filters)
    except AttributeError:
        raise RuntimeError("No kcorrect available")
    maggies = 10.**(-0.4*mags)
    maggies_ivars = 1./(maggies*0.4*np.log(10)*magerrs)**2
    kcorrect_arr = np.zeros_like(magerrs)
    for i, (maggie, z, maggies_ivar) in enumerate(zip(maggies, redshifts, maggies_ivars)):
        coffs = kc.fit_nonneg(z, maggie, maggies_ivar)
        #Reconstruct the magnitudes as observed and in the rest frame
        rmaggies = kc.reconstruct_maggies(coffs, redshift=z)
        reconstruct_maggies = kc.reconstruct_maggies(coffs, redshift=bandshift)
        reconstruct_maggies = reconstruct_maggies/(1.+bandshift)
        kcorrect_res_this = 2.5*np.log10(reconstruct_maggies/rmaggies)
        kcorrect_arr[i] = kcorrect_res_this[1:]
    return kcorrect_arr
