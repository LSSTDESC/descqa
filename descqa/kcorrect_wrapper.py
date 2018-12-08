import kcorrect as kc
import numpy as np
from tqdm import tqdm
def kcorrect(mags, magerrs, redshifts, bandshift, filters="des_filters.dat"):
    kc.load_templates()
    kc.load_filters(f=filters)
    maggies=10.**(-0.4*mags)
    maggies_ivars = 1./(maggies*0.4*np.log(10)*magerrs)**2 
    kcorrect_arr = np.zeros_like(magerrs)
    for i, ziparr in tqdm(enumerate(zip(maggies, redshifts, maggies_ivars))):
        maggie, z, maggies_ivar = ziparr
        coffs = kc.fit_nonneg(z, maggie, maggies_ivar)
        #Reconstruct the magnitudes as observed and in the rest frame
        rmaggies = kc.reconstruct_maggies(coffs, redshift=z)
        reconstruct_maggies = kc.reconstruct_maggies(coffs, redshift=bandshift)
        reconstruct_maggies = reconstruct_maggies/(1.+bandshift)
        kcorrect=reconstruct_maggies/rmaggies
        kcorrect=2.5*np.log10(kcorrect)
        kcorrect_arr[i] = kcorrect[1:]
    return kcorrect_arr

