import kcorrect as kc
import numpy as np
from tqdm import tqdm
def kcorrect(mags, magerrs, redshifts, bandshift, filters="des_filters.dat"):
   kc.load_templates()
   kc.load_filters(f=filters)
   maggies=10.**(-0.4*mags)
   maggies_ivar= 1./(maggies*0.4*np.log(10)*magerrs)**2 
   kcorrect_arr = np.zeros_like(magerrs)
   pbar = tqdm(total=len(mags))
   for i, mag in enumerate(mags):
      if i%100 == 0:
         pbar.update(100)
      magivar = 1./magerrs[i]**2. #inverse variance of maggies
      z = redshifts[i]
      coffs = kc.fit_nonneg(z, maggies[i], maggies_ivar[i])
      #Reconstruct the magnitudes as observed and in the rest frame
      rmaggies = kc.reconstruct_maggies(coffs, redshift=z)
      reconstruct_maggies = kc.reconstruct_maggies(coffs, redshift=bandshift)
      reconstruct_maggies = reconstruct_maggies/(1.+bandshift)
      kcorrect=reconstruct_maggies/rmaggies
      kcorrect=2.5*np.log10(kcorrect)
      kcorrect_arr[i] = kcorrect[1:]
   pbar.close()
   return kcorrect_arr

