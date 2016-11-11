import numpy as np

def chiofz(zval=0.45,omm=0.31):
  """
  comoving distance to redshift zval
  omega matter = omm, universe assumed flat
  use for volume if specify region with ra/dec
  """
  Nint = 300000
  zp1int = np.linspace(1,zval+1,Nint)
  ez = np.sqrt(omm*zp1int*zp1int*zp1int + (1-omm))
  tmp = 2997.925* np.trapz(1/ez, dx = zp1int[1]-zp1int[0])
  return(tmp)





def getvol(boxside,ramin,ramax,decmin,decmax,zmin,zmax,omm=0.31):
    """
    boxside > 0:
    volume for periodic box of boxside Mpc/h in each direction
    boxside < 0:
    volume for region given by ra  between ramin and ramax
                               dec between decmin and decmax
                               z   between zmin and zmax
                        
    in both cases resulting volume is in (Mpc/h)^3                           
    """
    if (boxside < 0):
       print "using light cone"
       chimax = chiofz(zmax,omm)  #[Mpc/h]
       chimin = chiofz(zmin,omm)  #[Mpc/h]
       angvol = -(np.cos((90-decmin)*np.pi/180) - np.cos((90-decmax)*np.pi/180))*(np.pi*(ramax-ramin)/180.)
       chivol =(chimax*chimax*chimax - chimin*chimin*chimin)/3.
       vol = chivol*angvol  # in [Mpc/h]^3
   if (boxside>0):
       print "using periodic box, side %8.2f Mpc/h"%(boxside)
       vol = boxside*boxside*boxside
   return(vol)
