import numpy as np

class GalaxyCatalog():
    """
    Makes cat object
    """
    ##KWARG KEYS
    zlo='zlo'
    zhi='zhi'
    stellar_mass='stellar_mass'

    def __init__(self):
        self.galaxycatalog={}

class ANLGalaxyCatalog(GalaxyCatalog):
    #ANL Galaxy Catalog

    #VALUE ADDED DICT KEYS
    #By default nodeIndex written out for every galaxy
    redshift='redshift'
    ra='ra'
    dec='dec'
    v_pec='v_pec'
    disk_='disk_'
    bulge_='bulge_'
    agn_='agn_'
    mass='mass'
    age='age'
    stellar='stellar'
    gas='gas'
    metallicity='metallicity'
    log_='log_'
    sfr='sfr'
    ellipticity='ellipticity'
    stellarmass=stellar+mass
    log_stellarmass=log_+stellar+mass
    gasmass=gas+mass
    positionX='positionX'
    positionY='positionY'
    positionZ='positionZ'
    velocityX='velocityX'
    velocityY='velocityY'
    velocityZ='velocityZ'
    disk_ra=disk_+ra
    disk_dec=disk_+dec
    disk_sigma0=disk_+'sigma0'
    disk_re=disk_+'re'
    disk_index=disk_+'index'
    disk_a=disk_+'a'
    disk_b=disk_+'b'
    disk_theta_los=disk_+'theta_los'
    disk_phi=disk_+'phi'
    log_disk_stellarmass=log_+disk_+stellarmass
    disk_metallicity=disk_+metallicity
    disk_age=disk_+age
    disk_sfr=disk_+sfr
    disk_ellipticity=disk_+ellipticity
    bulge_ra=bulge_+ra
    bulge_dec=bulge_+dec
    bulge_sigma0=bulge_+'sigma0'
    bulge_re=bulge_+'re'
    bulge_index=bulge_+'index'
    bulge_a=bulge_+'a'
    bulge_b=bulge_+'b'
    bulge_theta_los=bulge_+'theta_los'
    bulge_phi=bulge_+'phi'
    log_bulge_stellarmass=log_+bulge_+stellarmass
    bulge_age=bulge_+age
    bulge_sfr=bulge_+sfr
    bulge_metallicity=bulge_+metallicity
    bulge_ellipticity=bulge_+ellipticity
    agn_ra=agn_+ra
    agn_dec=agn_+dec
    agn_mass=agn_+mass
    agn_accretnrate=agn_+'accretnrate'
    rest=':rest:'
    observed=':observed:'
    SDSS_u='SDSS_u'
    SDSS_g='SDSS_g'
    SDSS_r='SDSS_r'
    SDSS_i='SDSS_i'
    SDSS_z='SDSS_z'
    SDSS=[SDSS_u,SDSS_g,SDSS_r,SDSS_i,SDSS_z]
    SDSS_R=[f+rest for f in SDSS]
    SDSS_O=[f+observed for f in SDSS]
    SDSS_RO=SDSS_R+SDSS_O
    DES_g='DES_g'
    DES_r='DES_r'
    DES_i='DES_i'
    DES_z='DES_z'
    DES_Y='DES_Y'
    DES=[DES_g,DES_r,DES_i,DES_z,DES_Y]
    DES_R=[f+rest for f in DES]
    DES_O=[f+observed for f in DES]
    DES_RO=DES_R+DES_O
    LSST_u='LSST_u'
    LSST_g='LSST_g'
    LSST_r='LSST_r'
    LSST_i='LSST_i'
    LSST_z='LSST_z'
    LSST_y4='LSST_y4'
    LSST=[LSST_u,LSST_g,LSST_r,LSST_i,LSST_z,LSST_y4]
    LSST_R=[f+rest for f in LSST]
    LSST_O=[f+observed for f in LSST]
    LSST_RO=LSST_R+LSST_O
    B='B'
    U='U'
    V='V'
    CFHT_g='CFHTL_g'
    CFHT_r='CFHTL_r'
    CFHT_i='CFHTL_i'
    CFHT_z='CFHTL_z'
    CFHT=[CFHT_g,CFHT_r,CFHT_i,CFHT_z]
    CFHT_R=[f+rest for f in CFHT]
    CFHT_O=[f+observed for f in CFHT]
    CFHT_RO=CFHT_R+CFHT_O

    #ANL subclass
    def __init__(self,mockcat):
        if (type(mockcat)==dict):
            self.galaxycatalog=mockcat
        else:
            self.galaxycatalog={}
        #endif

    def get_quantities(self,ids,*args,**kwargs):
        if(type(ids)==str):
            ids=[ids]
        elif not(type(ids)==list):
            print "Wrong format"
            return
        #endif
        data=[]
        for id in ids:
            print "Requested data for",id
            if(id==self.stellar_mass):
                zlo=None
                zhi=None
                #parse kwargs
                if any(kwargs):
                    if(self.zlo in kwargs.keys()):
                        zlo=kwargs[self.zlo]
                    if(self.zhi in kwargs.keys()):
                        zhi=kwargs[self.zhi]
                    print "Using zlo=",zlo,"zhi=",zhi
                    if(zlo is not None and zhi is not None):
                        sm=self.get_stellarmasses(zlo,zhi)
                        data.append(sm)
                    else:
                        print "Must supply",self.zlo,self.zhi
                else:
                    print "Must supply",self.zlo,self.zhi
                #endif
            else:
                print 'unknown option'
            #endif

        #endfor
        return data

    def get_stellarmasses(self,zlo,zhi):
        nout=0
        sm=np.asarray([])
        for key in self.galaxycatalog.keys():
            minz=min(self.galaxycatalog[key][self.redshift])
            maxz=max(self.galaxycatalog[key][self.redshift])
            if(minz<zhi and maxz>zlo):
                print 'Adding',key,'data'
                if (self.galaxycatalog[key].has_key(self.stellarmass)):
                    sm_x=self.galaxycatalog[key][self.stellarmass]
                elif (self.galaxycatalog[key].has_key(self.log_stellarmass)):
                    log_sm_x=self.galaxycatalog[key][self.log_stellarmass]
                    sm_x=np.power(10,log_sm_x)
                else:
                    print "Data for",self.stellarmass,"or",self.log_stellarmass,"NOT available"
                if (nout>0):
                    z=np.concatenate((sm,sm_x),axis=0)
                else:
                    sm=sm_x
                #endif
                nout=nout+1
            #endif

        #endfor
        return sm

