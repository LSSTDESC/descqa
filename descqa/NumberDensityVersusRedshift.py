from __future__ import print_function, unicode_literals, absolute_import 
import os
import numpy as np
import re
import math
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['NumberDensityVersusRedshift']

#observations
Coil2004 = 'Coil2004'
Coil2004_magbin = 'Coil2004_magbin'
Coil2004_maglim = 'Coil2004_maglim'
DEEP2_JAN = 'DEEP2_JAN'
Coil2004_lbl = 'Coil et. al. 2004'
DEEP2_JAN_lbl = 'DEEP2 (JAN, p.c.)'

#setup dict with parameters needed to read in validation data 
data_info = {
    'Coil2004_magbin': {'filename_template': 'N_z/DEEP2/Coil_et_al_2004_Table3_{}.txt', 'usecols':[0,1,2,4], 'skiprows':2},
    'Coil2004_maglim': {'filename_template': 'N_z/DEEP2/Coil_et_al_2004_Table4_{}.txt', 'usecols':[0,1,2], 'skiprows':3},
    'DEEP2_JAN':{'filename_template': 'N_z/DEEP2/JANewman_{}.txt','usecols':[0,1,2,3], 'skiprows':1},
}

#plotting constants
figx_p = 9
figy_p = 11
lw2 = 2
fsize = 16
lsize = 10  
default_colors = ['blue','r','m','g','navy','y','purple','gray','c','orange','violet','coral','gold','orchid','maroon','tomato','sienna','chartreuse','firebrick','SteelBlue']
validation_color = 'black'

class NumberDensityVersusRedshift(BaseValidationTest):
    """
    validation test to show N(z) distributions
    """
    def __init__(self, z='redshift_true', band='i', N_zbins=44,zlo=0., zhi=1.1, observation='', m_lo=27, m_hi=18, ncolumns=2, normed=True, **kwargs):
        
        #catalog quantities
        self.z = z
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                              )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]
        self.band = band
        
        #z-bounds and binning
        self.zlo = zlo
        self.zhi = zhi
        self.N_zbins = N_zbins
        self.zbins = np.linspace(zlo,zhi,N_zbins+1)
        self.filters = [(lambda z: (z > zlo) & (z < zhi), self.z)]

        #validation data
        self.validation_data = {}
        possible_observations = data_info.keys()
        self.observation = observation
        #check for valid combinations
        if len(observation)==0:
            print ('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        else:
            #check that observation is known
            if not(observation in possible_observations):
                raise ValueError('Observation {} not available'.format(observation))
            else:
                #fetch validation data
                self.validation_data = self.get_validation_data(band, observation)
                
        #plotting variables
        self.normed = normed
        self.ncolumns = int(ncolumns)

        #setup subplot configuration and get magnitude cuts for each plot
        self.mag_lo,self.mag_hi = self.init_plots(m_lo,m_hi)
        
        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows,self.ncolumns,figsize=(figx_p,figy_p),sharex='col')
        self.first_pass = True    #only plot validation data once

        #other
        self._other_kwargs = kwargs

        return

    def init_plots(self, mlo, mhi):
        #get magnitude cuts based on validation data or default limits
        mag_lo = self.validation_data.get('mag_lo', [float(m) for m in range(mhi, mlo+1)])
        mag_hi = self.validation_data.get('mag_hi', [])

        #setup plots and determine number of rows required for subplots
        self.nplots = len(mag_lo)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns

        #colors
        self.colors = iter(default_colors)

        return mag_lo, mag_hi

    def get_catalog_data(self, gc, quantities, filters=[]):
        
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True,summary='Missing requested quantities')
        
        data = gc.get_quantities(quantities,filters=filters)
        #make sure data entries are all finite 
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data

    def get_validation_data(self, band, observation):

        validation_data = {}
        z0errors = np.asarray([])
        datafile = data_info[observation]['filename_template'].format(band)
        filename = os.path.join(self.data_dir,datafile)
        usecols = data_info[observation]['usecols']
        skiprows= data_info[observation]['skiprows']

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            if observation.find(Coil2004)!=-1:
                if len(usecols)==4:
                    mag_hi,mag_lo,z0values,z0errors = np.loadtxt(filename,unpack=True,usecols=usecols,skiprows=skiprows)
                    validation_data['z0errors'] = z0errors
                else:
                    mag_hi,mag_lo,z0values = np.loadtxt(filename,unpack=True,usecols=usecols,skiprows=skiprows)
                validation_data['z0values'] = z0values
                validation_data['mag_hi'] = mag_hi
                validation_data['mag_lo'] = mag_lo
                validation_data['label'] = Coil2004_lbl

            elif observation.find(DEEP2_JAN)!=-1:
                mag_hi_lim,mag_lo_lim,z0const,z0linear = np.loadtxt(filename,unpack=True,usecols=[0,1,2,3],skiprows=1)
                validation_data['z0const'] = z0const
                validation_data['z0linear'] = z0linear
                validation_data['mag_hi'] = []
                validation_data['mag_lo'] = [float(m) for m in range(int(mag_hi_lim),int(mag_lo_lim)+1)]
                validation_data['label'] = DEEP2_JAN_lbl
            else:
                raise ValueError('Observation {} not available'.format(observation))
        else:
            raise ValueError("{}-band data file {} not found or is empty".format(band,filename))
            
        return validation_data

    def run_on_single_catalog(self, galaxy_catalog, catalog_name, base_output_dir):
        
        #check for skip_messages
        if(hasattr(self,'skip_message')):
            return TestResult(skipped=True, summary= self.skip_message)

        #get catalog data
        mag_field = galaxy_catalog.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped = True,summary = 'Missing requested quantities')
        catalog_data = self.get_catalog_data(galaxy_catalog, [self.z, mag_field], filters=self.filters)
        filtername = mag_field.partition(self.band + '_')[-1].upper()
        filelabel = '_'.join([filtername,self.band])

        fig, ax  =  plt.subplots(self.nrows,self.ncolumns,figsize=(figx_p,figy_p),sharex='col')
        self.yaxis = 'P(z|m)' if self.normed else 'N(z|m)'

        catalog_color = next(self.colors)
        
        #loop over magnitude cuts and make plots
        results = {}
        for n,cut in enumerate(self.mag_lo):
            #initialize default values to use for validation-data fits
            z0 = 0.
            z0err = 0.
            #get masks for magnitude cuts and get validation-data fits depending on selected observation
            if self.observation.find(Coil2004)!= -1:
                mask = (catalog_data[mag_field]<=cut) & (catalog_data[mag_field]>self.mag_hi[n])
                cutlabel = ' '.join(['$',str(self.mag_hi[n]),'< $',self.band,'$<=',str(cut),'$'])
                
                #find correct row in validation data fits to use in plot
                omask = (self.mag_hi.astype(int)==int(self.mag_hi[n])) & (self.mag_lo.astype(int)==int(cut))
                z0 = self.validation_data['z0values'][omask]
                if('z0errors' in self.validation_data.keys()):
                    z0err = self.validation_data['z0errors'][omask]
            else:
                mask = (catalog_data[mag_field]<cut)
                cutlabel =  ' '.join([self.band,'$<',str(cut),'$'])
                if(self.observation.find(DEEP2_JAN)!= -1):
                    z0 = self.validation_data['z0const'] + self.validation_data['z0linear']*(cut)
            total = '(# of galaxies = {})'.format(np.sum(mask))
 
            #bin catalog_data
            N,binEdges = np.histogram(catalog_data[self.z][mask],bins=self.zbins)
            sumz,binEdges = np.histogram(catalog_data[self.z][mask],bins=self.zbins,weights=catalog_data[self.z][mask])
            meanz = sumz/N

            #make subplot
            ncol = int(n%self.ncolumns)
            nrow = n//self.ncolumns
            catalog_label = ' '.join([catalog_name,re.sub(self.band,filtername+' '+self.band,cutlabel)])
            validation_label = ' '.join([self.validation_data['label'],cutlabel])
            reskey = re.sub('\$','',cutlabel)
            results[reskey] = self.make_subplot(meanz,catalog_data[self.z][mask],n,ax[nrow,ncol],z0,z0err,catalog_color,validation_color,catalog_label,validation_label)
            results[reskey]['total'] = total

            #add curve for this catalog to summary plot
            if self.first_pass:  #add validation data if evaluating first catalog
                summary = self.make_subplot(meanz,catalog_data[self.z][mask],n,self.summary_ax[nrow,ncol],z0,z0err,catalog_color,validation_color,catalog_label,validation_label)
            else:
                summary = self.make_subplot(meanz,catalog_data[self.z][mask],n,self.summary_ax[nrow,ncol],0.,0.,catalog_color,validation_color,catalog_label,validation_label)

        #make empty subplots invisible 
        if len(self.mag_lo)%self.ncolumns!=0: #check for empty subplots
                ax[self.nrows-1,self.ncolumns-1].set_visible(False)   #assumes ncolumns=2
                
        #save results for catalog and validation data in txt files
        for filename, dtype, comment, info in zip([filelabel,self.observation],['y','fit'],[filtername,''],['total','']):
            with open(os.path.join(base_output_dir,'Nvsz_'+filename+'.txt'),'ab') as f_handle:     #open file in append mode
                 for key in results.keys():
                     xinfo = results[key][info] if len(info)>0 else ''
                     self.save_quantities(dtype, results[key], f_handle,comment=' '.join([comment,key,xinfo]))

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #save figure
        plt.savefig(os.path.join(base_output_dir, 'Nvsz_'+filelabel+'.png'))
        plt.close()
        return TestResult(0, passed = True)

    def make_subplot(self, meanz, catalog_data, nplot, f, z0, z0err, catalog_color, validation_color, catalog_label, validation_label):

        results = {}
        results['meanz'] = meanz
        if nplot%self.ncolumns==0:  #1st column
            f.set_ylabel('$'+self.yaxis+'$',size=fsize)

        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            #print "noticks",nplot
            for axlabel in f.get_xticklabels():
                axlabel.set_visible(False)
                    
                #prevent overlapping yaxis labels
                f.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            f.set_xlabel('$z$',size=fsize)
            for axlabel in f.get_xticklabels():
                axlabel.set_visible(True)

        #plot catalog data if available
        if len(catalog_data)>0:
            y,binEdges,_ = f.hist(catalog_data,bins=self.zbins,label=catalog_label,color=catalog_color,lw=lw2,normed=self.normed,histtype='step')
            results['y'] = y

        #plot validation data if available
        if z0>0.:
            ndata = meanz**2*np.exp(-meanz/z0)
            if self.normed:
                norm = self.nz_norm(self.zhi,z0)-self.nz_norm(self.zlo,z0)
                f.plot(meanz,ndata/norm,label=validation_label,ls='--',color=validation_color,lw=lw2)
                results['fit'] = ndata/norm
            else:
                raise ValueError("Only fits to normed plots are implemented so far")
            if z0err > 0.:
                nlo = meanz**2*np.exp(-meanz/(z0-z0err))
                nhi = meanz**2*np.exp(-meanz/(z0+z0err))
                if self.normed:
                    normlo = self.nz_norm(self.zhi,z0-z0err)-self.nz_norm(self.zlo,z0-z0err)
                    normhi = self.nz_norm(self.zhi,z0+z0err)-self.nz_norm(self.zlo,z0+z0err)                    
                    f.fill_between(meanz, nlo/normlo, nhi/normhi, alpha=0.3,facecolor=validation_color)
                    results['fit+'] = nhi/normhi
                    results['fit-'] = nlo/normlo
                else:
                    raise ValueError("Only fits to normed plots are implemented so far")

        self.post_process_plot(f)

        return results

    def nz_norm(self, z, z0):
        nz_norm = z0*math.exp(-z/z0)*(-z*z-2.*z*z0-2.*z0*z0)
        return nz_norm

    def post_process_plot(self, ax):
        plt.subplots_adjust(hspace=0)
        ax.legend(loc='best',fancybox=True, framealpha=0.5, fontsize=lsize,numpoints=1)

    def save_quantities(self, keyname, results, filename, comment=''):

        if keyname+'-' in results and keyname+'+' in results:
            fields = ('meanz', keyname, keyname+'-', keyname+'+')
            header = ', '.join(['Data columns are: <z>',keyname,keyname+'-',keyname+'+',' '])
        else:                                
            fields = ('meanz', keyname)
            header = ', '.join(['Data columns are: <z>',keyname,' '])
        np.savetxt(filename, np.vstack((results[k] for k in fields)).T,
                fmt='%12.4e', header=header+comment)  
 

    def conclude_test(self, output_dir):
        #make empty subplots invisible 
        if len(self.mag_lo)%self.ncolumns!=0: #check for empty subplots
                self.summary_ax[self.nrows-1,self.ncolumns-1].set_visible(False)   #assumes ncolumns=2 for now
        plt.subplots_adjust(hspace=0)  #compress space
                
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
