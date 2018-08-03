from __future__ import unicode_literals, absolute_import, division
from .plotting import plt 
from .base import BaseValidationTest, TestResult
import numpy as np 
import matplotlib
from matplotlib.colors import LogNorm
from astropy.cosmology import WMAP7 as cosmo
import astropy.stats as stat
import os
import pdb

__all__ = ['VirialScaling']
class VirialScaling(BaseValidationTest):
    """
    Tests the relationship between FOF halo mass and velocity dispersion of cluster member galaxies 
    """
    def __init__(self, masscut, c_axis, stellarcut, disp_func,**kwargs):
        '''initialize the test with the folowing quantities:
        masscut: include halos with masses at least as large as specified value
        c_axis: pass 'number' for color axis to display cluster galaxy counts, and 'redshift' for cluster redshifts
        disp_func: pass 'biweight' to calculate velocity dispersion using biweight scale, defaults to standard deviation'''

        self.stellarcut = stellarcut
        self.masscut = masscut
        self.c_axis = c_axis
        self.disp_func = disp_func
        self.summary_fig, self.summary_ax = plt.subplots()
		
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''collect quantities and plot the relationship between velocity dispersion and halo mass'''

        if not catalog_instance.has_quantities(['halo_mass','halo_id', 'velocity_x', 'velocity_y', 'velocity_z', 'redshift']):
            return TestResult(skipped = True, summary = 'do not have needed quantities')

        #list containing each galaxy's larger cluster mass
        complete_mass = catalog_instance.get_quantities('halo_mass')['halo_mass']

        #sort the complete_mass list 
        #make a list of indices corresponding to halos that make the mass cut
        mass_indices = np.argsort(complete_mass)
        complete_mass = complete_mass[mass_indices]
        complete_mass = np.array(complete_mass)
        start_index = np.searchsorted(complete_mass, self.masscut)
        cut_indices = list(range(start_index, np.size(complete_mass)))

        #make a list of each galaxy's halo ID, sort it in same fashion as mass, and cut according to mass cut
        complete_id_list = catalog_instance.get_quantities('halo_id')['halo_id']
        complete_id_list = complete_id_list[mass_indices]
        
        cut_id_list = complete_id_list[cut_indices]

        #sort the cut_id_list in increasing order and get the unique values to prepare for looping
        indexing_indices = np.argsort(cut_id_list)
        cut_id_list = cut_id_list[indexing_indices]
        unique_id_list = np.unique(cut_id_list)
		
        #make a list of masses that make the mass cut and are sorted in the same fashion as the id list
        cut_masses = complete_mass[cut_indices][indexing_indices]
        
        #make a list of galaxy masses, used to check galaxy mass and disregard any galaxies with low stellar masses
        stellar_masses = catalog_instance.get_quantities('stellar_mass')
        stellar_masses = stellar_masses['stellar_mass'][mass_indices][cut_indices][indexing_indices]

        #list to contain velocity magnitudes of galaxies within a cluster
        vel_mag_list = np.array([])
        #list to contain velocity dispersions of cluster galaxies
        vel_dispersion = np.array([])
        #list to contain the masses of clusters to be plotted
        mass = np.array([])


        #fetch and sort each velocity component for each galaxy according to the id list
        vx = catalog_instance.get_quantities('velocity_x')
        vx_list = vx['velocity_x'][mass_indices][cut_indices][indexing_indices]
        vy = catalog_instance.get_quantities('velocity_y')
        vy_list = vy['velocity_y'][mass_indices][cut_indices][indexing_indices]
        vz = catalog_instance.get_quantities('velocity_z')
        vz_list = vz['velocity_z'][mass_indices][cut_indices][indexing_indices]

        #Get a list of redshifts for each galaxy in the catalog, sorted according to the ID's
        redshifts = catalog_instance.get_quantities('redshift')['redshift']
        redshifts = redshifts[mass_indices][cut_indices][indexing_indices]

        largest = 0
        smallest = np.max(cut_masses)
        
        #list to contain redshifts of all galaxies within a particular cluster
        redshift_list = np.array([])
        #list to contain representative redshift for each cluster (determined from galaxy redshifts)
        median_r = np.array([])

        #galaxy counter and list to store number of galaxies in each cluster

        galaxy_num  = 0
        galaxy_num_list = np.array([])

        #function used to calculate velocity dispersion
        def dispersion(val_array):
            if(self.disp_func == "biweight"):
                dis = stat.biweight_scale(val_array)
            else:
                dis = np.std(val_array)
            return dis

        #for each cluster above the mass cut
        for unique_id in unique_id_list:

            #find location of this cluster's galaxies in the list 
            index = np.searchsorted(cut_id_list, unique_id)

            #add the cluster's mass to a list
            mass = np.append(mass, cut_masses[index])

            #for every galaxy that is part of the same cluster, make list of velocity magnitudes,
            #galaxy counts, and redshifts
            while unique_id == cut_id_list[index]:

                if(stellar_masses[index] > self.stellarcut):
                    vel_mag = np.sqrt(np.power(vx_list[index],2)+np.power(vy_list[index],2)+np.power(vz_list[index],2))
                    vel_mag_list = np.append(vel_mag_list, vel_mag)
                    redshift_list = np.append(redshift_list, redshifts[index])		
                    galaxy_num += 1
                
                index+=1

                if (index == np.size(cut_id_list)):
                    break

            #append each calculated value for the cluster to the proper list
            galaxy_num_list = np.append(galaxy_num_list, galaxy_num)
           
            mask_num = galaxy_num_list>5
            galaxy_num = 0
	    
            vel_dispersion = np.append(vel_dispersion, dispersion(vel_mag_list))
	 
            #use a representative, robust estimate of redshift for the whole cluster
            median_r = np.append(median_r, np.median(redshift_list))
                        
            #la = cut_masses[index-1]*(cosmo.H(np.median(redshift_list)).value/100)
            #sm = cut_masses[index-1]*(cosmo.H(np.median(redshift_list)).value/100) 
            if(cut_masses[index-1]*((cosmo.H(np.median(redshift_list))).value/100)>largest):
               largest = cut_masses[index-1]*(cosmo.H(np.median(redshift_list)).value/100)
            
            if(cut_masses[index-1]*((cosmo.H(np.median(redshift_list))).value/100)<smallest):
               smallest = cut_masses[index-1]*(cosmo.H(np.median(redshift_list)).value/100)  
            
            #reset lists to empty for the next iteration/cluster
            vel_mag_list = np.array([])
            redshift_list = np.array([])
            

        #fig, ax = plt.subplots(nrows=1,ncols=1)
        #make different plots depending on what you want the color axis to show

        x_axis = np.multiply(mass[mask_num], (cosmo.H(median_r[mask_num])/100))

        if (self.c_axis == 'number'):
           img = self.summary_ax.scatter(x_axis, vel_dispersion[mask_num], c = galaxy_num_list[mask_num], norm = LogNorm(), label = catalog_name + ' cluster') 

        elif(self.c_axis == 'redshift'):
           img =self.summary_ax.scatter(x_axis, vel_dispersion[mask_num], c = median_r[mask_num], norm = LogNorm(), label = catalog_name + ' cluster')	
        #make plot
        x = np.linspace(smallest*.75, largest*1.5)
        self.summary_ax.plot(x, eval("1082*(x/10**15)**.3361"), c = "red", label = "Evrard et al. 2007")
        self.summary_ax.legend()
        bar = self.summary_fig.colorbar(img, ax = self.summary_ax)
        self.summary_ax.set_xscale('log')
        self.summary_ax.set_ylim(np.min(vel_dispersion[mask_num])*.3, np.max(vel_dispersion[mask_num])*5)
        self.summary_ax.set_xlim(smallest*.75, largest*1.5)
        self.summary_ax.set_yscale('log')
        self.summary_ax.set_xlabel('$Mh(z)  (M_{\odot})$')
        self.summary_ax.set_ylabel('$\sigma_v$ (km/s)')
        #label color axis depending on what you want to show
        if(self.c_axis == 'number'):
            bar.ax.set_ylabel('galaxies per cluster')
        else:
            bar.ax.set_ylabel('median redshift')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mass_virial_scaling.png'))
        plt.close()
		
        return TestResult(inspect_only = True)
		
    def conclude_test(self, output_dir):
        '''conclude the test'''
        self.summary_fig.savefig(os.path.join(output_dir, 'mass_virial_scaling.png'))
        plt.close(self.summary_fig)
