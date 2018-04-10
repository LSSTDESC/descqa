from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt
import kmeans_radec
__all__ = ['ConditionalLuminosityFunction_redmapper']

class ConditionalLuminosityFunction_redmapper(BaseValidationTest):
    def __init__(self, **kwargs):
        self.band = kwargs.get('band1', 'r')
        possible_mag_fields = ('mag_{0}_lsst',)
        self.possible_mag_fields = [f.format(self.band) for f in possible_mag_fields]
        self.njack = kwargs.get('njack',20)
        self.lambd_bins = np.linspace(*kwargs.get('lambd_bins',(5,100,6)))
        self.z_bins = np.linspace(*kwargs.get('z_bins', (0.2, 1.0, 5)))
        self.magnitude_bins = np.linspace(*kwargs.get('magnitude_bins', (-26, -18, 29)))
        self.n_magnitude_bins = len(self.magnitude_bins) - 1
        self.nlambd_bins      = len(self.lambd_bins) - 1
        self.n_z_bins         = len(self.z_bins) - 1

        self.dmag = self.magnitude_bins[1:] - self.magnitude_bins[:-1]
        self.lambd_center = (self.lambd_bins[1:] + self.lambd_bins[:-1])*0.5
        self.mag_center = (self.magnitude_bins[1:] + self.magnitude_bins[:-1])*0.5
        self._other_kwargs = kwargs
        
        
    def prepare_galaxy_catalog(self, gc):
        
        quantities_needed = {'cluster_id_member', 'cluster_id', 'ra','dec','ra_cluster','dec_cluster','richness','redshift','LIM_LIMMAG_DERED','p_mem','SCALEVAL','p_cen','redshift_cluster'}
        print("test1")
        try:
            magnitude_field = gc.first_available(*self.possible_mag_fields)
        except ValueError:
            return
        quantities_needed.add(magnitude_field)
        if not gc.has_quantities(quantities_needed):
            print(quantities_needed)
            return

        print("test2")
        return magnitude_field, quantities_needed
        
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        prepared = self.prepare_galaxy_catalog(catalog_instance)
        if prepared is None:
            TestResult(skipped=True)
        magnitude_field, quantities_needed = prepared
        quant = catalog_instance.get_quantities(quantities_needed)
        mag = catalog_instance.get_quantities(magnitude_field)[magnitude_field]
        Mag = mag-catalog_instance.cosmology.distmod(quant['redshift']).value
        mask = (quant['richness']>1)
        cenMag, cengalindex = self.get_central_mag_id(quant['cluster_id'][mask], quant['cluster_id_member'],quant['ra_cluster'][mask],quant['dec_cluster'][mask],quant['ra'],quant['dec'],Mag)
        limmag = quant['LIM_LIMMAG_DERED'][mask]
        limmag = limmag - catalog_instance.cosmology.distmod(quant['redshift_cluster']).value[mask]
        pcen_all = np.zeros(len(quant['ra']))
        pcen_all[cengalindex.flatten().astype(int)] = 1.
        jackList = self.make_jack_samples_simple(quant['ra_cluster'][mask],quant['dec_cluster'][mask])
        
        match_index_jack = self.getjackgal(jackList[0], quant['cluster_id'][mask], quant['cluster_id_member'])
        cenclf, satclf, covar_cen, covar_sat = self.redm_clf( Mag,cenMag,cengalindex,
                       pcen_all,jackList,quant['cluster_id'][mask],quant['cluster_id_member'],
                       match_index=match_index_jack,
                       limmag=limmag,scaleval =quant['SCALEVAL'][mask], pmem=quant['p_mem'], pcen=quant['p_cen'][mask],
                     cluster_lm = quant['richness'][mask], cluster_z=quant['redshift_cluster'][mask]
                     )
        
#        print(cenclf.shape, covar_cen.shape)
        clf = {'satellites':satclf, 'centrals':cenclf}
        covar = {'satellites':covar_sat, 'centrals':covar_cen}
        self.make_plot(clf, covar, catalog_name, os.path.join(output_dir, 'clf_redmapper.png'))
        return TestResult(inspect_only=True)

    def make_plot(self, clf, covar,name,save_to):
        fig, ax = plt.subplots(self.nlambd_bins, self.n_z_bins, sharex=True, sharey=True, figsize=(12,10), dpi=100)

        for i in range(self.n_z_bins):
            for j in range(self.nlambd_bins):
                ax_this = ax[j,i]
                for k, fmt in zip(('satellites', 'centrals'), ('^', 'o')):
                    ax_this.errorbar(self.mag_center, clf[k][i,j], yerr=np.sqrt(np.diag(covar[k][i,j])), label=k, fmt=fmt)
                ax_this.set_ylim(0.05, 50)
                bins = self.lambd_bins[j], self.lambd_bins[j+1], self.z_bins[i], self.z_bins[i+1]
                ax_this.text(-25, 10, '${:.1E}\\leq \lambda <{:.1E}$\n${:g}\\leq z<{:g}$'.format(*bins))
                ax_this.set_yscale("log")
        ax_this.legend(loc='lower right', frameon=False, fontsize='medium')

        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(False)
        ax.set_ylabel(r'$\phi(M_{{{}}}\,|\,M_{{\rm vir}},z)\quad[{{\rm Mag}}^{{-1}}]$'.format(self.band))
        ax.set_xlabel(r'$M_{{{}}}\quad[{{\rm Mag}}]$'.format(self.band))
        ax.set_title(name)

        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)

    def get_central_mag_id(self, cat_mem_match_id, mem_mem_match_id,ra_cluster,dec_cluster,ra,dec,mag):
        ncluster = len(cat_mem_match_id)
        ngals = len(mem_mem_match_id)
        cenmag = np.zeros([ncluster,1])
        cengalindex = np.zeros([ncluster,1])-1
        count_lo = 0
        count_hi = 0
        ncent = 1
        for i in range(ncluster):
            idList=[]
            while cat_mem_match_id[i] != mem_mem_match_id[count_lo]:
                if  cat_mem_match_id[i] > mem_mem_match_id[count_lo]:
                    idList.append(mem_mem_match_id[count_lo])
                #print(cat_mem_match_id[i],mem_mem_match_id[count_lo])
                count_lo = count_lo+1
                count_hi = count_lo
            while cat_mem_match_id[i] == mem_mem_match_id[count_hi]:
                count_hi += 1
                if count_hi >= ngals:
                    break
            #if len(idList)!=0:
            #    print("no ID for mem id = {0}".format(np.unique(idList)))
            for j in range(ncent):
                if len(ra_cluster.shape)==1:
                    place = np.where((np.abs(ra_cluster[i]-ra[count_lo:count_hi])<1E-5) &
                              (np.abs(dec_cluster[i]-dec[count_lo:count_hi])<1E-5) )[0]
                if len(place) == 0:
                    print("WARNING:  Possible ID issue in get_central_mag")
                    cenmag[i][j] = 0
                    cengalindex[i][j] = -1
                    continue
                cenmag[i][j] = mag[count_lo+place[0]]
                cengalindex[i][j] = count_lo+place[0]

            count_lo = count_hi
        return cenmag, cengalindex
    def make_jack_samples_simple(self, RA, DEC):
        radec=np.zeros( (len(RA), 2) )
        radec[:,0] = RA.flatten()
        radec[:,1] = DEC.flatten()
        _maxiter=100
        _tol=1.0e-5
        _km = kmeans_radec.kmeans_sample(radec, self.njack, maxiter=_maxiter, tol=_tol)
        uniquelabel = np.unique(_km.labels)
        jacklist = np.empty(self.njack, dtype=np.object)
        for i in range(self.njack):
             jacklist[i]=np.where(_km.labels!=uniquelabel[i])[0]
        return jacklist
    
    def getjackgal(self,jackclusterList,c_mem_id,g_mem_id, match_index=None):
        nclusters = len(c_mem_id)
        ngals = len(g_mem_id)

        if match_index is None:
            ulist = np.zeros(nclusters).astype(int)
            place = 0
            for i in range(ngals):
                if place == 0:
                    ulist[place]=0
                    place = place+1
                    continue
                if g_mem_id[i] == g_mem_id[i-1]:
                    continue
                ulist[place] = i
                place = place+1
                if place >= nclusters:
                    break
            match_index = np.zeros(np.max(c_mem_id)+1).astype(int)-1
            match_index[c_mem_id] = ulist
            return match_index
        max_id = np.max(c_mem_id)
        galpos = np.arange(len(g_mem_id))
        gboot_single=[]
        for index in jackclusterList:
            mygal = match_index[c_mem_id[index]]
            if c_mem_id[index] == max_id:
                endgal = len(g_mem_id)
            else:
                endgal = match_index[c_mem_id[index+1]]
            if endgal < mygal:
                print("Something has gone wrong, ")
            glist = galpos[mygal:endgal]
            if len(glist) > 0:
                gboot_single.extend(glist)
            else:
                gboot_single.extend([])

        return gboot_single
    def count_galaxies_p(self,c_mem_id,scaleval,g_mem_id,p,mag,lumbins):
        nclusters = len(c_mem_id)
        nlum = len(lumbins)
        dlum = lumbins[1]-lumbins[0]
        minlum = lumbins[0]-dlum/2.
        maxlum = lumbins[-1]+dlum/2.
        count_arr = np.zeros([nclusters,nlum])

        max_id =np.max(c_mem_id)
        index = np.zeros(max_id+1) - 100
        index[c_mem_id] = range(len(c_mem_id))
        mylist = np.where( (mag <= maxlum) & (mag >= minlum) )[0]
        if len(mylist)==0:
            print("WARNING:  No galaxies found in range {0}, {1}".format(minlum,maxlum))
            return count_arr

        for i in range(len(mylist)):
            if g_mem_id[mylist[i]] > max_id:
                continue
            mycluster = int(index[g_mem_id[mylist[i]]])
            mybin = np.floor((mag[mylist[i]]-minlum)/dlum)
            mybin = mybin.astype(int)
            if (mybin < 0) | (mybin >= nlum) | (mycluster == -100):
                continue
            count_arr[mycluster,mybin] += p[mylist[i]]*scaleval[mycluster]

        return count_arr
    
    def count_galaxies_p_cen(self,cenmag,lumbins,p_cen):
        nlum = len(lumbins)
        nclusters = len(cenmag)
        dlum = lumbins[1]-lumbins[0]
        minlum = lumbins[0]-dlum/2.
        chto_countArray=np.zeros([len(cenmag),nlum])
        mybin = np.floor((cenmag[:,:]-minlum)/dlum).astype(np.int)
        p_cen = p_cen.reshape(-1,1)
        ncen = np.zeros(p_cen.shape)
        ncen = np.hstack(((np.ones(p_cen.shape[0])).reshape(-1,1),ncen[:,:-1])).astype(np.int)
        weight = p_cen

        ys = np.outer(np.ones(p_cen.shape[0]),np.arange(p_cen.shape[1]))
        mask = ((mybin>=0) & (mybin<nlum)& (ys<ncen)).astype(np.float64)
        newbin=mybin*mask
        indexArray=np.arange(nlum)
        for i in indexArray:
            masknew=(mybin==i).astype(np.float64)*mask
            newNewbin=np.sum(masknew*weight,axis=1)
            chto_countArray[:,i]=newNewbin[:]
        return chto_countArray
    def cluster_Lcount(self,lumbins,limmag):
        nclusters_lum = np.zeros_like(lumbins)
        binlist = np.zeros_like(limmag).astype(int)
        dlum = lumbins[1]-lumbins[0]
        minlum = lumbins[0]-dlum/2.
        nlumbins = len(lumbins)
        p = np.zeros_like(limmag)+1

        for i in range(len(limmag)):
            mybin = int(np.floor( (limmag[i] - minlum)/dlum ))
            binlist[i] = 0
            if mybin > nlumbins:
                nclusters_lum = nclusters_lum + p[i]
                binlist[i] = nlumbins
            if (mybin > 0) & (mybin <= nlumbins):
                nclusters_lum[:mybin] = nclusters_lum[:mybin] + p[i]
                binlist[i] = mybin
        return nclusters_lum, binlist
    
    def make_single_clf(self,lm,z,lumbins,count_arr,lm_min,lm_max,zmin,zmax,
                    limmag=[]):
        dlum = lumbins[1]-lumbins[0]
        clf = np.zeros_like(lumbins).astype(int)

        clist = np.where( (z >= zmin) & (z < zmax) & (lm >= lm_min) & (lm < lm_max) )[0]

        if len(clist) == 0:
            print("WARNING: no clusters found for limits of: {0} {1} {2} {3}".format(lm_min,lm_max,zmin,zmax))
            return clf

        [nclusters_lum, binlist] = self.cluster_Lcount(lumbins,limmag[clist])
        for i in range(len(clist)):
            clf[:binlist[i]] = clf[:binlist[i]] + count_arr[clist[i],:binlist[i]]

        clf = clf/nclusters_lum/dlum

        return clf
    
    def redm_clf(self, Mag,cenMag,cengalindex,
                           pcen_all,jacklist,cluster_id, cluster_id_member,
                           match_index,
                           limmag,scaleval,pmem, pcen, cluster_lm, cluster_z):
        lumbins = self.magnitude_bins[1:]
        nlum = len(lumbins)
        zmin = self.z_bins[:-1]
        zmax = self.z_bins[1:]
        my_nz = len(zmin)
        lm_min = self.lambd_bins[:-1]
        lm_max = self.lambd_bins[1:]
        lm_min = np.repeat([lm_min],my_nz,axis=0)
        lm_max = np.repeat([lm_max],my_nz,axis=0)


        nlambda = len(lm_min[0])
        nz = len(zmin)
        njack = len(jacklist)
        nclusters = len(cenMag)
        cenclf = np.zeros([nz,nlambda,nlum])
        satclf = np.zeros([nz,nlambda,nlum])
        sat_count_arr = self.count_galaxies_p(cluster_id,scaleval,cluster_id_member,
                                 pmem*(1-pcen_all),Mag,lumbins)
        cen_count_arr = self.count_galaxies_p_cen(cenMag,lumbins,pcen)
        

        for i in range(nz):
            for j in range(nlambda):
                cenclf[i,j] = self.make_single_clf(cluster_lm,cluster_z,
                                          lumbins,cen_count_arr,lm_min[i,j],lm_max[i,j],
                                          zmin[i],zmax[i],limmag=limmag)
                satclf[i,j] = self.make_single_clf(cluster_lm,cluster_z,
                                          lumbins,sat_count_arr,lm_min[i,j],lm_max[i,j],
                                          zmin[i],zmax[i],limmag=limmag,)
        njack = len(jacklist)
        cenclf_jack = np.zeros([njack,nz,nlambda,nlum])
        satclf_jack = np.zeros([njack,nz,nlambda,nlum])
        
        for i in range(njack):
            gjack = self.getjackgal(jacklist[i], cluster_id, cluster_id_member, match_index)
            sat_count_arr_b = self.count_galaxies_p(cluster_id[jacklist[i]],
                                 scaleval[jacklist[i]],cluster_id_member[gjack],
                                 (pmem*(1-pcen_all))[gjack],Mag[gjack],lumbins)
            cen_count_arr_b = self.count_galaxies_p_cen(cenMag[jacklist[i]],lumbins,pcen[jacklist[i]])
            my_limmag = limmag[jacklist[i]]
            for j in range(nz):
                for k in range(nlambda):
                    cenclf_jack[i,j,k,:] = self.make_single_clf(cluster_lm[jacklist[i]],cluster_z[jacklist[i]],
                                          lumbins,cen_count_arr_b,lm_min[j,k],lm_max[j,k],
                                          zmin[j],zmax[j],limmag=limmag)
                    satclf_jack[i,j,k,:] = self.make_single_clf(cluster_lm[jacklist[i]],cluster_z[jacklist[i]],
                                          lumbins,sat_count_arr_b,lm_min[j,k],lm_max[j,k],
                                          zmin[j],zmax[j],limmag=limmag,)
        covar_cen = np.zeros([nz,nlambda,nlum,nlum])
        covar_sat = np.zeros([nz,nlambda,nlum,nlum])
        meanjack_cen = np.zeros([nz,nlambda,nlum])
        meanjack_sat = np.zeros([nz,nlambda,nlum])
        for i in range(nz):
            for j in range(nlambda):
                for k in range(nlum):
                    for l in range(nlum):
                        mean_jack_cen_k = np.mean(cenclf_jack[:,i,j,k])
                        mean_jack_cen_l = np.mean(cenclf_jack[:,i,j,l])
                        mean_jack_sat_k = np.mean(satclf_jack[:,i,j,k])
                        mean_jack_sat_l = np.mean(satclf_jack[:,i,j,l])
                        meanjack_cen[i,j,k] = mean_jack_cen_k
                        meanjack_sat[i,j,k] = mean_jack_sat_k
                        covar_cen[i,j,k,l] = np.sum( ( cenclf_jack[:,i,j,k] - mean_jack_cen_k )*
                                                 ( cenclf_jack[:,i,j,l] - mean_jack_cen_l) )/(njack)*(njack-1)
                        covar_sat[i,j,k,l] = np.sum( ( satclf_jack[:,i,j,k] - mean_jack_sat_k )*
                                                 ( satclf_jack[:,i,j,l] - mean_jack_sat_l ) )/(njack)*(njack-1)
        return cenclf, satclf, covar_cen, covar_sat
    
