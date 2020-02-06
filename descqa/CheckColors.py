from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys
import numpy as np
import numexpr as ne
from .base import BaseValidationTest, TestResult
from .plotting import plt
from astropy.table import Table
from scipy.spatial import distance_matrix
import ot
from numba import jit
import matplotlib as mpl

__all__ = ['CheckColors']

# Transformations of DES -> SDSS and DES -> CFHT are derived from Equations A9-12 and
# A19-22 the paper: arxiv.org/abs/1708.01531
# Transformations of SDSS -> CFHT are from:
# http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html
color_transformation = {'des2sdss': {}, 'des2cfht': {}, 'sdss2cfht': {}, 'lsst2cfht': {}, 'lsst2sdss':{}, 'sdss2lsst':{}, 'cfht2sdss':{}, 'cfht2lsst':{}}
color_transformation['des2sdss']['g'] = '1.10421 * g - 0.104208 * r'
color_transformation['des2sdss']['r'] = '0.102204 * g + 0.897796 * r'
color_transformation['des2sdss']['i'] = '1.30843 * i - 0.308434 * z'
color_transformation['des2sdss']['z'] = '0.103614 * i + 0.896386 * z'
color_transformation['des2cfht']['g'] = '0.945614 * g + 0.054386 * r'
color_transformation['des2cfht']['r'] = '0.0684211 * g + 0.931579 * r'
color_transformation['des2cfht']['i'] = '1.18646 * i - 0.186458 * z'
color_transformation['des2cfht']['z'] = '0.144792 * i + 0.855208 * z'
color_transformation['sdss2cfht']['u'] = 'u - 0.241 * (u - g)'
color_transformation['sdss2cfht']['g'] = 'g - 0.153 * (g - r)'
color_transformation['sdss2cfht']['r'] = 'r - 0.024 * (g - r)'
color_transformation['sdss2cfht']['i'] = 'i - 0.085 * (r - i)'
color_transformation['sdss2cfht']['z'] = 'z + 0.074 * (i - z)'
color_transformation['cfht2sdss']['u'] = 'u + 0.342 * (u - g)'
color_transformation['cfht2sdss']['g'] = 'g + 0.014 + 0.133 * (g - r) + 0.031 * (g - r) * (g - r)'
color_transformation['cfht2sdss']['r'] = 'r + 0.05 * (r - i)'
color_transformation['cfht2sdss']['i'] = 'i + 0.087 * (r - i)'
color_transformation['cfht2sdss']['z'] = 'z - 0.057 * (i - z)'
#these were derived from cosmoDC2 GCRCatalogs version = 0.14.4
color_transformation['lsst2sdss']['u'] = '0.203 * (u - g) + u + 0.04'
color_transformation['lsst2sdss']['g'] = '0.119 * (g - r) + g + 0.001'
color_transformation['lsst2sdss']['r'] = '0.025 * (r - i) + r + 0.001'
color_transformation['lsst2sdss']['i'] = '0.013 * (i - z) + i + 0.001'
color_transformation['lsst2sdss']['z'] = '-0.031 * (z - y) + z + 0.001'
color_transformation['sdss2lsst']['u'] = '0.932 * u + 1.865'
color_transformation['sdss2lsst']['g'] = '-0.11 * (g - r) + g + 0.001'
color_transformation['sdss2lsst']['r'] = '-0.026 * (r - i) + r - 0.001'
color_transformation['sdss2lsst']['i'] = '-0.01 * (i - z) + i'
color_transformation['sdss2lsst']['z'] = '1.001 * z + 0.043' 
#for these I combined the transformations above, CFHT actually should be MegaCam
color_transformation['cfht2lsst']['u'] = '1.251 * u - 0.319 * g + 1.865'
color_transformation['cfht2lsst']['g'] = 'g + 0.00837 * (g - r) + 0.028 * (g - r) * (g - r) + 0.0055 * (r - i) + 0.013'
color_transformation['cfht2lsst']['r'] = 'r - 0.02 * (r - i) - 0.001'
color_transformation['cfht2lsst']['i'] = 'i + 0.086 * (r - i) - 0.00943 * (i - z)'
color_transformation['cfht2lsst']['z'] = '1.058 * z - 0.057 * i + 0.043' 

class kernelCompare:
    def __init__(self,D1, D2):
        self._D1 = D1
        self._D2 = D2
        self._XY = np.vstack((D1, D2))
        self._scale = self._computeScale(self._XY)
        self._n1 = len(D1)
        self._n2 = len(D2)
        
    def _computeScale(self,XY):
        '''Compute and determine the kernel parameter by
        mean absolute deviation
        '''
        Z = XY -  np.mean(XY,0)
        Z = np.abs(Z)
        scaleXY = np.median(Z, 0)
        return scaleXY

    def _rbf(self,z1, z2):
        diff = z1 - z2
        diff /= self._scale
        diffSq = np.sum(diff * diff,1)
        res = np.exp(-diffSq)
        return res
    
    @staticmethod
    @jit(nopython=True)
    def _MMD2ufast( X, Y, scale):
        '''Compute the unbiased MMD2u statistics in the paper. 
        $$Ek(x,x') + Ek(y,y') - 2Ek(x,y)$$
        This function implemnts a fast version in linear time. 
        '''
        n1 = len(X)
        n2 = len(Y)
        k1 = 0.0
        for i in range(n1-1):
            diff = (X[i,:] - X[i+1,:])/scale
            diffSq = np.sum(diff * diff)
            k1 += np.exp(-diffSq)
        k1 /= n1 - 1
        
        k2 = 0.0
        for i in range(n2-1):
            diff = (Y[i,:] - Y[i+1,:])/scale
            diffSq = np.sum(diff * diff)
            k2 += np.exp(-diffSq)
        k2 /= n2 - 1
        
        k3 = 0.0
        p = min(n1, n2)
        for i in range(p):
            diff = (X[i,:] - Y[i,:])/scale
            diffSq = np.sum(diff * diff)
            k3 += np.exp(-diffSq)
        k3 /= p
        result = k1 + k2 - 2*k3
        return result

    def _compute_null_dist(self,iterations=500):
        '''Compute the bootstrap null-distribution of MMD2u.
        '''
        mmd2u_null = np.zeros(iterations)
        for i in range(iterations):
            idx = np.random.permutation(self._n1 + self._n2)
            XY_i = self._XY[idx, :]
            mmd2u_null[i] = self._MMD2ufast(XY_i[:self._n1,:], XY_i[self._n1:,], self._scale)

        return mmd2u_null

    def compute(self,iterations=500):
        '''Compute MMD^2_u, its null distribution and the p-value of the
        kernel two-sample test.
        '''
        mmd2u = self._MMD2ufast(self._D1, self._D2, self._scale)
        mmd2u_null = self._compute_null_dist(iterations)
        p_value = max(1.0/iterations,
                      (mmd2u_null > mmd2u).sum() /float(iterations))

        return mmd2u, p_value

    def plotDiff(self, coord1, coord2):
        v0min = np.min(self._XY[:,coord1])
        v1min = np.min(self._XY[:,coord2])
        v0max = np.max(self._XY[:,coord1])
        v1max = np.max(self._XY[:,coord2])
        nSeq = 50
        xSeq = np.linspace(v0min, v0max, nSeq)
        ySeq = np.linspace(v1min, v1max, nSeq)
        #xySeq = np.array(np.meshgrid(xSeq, ySeq)).T.reshape(-1,2)
        fGrid = np.zeros((nSeq, nSeq))
        znew = np.mean(self._XY, 0)
        for i in range(nSeq):
            for j in range(nSeq):
                znew[coord1] = xSeq[i]
                znew[coord2] = ySeq[j]
                #fGrid[i,j] = xSeq[i] *  xSeq[i]  ySeq[j] * ySeq[j]
                fpart1 = np.mean(self._rbf(znew, self._D1))
                fpart2 = np.mean(self._rbf(znew, self._D2))
                fGrid[i,j] = fpart1 - fpart2
        fig, ax = plt.subplots()
        vmax = np.max(np.abs(fGrid))
        vmax = max(vmax, 0.0005)
        cs = plt.contourf(xSeq, ySeq, fGrid.T, 
                          cmap = plt.get_cmap("RdBu"),
                         norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax))
        fig.colorbar(cs, ax=ax, shrink=0.9)
        
def wass1dim(data1, data2, numBins = 200):
    ''' Compare two one-dimensional arrays by the 
    Wasserstein metric (https://en.wikipedia.org/wiki/Wasserstein_metric).
    The input data should have outliers removed.
    
    Parameters
    ----------
        data1, data2: two one-dimensional arrays to compare.
        numBins: the number of bins.
        
    Outputs
    -------
        result: the computed Wasserstein metric.
        
    '''
    numBins = 200 ## number of bins
    upper = np.max( (data1.max(), data2.max() ) )
    lower = np.min( (data1.min(), data2.min() ) )
    xbins = np.linspace(lower, upper, numBins + 1)
    density1, _ = np.histogram(data1, density = False, bins = xbins)
    density2, _ = np.histogram(data2, density = False, bins = xbins)
    density1 = density1 / np.sum(density1)
    density2 = density2 / np.sum(density2)
    
    # pairwise distance matrix between bins
    distMat = distance_matrix(xbins[1:].reshape(numBins,1), 
                              xbins[1:].reshape(numBins,1))
    M = distMat
    T = ot.emd(density1, density2, M) # optimal transport matrix
    result = np.sum(T*M) # the objective data
    return result


def CompareDensity(data1, data2):
    ''' Compare two multi-dimensional arrays by the 
    Wasserstein metric (https://en.wikipedia.org/wiki/Wasserstein_metric).
    The input data should have outliers removed before applying this funciton.
    The multidimensional input data is projected onto multiple directions. 
    The Wasserstein metric is computed on each projected result. 
    This function returns the averaged metrics and its standard error. 
    
    
    Parameters
    ----------
        data1: the first multi-dimensional dataset. Each row is 
                an observation. Each column is a covariate. 
        data2: the second multi-dimensional dataset.
        numBins: the number of bins.
        K: the number of trial random projections.
        
    Outputs
    -------
        mu, sigma: the average discrepency measure and its standard error.
        
    '''
    K = 40 #4000
    result = np.zeros(K)
    pCovariate = data1.shape[1]
    for i in range(K):
        # random projection onto one dimension
        transMat = np.random.normal(size = (pCovariate, 1))
        transMat = transMat / np.linalg.norm(transMat, 'fro')
        data1_proj = data1 @ transMat
        data2_proj = data2 @ transMat
        # record the discrepency on the projected dimension
        # between two datasets.
        result[i] = wass1dim(data1_proj, data2_proj)
    return result.mean(), result.std()/np.sqrt(K)

class CheckColors(BaseValidationTest):
    """
    Inspection test to represent 2D color plots
    """
    def __init__(self, **kwargs): # pylint: disable=W0231
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'CheckColors')
        self.mag_fields_to_check = kwargs['mag_fields_to_check']
        self.redshift_cut = kwargs['redshift_cut']  
        self.validation_catalog = kwargs['validation_catalog'] 
        self.redshift_cut_val = kwargs['redshift_cut_val'] 
        self.mag_fields_val = kwargs['mag_fields_val']
        self.path_val = kwargs['path_val']

        if len(kwargs['xcolor']) != 2 or len(kwargs['ycolor']) != 2:
            print('Warning: color string is longer than 2 characters. Only first and second bands will be used.')

        self.xcolor = kwargs['xcolor'] 
        self.ycolor = kwargs['ycolor']
        self.bands = set(kwargs['xcolor'] + kwargs['ycolor'])
        self.bands_val = kwargs['bands_val']   
        
        self.zlo = kwargs['zlo']
        self.zhi = kwargs['zhi']
        self.zbins = kwargs['zbins']
        
        self.magcut = kwargs['magcut']
        self.magcut_band = kwargs['magcut_band']
        
        self.levels = kwargs['levels'] 
        
        self.kernel_iterations = kwargs['kernel_iterations']
        
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        has_results = False
        redshift_bins = np.linspace(self.zlo, self.zhi, num=self.zbins+1)
        catval = Table.read(self.path_val)
        labels_val = {band: self.mag_fields_val.format(band) for band in self.bands_val}
        datamag_val = {k: catval[v] for k, v in labels_val.items()}
        camlist = ['lsst','des','cfht','sdss']
        filter_this = None
        for mag_field in self.mag_fields_to_check:
            for cam in camlist:
                if cam in mag_field:
                    filter_this = cam

            quantity_list = [mag_field.format(band) for band in self.bands]
            quantity_list.append(self.redshift_cut)

            if not catalog_instance.has_quantities(quantity_list):
                print('Catalog is missing a quantity from',quantity_list)
                continue
            dataall = catalog_instance.get_quantities(quantity_list)
            #labels = {band: mag_field.format(band) for band in self.bands}
            #datamag = {k: dataall[v] for k, v in labels.items()}

            ### Color transformation
            color_trans = None
            color_trans_name = None
            if self.validation_catalog == 'DEEP2' and filter_this != 'lsst' and filter_this != 'cfht':
                color_trans_name = '{}2cfht'.format(filter_this) #not sure this is right
            elif self.validation_catalog == 'DEEP2' and filter_this == 'lsst':
                color_trans_name = 'cfht2lsst'
            elif self.validation_catalog == 'SDSS' and filter_this == 'des':
                color_trans_name = 'des2sdss' #not sure this is right
            elif self.validation_catalog == 'SDSS' and filter_this == 'lsst':
                color_trans_name = 'sdss2lsst'
            if color_trans_name:
                color_trans = color_transformation[color_trans_name]

            filter_title = r'\mathrm{{{}}}'.format(filter_this.upper())

            if color_trans:
                #print('Transforming from %s to %s\n' % (self.validation_catalog,filter_this))
                datamag_val_transformed = {}
                for band in self.bands_val:
                    try:
                        datamag_val_transformed[band] = ne.evaluate(color_trans[band], local_dict=datamag_val, global_dict={})
                    except KeyError:
                        continue

                filter_title = (r'{}\rightarrow\mathrm{{{}}}'.format(filter_title, self.validation_catalog)
                            if datamag_val_transformed else filter_title)
                datamag_val_transformed['redshift'] = catval[self.redshift_cut_val] #to avoid confusion between z and redshift
                catval = datamag_val_transformed
                del datamag_val_transformed
                del datamag_val
            else:
                datamag_val['redshift'] = catval[self.redshift_cut_val]
                catval = datamag_val
                del datamag_val
            
            for i,zlo in enumerate(redshift_bins):
                if i == len(redshift_bins)-1:
                    continue
                zhi = redshift_bins[i+1]
                mask = (dataall[self.redshift_cut] > zlo) & (dataall[self.redshift_cut] < zhi) & (dataall[mag_field.format(self.magcut_band)] < self.magcut)
                mask_val = (catval['redshift'] > zlo) & (catval['redshift'] < zhi) & (catval[self.magcut_band] < self.magcut)
                try:
                    xcolor = np.array(dataall[mag_field.format(self.xcolor[0])][mask] - dataall[mag_field.format(self.xcolor[1])][mask])
                    ycolor = np.array(dataall[mag_field.format(self.ycolor[0])][mask] - dataall[mag_field.format(self.ycolor[1])][mask])
                    xcolor_val = np.array(catval['{}'.format(self.xcolor[0])][mask_val] - catval['{}'.format(self.xcolor[1])][mask_val])
                    ycolor_val = np.array(catval['{}'.format(self.ycolor[0])][mask_val] - catval['{}'.format(self.ycolor[1])][mask_val])
                except KeyError:
                    print('Key not found')
                    sys.exit()
                has_results = True

                ### plot hexbin plot for catalog
                fig, ax = plt.subplots()
                ax.hexbin(xcolor, ycolor, gridsize=(100), cmap='GnBu', mincnt=1, bins='log')
                # plot contour plot for validation
                xmin = -0.5
                xmax = 1.0
                ymin = 0.0
                ymax = 2.0
                hrange = [[xmin,xmax],[ymin,ymax]]
                counts,xbins,ybins = np.histogram2d(xcolor_val,ycolor_val,range=hrange,bins=[30,30])
                cntr1 = ax.contour(counts.transpose(), extent=[xmin,xmax,ymin,ymax],
                                   colors='black',linestyles='solid',levels=self.levels)
                h1,_ = cntr1.legend_elements()
                
                ### CompareDensity block
                simdata = np.column_stack([xcolor,ycolor])
                valdata = np.column_stack([xcolor_val,ycolor_val])
                cd = CompareDensity(simdata,valdata)
                print('Compare density result',cd) 
                
                ### kernel comparison block
                obj = kernelCompare(simdata, valdata)
                MMD, pValue = obj.compute(iterations=self.kernel_iterations)
                print("MMD statistics is {}".format(MMD))
                print("The p-value of the test is {}".format(pValue))

                ax.set_xlabel('{} - {}'.format(mag_field.format(self.xcolor[0]), mag_field.format(self.xcolor[1])))
                ax.set_ylabel('{} - {}'.format(mag_field.format(self.ycolor[0]), mag_field.format(self.ycolor[1])))
                title = "{} = {:.2} - {:.2}".format(self.redshift_cut, zlo, zhi)
                ax.text(0.05, 0.95, title, transform=ax.transAxes, 
                        verticalalignment='top', color='black', fontsize='small')
                title1 = "Compare metric {:.4} +- {:.4}".format(cd[0],cd[1])
                title2 = "Kernel comparison MMD {:.4} p-value = {:.3}".format(MMD,pValue)
                ax.text(0.05, 0.85, title1, transform=ax.transAxes, 
                        verticalalignment='top', color='black', fontsize='small')
                ax.text(0.05, 0.80, title2, transform=ax.transAxes, 
                        verticalalignment='top', color='black', fontsize='small')
                ax.set_title('Color inspection for {} vs {}'.format(catalog_name, self.validation_catalog))

                plt.legend([h1[0]], [self.validation_catalog])
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, '{}_{}_{}_{}.png'.format(self.xcolor,               self.ycolor,str(i),mag_field.replace('_{}_', '_'))))
                plt.close(fig)

        if not has_results:
            return TestResult(skipped=True)

        return TestResult(inspect_only=True)
