from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['ObjectObscurationTest']

def ring(i, intr, extr):
    area = np.pi*(extr*extr-intr*intr)
    return area

class ObjectObscurationTest(BaseValidationTest):
    """
    A test to quantify object obscuration by, e.g., foreground stars
    """
    def __init__(self, **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.nbins = kwargs.get('nbins', 10)
        self.distance = kwargs.get('distance', 20) #max. distance in arcsec
        self.reference_band = kwargs.get('reference_band','r')
        self.cutname = self.reference_band+'_mag'
        self.cut = kwargs.get('cut',[[18, 19], [19, 20],[20,21],[21,22]])
        
        # load validation data
        with open(os.path.join(self.data_dir, 'README.md')) as f:
            self.validation_data = f.readline().strip()

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()


    def post_process_plot(self, ax):
        
        return
            
    def create_sample(self,catalog_data, selection, is_binned = False):
        
        data_selection = [catalog_data[selection]['ra']*3600.,
                          catalog_data[selection]['dec']*3600.] #to arcsecs
    
        nBbin = [] #for use if is_binned = True

        if is_binned:
            cutname_values = data_selection[self.cutname]        

            xBbin, yBbin = [], []
            for i in range(len(self.cut)):
                xBbin.append(xB[(self.cut[i][0] <= cutname_values) & (cutname_values < self.cut[i][1])])
                yBbin.append(yB[(self.cut[i][0] <= cutname_values) & (cutname_values < self.cut[i][1])])
                nBbin.append(len(xBbin[i]))

            for i in range(len(self.cut)):
                for j in range(nBbin[i]):
                    tmpB.append([xBbin[i][j], yBbin[i][j]])
                sample.append(np.asarray(tmpB))
                del tmpB
        else:
            sample = np.asarray(data_selection)

        return sample, nBbin
    
    def calc_nearest_neighbors(self, main_sample, bright_sample):
        neigh = NN(radius = self.distance+1, metric = 'euclidean')
        neigh.fit(main_sample)
        distB = [] #distance from central bright object

        for i in range(ncut):
            dist, ind = neigh.radius_neighbors(bright_sample[i])
            distB.append(dist)

        return distB

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # check if needed quantities exist       
        quantities = ['ra','dec','extendedness',self.cutname]
        if not catalog_instance.has_quantities(quantities):
            return TestResult(skipped=True, summary='do not have needed quantities')
        
        # create filter labels
        filters=[]
        for i, filt in enumerate(self.catalog_filters):
            filters.append(filt['filters'])
        filters = list(chain(*filters))

        #load data
        if len(filters) > 0:
                catalog_data = catalog_instance.get_quantities(quantities,
                                                               filters=filters,
                                                               return_iterator=False)
            else:
                catalog_data = catalog_instance.get_quantities(quantities,
                                                               return_iterator=False)
        
        # create main sample
        main_cut = 'extendedness == 1'        
        main_sample = create_sample(main_cut, catalog_data, is_binned = False)
        
        # create multiple bright samples         
        bright_cut = 'extendedness == 0'
        bright_sample, nBbin = create_sample(bright_cut, catalog_data, is_binned = True)
                    
        ## calculate nearest neighbors
        distB = calc_nearest_neighbors(main_sample, bright_sample)

        ## plot results        
        hist,histerr = [],[]
        area = np.zeros((self.nbins))
        colors = ['blue', 'green', 'yellow', 'orange', 'red', 'black']

        bins,step = np.linspace(0., self.distance, self.nbins+1, retstep = True)
        midbins = bins+0.5
        midbins = midbins[0:-1]

        plt.figure()
        for i in range(len(self.cut)):
            d, derr, cnt = np.zeros((self.nbins)), np.zeros((self.nbins)), np.zeros((self.nbins))
            for j in range(nBbin[i]):
                tmphist, _ = np.histogram(distB[i][j], bins)
                d += tmphist
                derr += tmphist
                cnt += 1
            for k in range(self.nbins):
                area[k] = ring(k+1,_[k],_[k+1]) # i=0 will give you funny results as area() is defined
            d = d/cnt/area
            derr = np.sqrt(derr)/cnt/area
            hist.append(d/d[nbins-1])
            histerr.append(derr/d[nbins-1])
            plt.errorbar(midbins, hist[i], yerr = histerr[i], 
                         fmt = 'o', label = self.cutname+str(cut[i]), color = colors[i])

        #self.post_process_plot(ax)
        plt.xlim(1,20)
        plt.ylim(0,1.1)
        plt.ylabel('Relative abundance of extended sources')
        plt.xlabel('Distance from bright star (arcsec)') 
        plt.legend()
        fig.savefig(os.path.join(output_dir, 'obscuration.png'))
        plt.close(fig)

        #score = data[0] #calculate your summary statistics
        return TestResult(score = 1, passed=True) # TBD

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir)
        #self.post#_process_plot(self.summary_ax)
        #self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        #plt.close(self.summary_fig)
