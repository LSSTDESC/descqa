"""
configuration routines for analysis tools
"""
from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np

from lsst.analysis.tools.tasks.base import _StandinPlotInfo
from lsst.analysis.tools.actions.vector import SnSelector, MagColumnNanoJansky, MagDiff
from lsst.analysis.tools.analysisPlots.analysisPlots import ( 
    WPerpPSFPlot, 
    ShapeSizeFractionalDiffScatterPlot,
    E1DiffScatterPlot,
    E2DiffScatterPlot,
)
from lsst.analysis.tools.analysisMetrics import ( 
    WPerpPSFMetric,
    ShapeSizeFractionalMetric,
    E1DiffMetric,
    E2DiffMetric,
)
from .plotting import plt


__all__ = [
    "shapeSizeFractional",
    "E1Diff",
    "E2Diff",
    "WPerpPSF",
    ]

class analysisToolsReconfigure():
    return_types=["plot","metric"] # add metric names? 
    band="default"
    metric={}
    plot={}
    key_list=[]
    plot_name="defaultName.png"

    def get_keys(self):
        key_list_metric = list(self.metric.prep.getInputSchema())
        key_list = [key_list_metric[i][0] for i in range(len(key_list_metric))]
        key_list_plot = list(self.plot.prep.getInputSchema())
        key_list2 = [key_list_plot[i][0] for i in range(len(key_list_plot))]
        key_list.extend(key_list2)
        self.key_list=key_list


    def run(self,data,output_dir,metric=True, plot=True):
        # run analysis_tools metric and plot code 
        if metric:
            self.metric_values = self.metric(data,band=self.band)
            for key in self.metric_values.keys():
                print(self.metric_values[key])
        if plot: 
            stage1 = self.plot.prep(data,band=self.band)
            stage2 = self.plot.process(data,band=self.band)
            plot = self.plot.produce(stage2, plotInfo=_StandinPlotInfo(), band=self.band,skymap='DC2')
            plt.savefig(output_dir+self.plot_name)
            plt.close()

class shapeSizeFractional(analysisToolsReconfigure):
    "Reconfig class for shapeSizeFractional metric from analysis tools"
    #def __init__(self):
    metric=ShapeSizeFractionalMetric()
    plot=ShapeSizeFractionalDiffScatterPlot()
    plot_name="shapeSizeFractional.png"
    def reconfigure(self,band="r"):
        ''' Call analysis tools shapesize and reconfigure for DP0.2'''
        self.band=band

        # custom config 
        self.plot.produce.addSummaryPlot = False
        self.plot.prep.selectors.snSelector.bands = self.band
        
    
        # populate prep
        self.metric.populatePrepFromProcess()
        self.plot.populatePrepFromProcess()

        # get list of quantities
        self.get_keys()

        
class E1Diff(analysisToolsReconfigure):
    "Reconfig class for E1Diff metric from analysis tools"
    #def __init__(self):
    metric=E1DiffMetric()
    plot=E1DiffScatterPlot()
    plot_name="E1Diff.png"
    def reconfigure(self,band="r"):
        ''' Call analysis tools shapesize and reconfigure for DP0.2'''
        self.band=band

        # custom config 
        self.plot.produce.addSummaryPlot = False
        self.plot.prep.selectors.snSelector.bands = self.band
        
    
        # populate prep
        self.metric.populatePrepFromProcess()
        self.plot.populatePrepFromProcess()

        # get list of quantities
        self.get_keys()
        

class E2Diff(analysisToolsReconfigure):
    "Reconfig class for E2diff metric from analysis tools"
    metric=E1DiffMetric()
    plot=E1DiffScatterPlot()
    plot_name="E1Diff.png"
    def reconfigure(self,band="r"):
        ''' Call analysis tools shapesize and reconfigure for DP0.2'''
        self.band=band

        # custom config 
        self.plot.produce.addSummaryPlot = False
        self.plot.prep.selectors.snSelector.bands = self.band
        
    
        # populate prep
        self.metric.populatePrepFromProcess()
        self.plot.populatePrepFromProcess()

        # get list of quantities
        self.get_keys()

class WPerpPSF(analysisToolsReconfigure):
    """Reconfig class for WPerpPsf metric from analysis tools
    of note we need to replace ExtinctionCorrectedMagDiff with MagDiff 
    for DC2 data

    Also make sure config.yaml contains bands: ['g','r','i']
    """
    metric=WPerpPSFMetric()
    plot=WPerpPSFPlot()
    plot_name="WPerpPSF.png"
    def reconfigure(self,band="r"):
        ''' Call analysis tools WPerpPSF and reconfigure for DP0.2'''
        self.band=band

        # custom config 
        self.plot.prep.selectors.flagSelector.bands=["g","r","i"]
        self.plot.prep.selectors.snSelector.bands=[self.band]
        self.plot.prep.selectors.snSelector.fluxType="{band}_psfFlux"

        self.plot.process.buildActions.x = MagDiff()
        self.plot.process.buildActions.x.col1 = "g_psfFlux"
        self.plot.process.buildActions.x.col2 = "r_psfFlux"
        self.plot.process.buildActions.x.returnMillimags=False
        self.plot.process.buildActions.y = MagDiff()
        self.plot.process.buildActions.y.col1 = "r_psfFlux"
        self.plot.process.buildActions.y.col2 = "i_psfFlux"
        self.plot.process.buildActions.y.returnMillimags=False
        
        self.metric.prep.selectors.snSelector.bands=[self.band]
        self.metric.prep.selectors.snSelector.fluxType="{band}_psfFlux"

        self.metric.process.buildActions.x = MagDiff()
        self.metric.process.buildActions.x.col1 = "g_psfFlux"
        self.metric.process.buildActions.x.col2 = "r_psfFlux"
        self.metric.process.buildActions.x.returnMillimags=False
        self.metric.process.buildActions.y = MagDiff()
        self.metric.process.buildActions.y.col1 = "r_psfFlux"
        self.metric.process.buildActions.y.col2 = "i_psfFlux"
        self.metric.process.buildActions.y.returnMillimags=False
    
        # populate prep
        self.metric.populatePrepFromProcess()
        self.plot.populatePrepFromProcess()

        # get list of quantities
        self.get_keys()