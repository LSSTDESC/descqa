"""
configuration routines for analysis tools
"""
from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np

from lsst.analysis.tools.tasks.base import _StandinPlotInfo
from lsst.analysis.tools.analysisPlots.analysisPlots import ( 
    WPerpPSFPlot, 
    ShapeSizeFractionalDiffScatterPlot,
    E1DiffScatterPlot,
    E2DiffScatterPlot,
)
from lsst.analysis.tools.analysisMetrics import ( 
    ShapeSizeFractionalMetric,
    E1DiffMetric,
    E2DiffMetric,
)
from .plotting import plt


__all__ = ["shapeSizeFractional"]

class shapeSizeFractional():
    return_types=["plot","metric"] # add metric names? 
    band="r"
    metric={}
    plot={}
    key_list=[]
    def reconfigure(self,band="r"):
        ''' Call analysis tools shapesize and reconfigure for DP0.2'''
        self.band=band
        # call base class
        shapesize = ShapeSizeFractionalMetric()
        shapesize_plot = ShapeSizeFractionalDiffScatterPlot()
        shapesize_plot.produce.addSummaryPlot = False
        # config 
        shapesize_plot.prep.selectors.snSelector.bands = self.band

        #shapesize_plot.prep.bands=['r']
    
        # populate prep
        shapesize.populatePrepFromProcess()
        shapesize_plot.populatePrepFromProcess()

        # get list of quantities
        key_list_full = list(shapesize.prep.getInputSchema())
        key_list = [key_list_full[i][0] for i in range(len(key_list_full))]
        key_list_full2 = list(shapesize_plot.prep.getInputSchema())
        key_list2 = [key_list_full2[i][0] for i in range(len(key_list_full2))]
        key_list.extend(key_list2)

        self.metric=shapesize
        self.plot=shapesize_plot
        self.key_list=key_list

    def run(self,data,output_dir,metric=True, plot=True):
        if metric:
            self.metric_values = self.metric(data,band=self.band)
            for key in self.metric_values.keys():
                print(self.metric_values[key])
        if plot: 
            stage1 = self.plot.prep(data,band=self.band)
            stage2 = self.plot.process(data,band=self.band)
            plot = self.plot.produce(stage2, plotInfo=_StandinPlotInfo(), band=self.band,skymap='DC2')
            plt.savefig(output_dir+"shapesizeTest.png")
            plt.close()
      


def reconfigured_E1Diff(band):
    ''' Call analysis tools wperpPSFPlot and reconfigure for DP0.2'''
    # call base class
    e1diff = E1DiffMetric()
    e1diff_plot = E1DiffScatterPlot()
    e1diff_plot.produce.addSummaryPlot = False
    # config 
    e1diff_plot.prep.selectors.snSelector.bands = band

    #shapesize_plot.prep.bands=['r']
 
    # populate prep
    e1diff.populatePrepFromProcess()
    e1diff_plot.populatePrepFromProcess()


    # get list of quantities
    key_list_full = list(e1diff.prep.getInputSchema())
    key_list = [key_list_full[i][0] for i in range(len(key_list_full))]
    key_list_full2 = list(e1diff_plot.prep.getInputSchema())
    key_list2 = [key_list_full2[i][0] for i in range(len(key_list_full2))]
    key_list.extend(key_list2)

    return e1diff, e1diff_plot, key_list

def reconfigured_E2Diff(band):
    ''' Call analysis tools wperpPSFPlot and reconfigure for DP0.2'''
    # call base class
    e2diff = E2DiffMetric()
    e2diff_plot = E2DiffScatterPlot()
    e2diff_plot.produce.addSummaryPlot = False
    # config 
    e2diff_plot.prep.selectors.snSelector.bands = band

    #shapesize_plot.prep.bands=['r']
 
    # populate prep
    e2diff.populatePrepFromProcess()
    e2diff_plot.populatePrepFromProcess()


    # get list of quantities
    key_list_full = list(e2diff.prep.getInputSchema())
    key_list = [key_list_full[i][0] for i in range(len(key_list_full))]
    key_list_full2 = list(e2diff_plot.prep.getInputSchema())
    key_list2 = [key_list_full2[i][0] for i in range(len(key_list_full2))]
    key_list.extend(key_list2)

    return e2diff, e2diff_plot, key_list
