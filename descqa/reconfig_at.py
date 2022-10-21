"""
configuration routines for analysis tools
"""
from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
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

__all__ = []


def reconfigured_shapesize(band):
    ''' Call analysis tools wperpPSFPlot and reconfigure for DP0.2'''
    # call base class
    shapesize = ShapeSizeFractionalMetric()
    shapesize_plot = ShapeSizeFractionalDiffScatterPlot()
    shapesize_plot.produce.addSummaryPlot = False
    # config 
    shapesize_plot.prep.selectors.snSelector.bands = band

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

    return shapesize, shapesize_plot, key_list

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
