#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import os
import numpy as np
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, CustomJS, Line
from bokeh.layouts import column, gridplot, row
from bokeh.models.widgets import Select


def get_quantity_labels(bands):
    quantities = ['ra','dec','ra_MATCH', 'dec_MATCH']#,'detect_isIsolated']
    for band in ['g','r','i','u','z','y']:
        for model in ['cModel','calib','gaap']:
            quantities.append('mag_'+band+'_'+model)
            quantities.append(model+'Flux_flag_'+band)
        quantities.append('mag_'+band+'_MATCH')
    return quantities 

def run_test(gc_data, outdir):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    #mask = ~gc_data['ra_MATCH'].mask
    ra = gc_data['ra']#[mask]
    dec = gc_data['dec']#[mask]
    ra_truth = gc_data['ra_MATCH']#.compressed()
    dec_truth = gc_data['dec_MATCH']#.compressed()

    mask_data = {}
    for model in ['cModel','calib','gaap']:
        mask_data[model] = np.zeros_like(ra,dtype=np.bool_)
        for bandpass in bands:
            mask_data[model] += gc_data[model+'Flux_flag_'+bandpass]
        mask_data[model] = np.logical_not(mask_data[model])


    # Create the histograms
    band_hists = {}
    for bandpass in bands:
        hist1, edges1 = np.histogramdd([gc_data['mag_'+bandpass+'_cModel'][mask_data['cModel']], gc_data['mag_'+bandpass+'_MATCH'][mask_data['cModel']]],bins=[np.linspace(22,28,200),np.linspace(22,28,200)])
        hist2, edges2 = np.histogramdd([gc_data['mag_'+bandpass+'_gaap'][mask_data['cModel']], gc_data['mag_'+bandpass+'_MATCH'][mask_data['cModel']]],bins=[np.linspace(22,28,200),np.linspace(22,28,200)])
        band_hists[bandpass] = [hist1, edges1, hist2, edges2]

    dataset_names={}
    dataset_names_reverse={}
    for i in range(6):
        dataset_names['Dataset '+str(i+1)] = bands[i]
        dataset_names_reverse[bands[i]] = 'Dataset '+str(i+1)

    #dataset_names = {'Dataset 1':'u', 'Dataset 2':'g'}
    #dataset_names_reverse = {'cModel':'Dataset 1','gaap':'Dataset 2'}

    # Define the histogram centers for plotting
    x_centers1 = (edges1[0][1:] + edges1[0][:-1]) / 2
    y_centers1 = (edges1[1][1:] + edges1[1][:-1]) / 2
    x_centers2 = (edges2[0][1:] + edges2[0][:-1]) / 2
    y_centers2 = (edges2[1][1:] + edges2[1][:-1]) / 2


    # Line Data 
    xLineRange = np.linspace(22,28,100)
    yLineRange = np.linspace(22,28,100)
    sourceLine = ColumnDataSource(data=dict(x=xLineRange,y=yLineRange))
    glyph = Line(x="x", y = "y", line_color = "#f46d43", line_width=2, line_alpha=0.6)

    # Create a ColumnDataSource for each histogram
    sources = {}
    for bandpass in bands:
        source1 = ColumnDataSource(data=dict(image=[band_hists[bandpass][0]], x=[x_centers1[0]], y=[y_centers1[0]], dw=[x_centers1[-1] - x_centers1[0]], dh=[y_centers1[-1] - y_centers1[0]]))
        source2 = ColumnDataSource(data=dict(image=[band_hists[bandpass][2]], x=[x_centers2[0]], y=[y_centers2[0]], dw=[x_centers2[-1] - x_centers2[0]], dh=[y_centers2[-1] - y_centers2[0]]))
        sources[bandpass] = [source1,source2]

    # Create the plot
    plot1 = figure(title='cModel', width=500, height=400)#, x_range=(54.5, 56.5), y_range=(-29.8, -28.2))
    images1 = []
    for bandpass in bands:
        images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=sources[bandpass][0], palette='Viridis11'))
    #images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source2, palette='Viridis11'))
    #images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source3, palette='Viridis11'))

    plot2 = figure(title='gaap', width=500, height=400, x_range=plot1.x_range, y_range=plot1.y_range)
    #plot2 = figure(title='cModelFlux_flag_u', width=500, height=400)#, x_range=plot1.x_range, y_range=plot1.y_range)
    images2 = []
    for bandpass in bands:
        #images2plot2.image('image', 'x', 'y', 'dw', 'dh', source=sources[bandpass][1], palette='Viridis11')
        ##image_tmp = plot2.line(np.linspace(22,30,100),np.linspace(22,30,100),line_color='black',line_width=0.5,line_dash='dashed')
        #images2.append(image_tmp)
        images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=sources[bandpass][1], palette='Viridis11'))
    #images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source2, palette='Viridis11'))
    #images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source3, palette='Viridis11'))

    for i in range(1, len(images1)):
        images1[i].visible = False
        images2[i].visible = False

    # Get the maximum value among all histograms
    max_value = -99999
    for bandpass in bands:
        max_value = max(np.max(band_hists[bandpass][0]), max_value)
        max_value = max(np.max(band_hists[bandpass][2]), max_value)


    # Set the same colorbar range for all images
    for image in images1:
        image.glyph.color_mapper.high = max_value
 
    for image in images2:
        image.glyph.color_mapper.high = max_value
    
    color_bar_1 = images1[0].construct_color_bar()
    plot1.add_layout(color_bar_1, 'right')
    color_bar_2 = images2[0].construct_color_bar()
    plot2.add_layout(color_bar_2, 'right')

    plot1.add_glyph(sourceLine,glyph)
    plot2.add_glyph(sourceLine,glyph)
    #for i in range(len(images1)):
    #    images2[i] = plot2.line(np.linspace(22,28,100),np.linspace(22,28,100),line_color='black',line_width=0.5,line_dash='dashed')


    # Create the dropdown menu
    dropdown1 = Select(title='Dataset', 
                  value=dataset_names['Dataset 1'], 
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6']])
    dropdown2 = Select(title='Dataset', 
                  value=dataset_names['Dataset 1'], 
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6']])

    # Create a JavaScript callback function for the dropdown menu
    callback = CustomJS(args=dict(plot1=plot1, 
                              plot2=plot2,
                              dataset_names=dataset_names, 
                              dataset_names_reverse=dataset_names_reverse, 
                              sources=[sources[bands[0]][0], sources[bands[0]][1]], 
                              images1=images1,
                              images2=images2,
                              dropdown1=dropdown1,
                              dropdown2=dropdown2), 
                    code="""

    const selectedDataset1 = dropdown1.value;
    for (let i = 0; i < images1.length; i++) {
        if (selectedDataset1 === dataset_names[`Dataset ${i + 1}`]) {
            images1[i].visible = true;
            plot1.title.text = dataset_names[`Dataset ${i + 1}`];
        } else {
            images1[i].visible = false;
        }
    }
    

    const selectedDataset2 = dropdown2.value;
    for (let i = 0; i < images2.length; i++) {
        if (selectedDataset2 === dataset_names[`Dataset ${i + 1}`]) {
            images2[i].visible = true;
            plot2.title.text = dataset_names[`Dataset ${i + 1}`];
        } else {
            images2[i].visible = false;
        }
    }
    """)

    dropdown1.js_on_change('value', callback)
    dropdown2.js_on_change('value', callback)

    # Display the plot and dropdown menu
    layout = column(row(dropdown1, dropdown2), row(plot1, plot2))

    filename = outdir+'/truth_match.html'
    output_file(filename=filename, title="truth_match")
    save(layout)

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0 
    passed = True

    return test_result, passed 



