#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import os
import numpy as np
import bokeh
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import column, gridplot, row
from bokeh.models.widgets import Select


def get_quantity_labels(bands):
    quantities = ['ra','dec']#,'detect_isIsolated']
    for band in ['g','r','i','u','z','y']:
        quantities.append('cModelFlux_flag_'+band)
    return quantities 

def run_test(gc_data, outdir):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    ra = gc_data['ra']
    dec = gc_data['dec']
    # Create the histograms
    # for bandpass in ['u', 'g', 'r', 'i', 'z', 'y']:
    hist1, edges1 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_i'])], dec[np.logical_not(gc_data['cModelFlux_flag_i'])]],bins=2000)#, range=((54.5, 56.5), (-29.8, -28.2)))
    hist2, edges2 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_g'])], dec[np.logical_not(gc_data['cModelFlux_flag_g'])]],bins=2000)#, range=((54.5, 56.5), (-29.8, -28.2)))
    hist3, edges3 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_r'])], dec[np.logical_not(gc_data['cModelFlux_flag_r'])]],bins=200)#, range=((54.5, 56.5), (-29.8, -28.2)))
    #hist3, edges3 = np.histogramdd([ra[np.logical_not(gc_data['detect_isIsolated'])], dec[np.logical_not(gc_data['detect_isIsolated'])]],bins=2000)#, range=((54.5, 56.5), (-29.8, -28.2)))


    dataset_names = {'Dataset 1':'cModelFlux_flag_i', 'Dataset 2':'cModelFlux_flag_g', 'Dataset 3':'detect_isIsolated'}
    dataset_names_reverse = {'cModelFlux_flag_i':'Dataset 1','cModelFlux_flag_g':'Dataset 2','detect_isIsolated':'Dataset 3'}

    # Define the histogram centers for plotting
    x_centers1 = (edges1[0][1:] + edges1[0][:-1]) / 2
    y_centers1 = (edges1[1][1:] + edges1[1][:-1]) / 2
    x_centers2 = (edges2[0][1:] + edges2[0][:-1]) / 2
    y_centers2 = (edges2[1][1:] + edges2[1][:-1]) / 2
    x_centers3 = (edges3[0][1:] + edges3[0][:-1]) / 2
    y_centers3 = (edges3[1][1:] + edges3[1][:-1]) / 2

    # Create a ColumnDataSource for each histogram
    source1 = ColumnDataSource(data=dict(image=[hist1], x=[x_centers1[0]], y=[y_centers1[0]], dw=[x_centers1[-1] - x_centers1[0]], dh=[y_centers1[-1] - y_centers1[0]]))
    source2 = ColumnDataSource(data=dict(image=[hist2], x=[x_centers2[0]], y=[y_centers2[0]], dw=[x_centers2[-1] - x_centers2[0]], dh=[y_centers2[-1] - y_centers2[0]]))
    source3 = ColumnDataSource(data=dict(image=[hist3], x=[x_centers3[0]], y=[y_centers3[0]], dw=[x_centers3[-1] - x_centers3[0]], dh=[y_centers3[-1] - y_centers3[0]]))

    # Create the plot
    plot1 = figure(title='cModelFlux_flag_i', width=500, height=400)#, x_range=(54.5, 56.5), y_range=(-29.8, -28.2))
    images1 = []
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source1, palette='Viridis11'))
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source2, palette='Viridis11'))
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source3, palette='Viridis11'))

    plot2 = figure(title='cModelFlux_flag_i', width=500, height=400, x_range=plot1.x_range, y_range=plot1.y_range)
    #plot2 = figure(title='cModelFlux_flag_u', width=500, height=400)#, x_range=plot1.x_range, y_range=plot1.y_range)
    images2 = []
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source1, palette='Viridis11'))
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source2, palette='Viridis11'))
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source3, palette='Viridis11'))

    for i in range(1, len(images1)):
        images1[i].visible = False
        images2[i].visible = False

    # Get the maximum value among all histograms
    max_value = max(np.max(hist1), np.max(hist2), np.max(hist3))

    # Set the same colorbar range for all images
    for image in images1:
        image.glyph.color_mapper.high = max_value
 
    for image in images2:
        image.glyph.color_mapper.high = max_value
    
    color_bar_1 = images1[0].construct_color_bar()
    plot1.add_layout(color_bar_1, 'right')
    color_bar_2 = images2[0].construct_color_bar()
    plot2.add_layout(color_bar_2, 'right')

    # Create the dropdown menu
    dropdown1 = Select(title='Dataset', 
                  value=dataset_names['Dataset 1'], 
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2', 'Dataset 3']])
    dropdown2 = Select(title='Dataset', 
                  value=dataset_names['Dataset 1'], 
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2', 'Dataset 3']])

    # Create a JavaScript callback function for the dropdown menu
    callback = CustomJS(args=dict(plot1=plot1, 
                              plot2=plot2,
                              dataset_names=dataset_names, 
                              dataset_names_reverse=dataset_names_reverse, 
                              sources=[source1, source2, source3], 
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

    filename = outdir+'/cModelFlux_flag_density.html'
    output_file(filename=filename, title="cModelFlux_flag_density")
    save(layout)

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0 
    passed = True

    return test_result, passed 



def run_test_double(gc_data, gc_data2, outdir):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    ra = gc_data['ra']
    dec = gc_data['dec']
    ra2 = gc_data2['ra']
    dec2 = gc_data2['dec']

    bins = 5000

    # first catalog 
    hist1, edges1 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_i'])], dec[np.logical_not(gc_data['cModelFlux_flag_i'])]],bins=bins)
    hist2, edges2 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_g'])], dec[np.logical_not(gc_data['cModelFlux_flag_g'])]],bins=bins)
    hist3, edges3 = np.histogramdd([ra[np.logical_not(gc_data['cModelFlux_flag_r'])], dec[np.logical_not(gc_data['cModelFlux_flag_r'])]],bins=bins)

    # second catalog 
    hist1b, edges1 = np.histogramdd([ra2[np.logical_not(gc_data2['cModelFlux_flag_i'])], dec2[np.logical_not(gc_data2['cModelFlux_flag_i'])]],bins=edges1)
    hist2b, edges2 = np.histogramdd([ra2[np.logical_not(gc_data2['cModelFlux_flag_g'])], dec2[np.logical_not(gc_data2['cModelFlux_flag_g'])]],bins=edges2)
    hist3b, edges3 = np.histogramdd([ra2[np.logical_not(gc_data2['cModelFlux_flag_r'])], dec2[np.logical_not(gc_data2['cModelFlux_flag_r'])]],bins=edges3)


    dataset_names = {'Dataset 1':'cModelFlux_flag_i', 'Dataset 2':'cModelFlux_flag_g', 'Dataset 3':'cModelFlux_flag_r'}
    dataset_names_reverse = {'cModelFlux_flag_i':'Dataset 1','cModelFlux_flag_g':'Dataset 2','cModelFlux_flag_r':'Dataset 3'}

    # Define the histogram centers for plotting
    x_centers1 = (edges1[0][1:] + edges1[0][:-1]) / 2
    y_centers1 = (edges1[1][1:] + edges1[1][:-1]) / 2
    x_centers2 = (edges2[0][1:] + edges2[0][:-1]) / 2
    y_centers2 = (edges2[1][1:] + edges2[1][:-1]) / 2
    x_centers3 = (edges3[0][1:] + edges3[0][:-1]) / 2
    y_centers3 = (edges3[1][1:] + edges3[1][:-1]) / 2

    # Create a ColumnDataSource for each histogram
    source1 = ColumnDataSource(data=dict(image=[hist1], x=[x_centers1[0]], y=[y_centers1[0]], dw=[x_centers1[-1] - x_centers1[0]], dh=[y_centers1[-1] - y_centers1[0]]))
    source2 = ColumnDataSource(data=dict(image=[hist2], x=[x_centers2[0]], y=[y_centers2[0]], dw=[x_centers2[-1] - x_centers2[0]], dh=[y_centers2[-1] - y_centers2[0]]))
    source3 = ColumnDataSource(data=dict(image=[hist3], x=[x_centers3[0]], y=[y_centers3[0]], dw=[x_centers3[-1] - x_centers3[0]], dh=[y_centers3[-1] - y_centers3[0]]))
    source1b = ColumnDataSource(data=dict(image=[hist1b], x=[x_centers1[0]], y=[y_centers1[0]], dw=[x_centers1[-1] - x_centers1[0]], dh=[y_centers1[-1] - y_centers1[0]]))
    source2b = ColumnDataSource(data=dict(image=[hist2b], x=[x_centers2[0]], y=[y_centers2[0]], dw=[x_centers2[-1] - x_centers2[0]], dh=[y_centers2[-1] - y_centers2[0]]))
    source3b = ColumnDataSource(data=dict(image=[hist3b], x=[x_centers3[0]], y=[y_centers3[0]], dw=[x_centers3[-1] - x_centers3[0]], dh=[y_centers3[-1] - y_centers3[0]]))



    # Create the plot
    plot1 = figure(title='cModelFlux_flag_i', width=500, height=400)#, x_range=(54.5, 56.5), y_range=(-29.8, -28.2))
    images1 = []
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source1, palette='Viridis11'))
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source2, palette='Viridis11'))
    images1.append(plot1.image('image', 'x', 'y', 'dw', 'dh', source=source3, palette='Viridis11'))

    plot2 = figure(title='cModelFlux_flag_i', width=500, height=400, x_range=plot1.x_range, y_range=plot1.y_range)
    #plot2 = figure(title='cModelFlux_flag_u', width=500, height=400)#, x_range=plot1.x_range, y_range=plot1.y_range)
    images2 = []
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source1b, palette='Viridis11'))
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source2b, palette='Viridis11'))
    images2.append(plot2.image('image', 'x', 'y', 'dw', 'dh', source=source3b, palette='Viridis11'))

    for i in range(1, len(images1)):
        images1[i].visible = False
        images2[i].visible = False
    # Get the maximum value among all histograms
    max_value = max(np.max(hist1), np.max(hist2), np.max(hist3),np.max(hist1b),np.max(hist2b),np.max(hist3b))

    # Set the same colorbar range for all images
    for image in images1:
        image.glyph.color_mapper.high = max_value

    for image in images2:
        image.glyph.color_mapper.high = max_value

    color_bar_1 = images1[0].construct_color_bar()
    plot1.add_layout(color_bar_1, 'right')
    color_bar_2 = images2[0].construct_color_bar()
    plot2.add_layout(color_bar_2, 'right')

    # Create the dropdown menu
    dropdown1 = Select(title='Dataset',
                  value=dataset_names['Dataset 1'],
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2', 'Dataset 3']])
    dropdown2 = Select(title='Dataset',
                  value=dataset_names['Dataset 1'],
                  options=[dataset_names[x] for x in ['Dataset 1', 'Dataset 2', 'Dataset 3']])

    # Create a JavaScript callback function for the dropdown menu
    callback = CustomJS(args=dict(plot1=plot1,
                              plot2=plot2,
                              dataset_names=dataset_names,
                              dataset_names_reverse=dataset_names_reverse,
                              sources=[source1, source2, source3],
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

    filename = outdir+'/cModelFlux_flag_density.html'
    output_file(filename=filename, title="cModelFlux_flag_density")
    save(layout)

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0
    passed = True

    return test_result, passed


