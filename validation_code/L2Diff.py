import numpy as np

def L2Diff(dataset_1, dataset_2, threshold=1.0):
    """
    compute the L2 norm for 2 datasets
    
    checks the x values to see if interpolation is needed
    threshold sets criterion for passing the test
    
    Parameters
    ----------
    dataset_1 : dictionary
        data to be validated against
        
        x : independent variable
        
        y : dependent variable
        
        y+ : asymmetric positive error on y
        
        y- : asymmetric negative error on y
        
        dy : 1-sigma symmetric error on y
        
    dataset_2 : dictionary
        validation data 
    
    threshold : float, optional
    
    Returns
    -------
    L2 : float
    
    sucess : boolean
    
    Notes
    -----
    Errors are assumed to be un-correlated.
    
    If assymetric errors on `y` are provided, the mean of the postive and 
    negative errors are taken. This is clearly not the 'correct' thing to do.
    """
    
    #clean up catalog 1 data to remove nans and infs
    mask=np.isfinite(np.vstack(dataset_1.values())).all(axis=0)
    x1 = dataset_1['x'][mask]
    y1 = dataset_1['y'][mask]
    #extract errors if they exist
    if(dataset_1.has_key('y+') and dataset_1.has_key('y-')):
        e1 = (dataset_1['y+'][mask]-dataset_1['y-'][mask])/2.
    elif(dataset_1.has_key('dy')):
        e1 = dataset_1['dy'][mask]
    else:
        e1 = None
    
    #clean up catalog 2 data to remove nans and infs
    mask=np.isfinite(np.vstack(dataset_2.values())).all(axis=0)
    x2 = dataset_2['x'][mask]
    y2 = dataset_2['y'][mask]
    #extract errors if they exist
    if(dataset_2.has_key('y+') and dataset_2.has_key('y-')):
        e2 = (dataset_2['y+'][mask]-dataset_2['y-'][mask])/2.
    elif(dataset_2.has_key('dy')):
        e2 = dataset_2['dy'][mask]
    else:
        e2 = None
    
    #ensure ranges of catalog and validation data are the same
    minx=max(np.min(x1),np.min(x2))
    maxx=min(np.max(x1),np.max(x2))
    select1=(x1>=minx) & (x1<=maxx)
    select2=(x2>=minx) & (x2<=maxx)
    x1=x1[select1]
    x2=x2[select2]
    y1=y1[select1]
    y2=y2[select2]
    if e1 is not None:
        e1=e1[select1]
    if e2 is not None:
        e2=e2[select2]
    
    #interpolate catalog data to data x-points
    if not(np.all(x1==x2)):
        y1int = np.interp(x2, x1, y1)
        if e1 is not None:
            e1int = np.interp(x2, x1, e1)
    else:
        y1int=y1
        if e1 is not None:
            e1int = e1
    
    #compute L2 norm and significance
    if e1 is not None:
        if e2 is not None:
            L2 = (np.sum( (y2 - y1int)**2 / (e1int**2 + e2**2) ))
        else:
            L2 = (np.sum( (y2 - y1int)**2 / e1int**2 ))
    else:
        if e2 is not None:
            L2 = (np.sum( (y2 - y1int)**2 / e2**2 ))
        else:
            L2 = (np.sum(y2 - y1int)**2 )
    
    #normalize by the number of points
    N_points = 1.0*len(y2)
    L2 = np.sqrt(L2/N_points)
    
    #return result
    if (L2 > threshold) or (np.isnan(L2)):
        success = False
    else:
        success = True
    
    return L2, success
    
    
