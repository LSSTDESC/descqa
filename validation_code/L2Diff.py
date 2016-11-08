import numpy as np

def L2Diff(dataset_1, dataset_2, threshold=1.0):
    """
    compute the L2 norm for 2 datasets
    dataset_1 the catalog data to be validated against
    dataset_2 the validation data 
    checks the x values to see if interpolation is needed
    threshold sets criterion for passing the test
    """
    x1     = dataset_1['x']
    y1     = dataset_1['y']
    if(dataset_1.has_key('y+') and dataset_1.has_key('y-')):
        e1 = (dataset_1['y+']-dataset_1['y-'])/2.
    elif(dataset_1.has_key('dy')):
        e1 = dataset_1['dy']
    else:
        e1 = None
    x2     = dataset_2['x']
    y2     = dataset_2['y']
    if(dataset_2.has_key('y+') and dataset_2.has_key('y-')):
        e2 = (dataset_2['y+']-dataset_2['y-'])/2.
    elif(dataset_2.has_key('dy')):
        e2 = dataset_2['dy']
    else:
        e2 = None

    # Ensure ranges of catalog and validation data are the same
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

    # Interpolate catalog data to data x-points and compute L2 norm and significance

    if not(np.all(x1==x2)):
        y1int = np.interp(x2, x1, y1)
        if e1 is not None:
            e1int = np.interp(x2, x1, e1)
    else:
        y1int=y1
        if e1 is not None:
            e1int = e1

    if e1 is not None:
        if e2 is not None:
            L2 = (np.sum( (y2 - y1int)**2 / (e1int**2 + e2**2) ))**0.5
        else:
            L2 = (np.sum( (y2 - y1int)**2 / e1int**2 ))**0.5
    else:
        if e2 is not None:
            L2 = (np.sum( (y2 - y1int)**2 / e2**2 ))**0.5
        else:
            L2 = (np.sum(y2 - y1int)**2 )**0.5

    # Issue verdict
    if (L2 > threshold) or (np.isnan(L2)):
        success=False
    else:
        success = True
    return L2, success
