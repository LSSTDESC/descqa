def L2Diff(dataset_1, dataset_2, threshold=1.0):

    xcol1     = dataset_1['x']
    ycol1     = dataset_1['theory_ycolumn']
    ecol1     = dataset_1['theory_ecolumn']
    xcol2     = dataset_2['data_xcolumn']
    ycol2     = dataset_2['data_ycolumn']
    ecol2     = dataset_2['data_ecolumn']

    # Limit comparison to "valid" range specified in files

    ok1 = np.where((x1 >= v1[0]) & (x1 <= v1[1]))
    x1  = x1[ok1]
    y1  = y1[ok1]
    if ecol1:
        e1 = e1[ok1]

    ok2 = np.where((x2 >= v2[0]) & (x2 <= v2[1]))
    x2  = x2[ok2]
    y2  = y2[ok2]
    if ecol2:
        e2 = e2[ok2]

    # Interpolate theory curve to data x-points and compute L2 norm and significance

    y1int = np.interp(x2, x1, y1)

    if ecol1:
        e1int = np.interp(x2, x1, e1)
        if ecol2:
            L2 = (np.sum( (y2 - y1int)**2 / (e1int**2 + e2**2) ))**0.5
        else:
            L2 = (np.sum( (y2 - y1int)**2 / e1int**2 ))**0.5
    else:
        if ecol2:
            L2 = (np.sum( (y2 - y1int)**2 / e2**2 ))**0.5
        else:
            L2 = (np.sum( (y2 - y1int)**2 ))**0.5

    print "L2 = %G" % L2

    # Issue verdict

    if (L2 > threshold) or (np.isnan(L2)):
        print "Almost! But you need to get to L2 = %G"%(threshold)
    else:
        print "SUCCESS"
