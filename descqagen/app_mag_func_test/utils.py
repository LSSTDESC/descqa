
import numpy as np

__all__ = ['get_ordered_list']

__author__=['Ducnan Campbell']

def get_ordered_list(data_cols):
    """
    return an ordered list from column dtypes dict
    """
    names = np.array(data_cols.keys())
    order = np.zeros(len(names)).astype('int')
    column_types = []
    for i in range(0,len(names)):
        order[i] = data_cols[names[i]][0]
        column_types.append(data_cols[names[i]][1])
    sort_inds = np.argsort(order)
    column_types = np.array(column_types)
    names = names[sort_inds]
    order = order[sort_inds]
    column_types = column_types[sort_inds]
    data_dtypes = [(x,np.dtype(y).type) for x, y in zip(names, column_types)]
    return data_dtypes