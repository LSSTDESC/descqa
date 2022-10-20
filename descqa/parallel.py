"""
MPI-related utility functions for descqa
"""
from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = [
    'send_to_master',
    'get_ra_dec',
]


def send_to_master(value, kind):
    """
    Parameters
    ----------
    value : ndarray
        rank-local array of values to communicate
    kind : str
        type of variable. Currently implemented options are double, bool 

    Returns
    -------
    recvbuf : ndarray
        array of all rank values for rank 0 
        None value otherwise
    """
    count = len(value)
    tot_num = comm.reduce(count)
    counts = comm.allgather(count)

    if rank==0:
        if kind=='double':
            recvbuf = np.zeros(tot_num)
        elif kind=='bool':
            recvbuf = np.zeros(tot_num)!=0.0
        elif kind=='int':
            recvbuf = np.zeros(tot_num).astype(int)
        else:
            raise NotImplementedError
    else:
        recvbuf = None

    displs = np.array([sum(counts[:p]) for p in range(size)])
    if kind=='double':
        comm.Gatherv([value,MPI.DOUBLE], [recvbuf,counts,displs,MPI.DOUBLE],root=0)
    elif kind=='float':
        comm.Gatherv([value,MPI.FLOAT], [recvbuf,counts,displs,MPI.FLOAT],root=0)
    elif kind=='int':
        comm.Gatherv([value,MPI.INT], [recvbuf,counts,displs,MPI.INT],root=0)
    elif kind=='bool':
        comm.Gatherv([value,MPI.BOOL], [recvbuf,counts,displs,MPI.BOOL],root=0)
    elif kind=='int64':
        comm.Gatherv([value,MPI.INT64_T], [recvbuf, counts, displs, MPI.INT64_T],root=0)
    else:
        raise NotImplementedError

    return recvbuf


def get_ra_dec(ra,dec,catalog_data):
    """
    Parameters
    ----------
    ra : str
        variable name representing right ascension 
    dec: str
        variable name representing declination
    catalog_data : GCRCatalog object
        GCRCatalog object holding catalog data 
        
    Returns
    -------
    recvbuf_ra : ndarray
        rank 0 outputs array of all ra values
        other ranks output a None value 
    recvbuf_dec : ndarray 
        rank 0 outputs array of all dec values 
        other ranks output a None value

    """
    data_rank={}
    recvbuf={}
    for quantity in [ra,dec]:
        data_rank[quantity] = catalog_data[quantity]
        count = len(data_rank[quantity])
        tot_num = comm.reduce(count)
        counts = comm.allgather(count)
        if rank==0:
            recvbuf[quantity] = np.zeros(tot_num)
        else:
            recvbuf[quantity] = None
        displs = np.array([sum(counts[:p]) for p in range(size)])
        comm.Gatherv([data_rank[quantity],MPI.DOUBLE], [recvbuf[quantity], counts, displs, MPI.DOUBLE],root=0)
    recvbuf_ra = recvbuf[ra]
    recvbuf_dec = recvbuf[dec]

    return recvbuf_ra, recvbuf_dec

