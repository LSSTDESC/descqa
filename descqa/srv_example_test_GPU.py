from __future__ import print_function, division, unicode_literals, absolute_import
import sys
from .base import BaseValidationTest, TestResult

from .external.example_cuda_test import get_quantity_labels
from .external.example_cuda_test import run_test


if 'mpi4py' in sys.modules:
    from mpi4py import MPI
    from .parallel import send_to_master
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    has_mpi = True
    print('Using parallel script, invoking parallel read')
else:
    size = 1
    rank = 0 
    has_mpi = False
    print('Using serial script')
    

__all__ = ['CheckTest']

class CheckTest(BaseValidationTest):
    """
    Run a minimalist example test 
    """

    def __init__(self, **kwargs):

        # arguments
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.bands = kwargs.get('bands')
        self.no_version = kwargs.get('no_version', False)


        if not any((
                self.catalog_filters,
        )):
            raise ValueError('you need to specify catalog_filters for these checks, add a good flag if unsure')

        self.current_catalog_name = None

        super(CheckTest, self).__init__(**kwargs)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        self.current_catalog_name = catalog_name
        self.test_score = 0
        self.test_passed = False
        version = getattr(catalog_instance, 'version', '') if not self.no_version else ''

        # create filter labels
        filters=[]
        for i, filt in enumerate(self.catalog_filters):
            filters = filt['filters']
   
        # Here is where you define what data you need to read in
        quantities = get_quantity_labels(bands = self.bands)
        quantities = tuple(quantities)

        # reading in the data 
        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities,filters=filters,return_iterator=False)
        else:
            catalog_data = catalog_instance.get_quantities(quantities,return_iterator=False)
        
        data_rank={}
        recvbuf={}
        for quantity in quantities:
            data_rank[quantity] = catalog_data[quantity]
            if has_mpi:
                if 'flag' in quantity:
                    recvbuf[quantity] = send_to_master(data_rank[quantity],'bool')
                else:
                    recvbuf[quantity] = send_to_master(data_rank[quantity],'double')
            else:
                recvbuf[quantity] = data_rank[quantity]

        # Here is where the test is actually being run 
        if rank==0:
            test_score, test_passed = run_test(data = recvbuf, outdir = output_dir)
            self.test_score = test_score
            self.test_passed = test_passed
        else:
            self.test_score = 0 
            self.test_passed = False
        if has_mpi:
            self.test_score = comm.bcast(self.test_score, root=0)
            self.test_passed = comm.bcast(self.test_passed, root=0)

        return TestResult(passed=self.test_passed, score=self.test_score)

    def conclude_test(self, output_dir):
        return 
