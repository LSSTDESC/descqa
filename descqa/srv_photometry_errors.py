from __future__ import print_function, division, unicode_literals, absolute_import
import sys
from .base import BaseValidationTest, TestResult
from .external.photometry_errors import get_quantity_labels
from .external.photometry_errors import run_test, run_test_double
import numpy as np

if 'mpi4py' in sys.modules:
    from mpi4py import MPI
    from .parallel import send_to_master, get_kind
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
    

__all__ = ['MatchTest']


class MatchTest(BaseValidationTest):
    """
    Run a simple matching test
    """

    def __init__(self, **kwargs):

        # arguments
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.bands = kwargs.get('bands')
        self.models = kwargs.get('models')
        self.secondary_catalog_filters = kwargs.get('secondary_catalog_filters', [])

        self.no_version = kwargs.get('no_version', False)


        if not any((
                self.catalog_filters,
        )):
            raise ValueError('you need to specify catalog_filters for these checks, add a good flag if unsure')

        self.current_catalog_name = None

        super(MatchTest, self).__init__(**kwargs)


    def run_on_two_catalogs(self,catalog_instances, catalog_names, output_dir):
        """
        catalog instances is a dictionary, with keys of self.catalogs_to_run.

        """
        self.current_catalog_name = catalog_names
        self.test_score = 0
        self.test_passed = False

        # create filter labels
        filters=[]
        secondary_filters=[]
        for i, filt in enumerate(self.catalog_filters):
            filters = filt['filters']
        for i, filt in enumerate(self.secondary_catalog_filters):
            secondary_filters = filt['filters']
        truths=[]
        # Here is where you define what data you need to read in
        if 'dp02' in catalog_names[0]:
            truth = "MATCH"
        else:
            truth = "truth"
        truths.append(truth)
        quantities = get_quantity_labels(bands = self.bands, models = self.models, truth =truth)
        quantities = tuple(quantities)


        # reading in the data
        if len(filters) > 0:
            catalog_data = catalog_instances[catalog_names[0]].get_quantities(quantities,filters=filters,return_iterator=False)
        else:
            catalog_data = catalog_instances[catalog_names[0]].get_quantities(quantities,return_iterator=False)

        data_rank={}
        recvbuf={}
        for quantity in quantities:
            data_rank[quantity] = catalog_data[quantity][~mask_tot]
            if has_mpi:
                if rank==0:
                    kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
                else:
                    kind = ''
                kind = comm.bcast(kind, root=0)
                recvbuf[quantity] = send_to_master(data_rank[quantity],kind)
            else:
                recvbuf[quantity] = data_rank[quantity]



        if 'dp02' in catalog_names[1]:
            truth = "MATCH"
        else:
            truth = "truth"
        truths.append(truth)
        quantities = get_quantity_labels(bands = self.bands, models = self.models, truth =truth)
        quantities = tuple(quantities)

        # reading in the secondary data
        if len(secondary_filters) > 0:
            catalog_data_secondary = catalog_instances[catalog_names[1]].get_quantities(quantities,filters=secondary_filters,return_iterator=False)
        else:
            catalog_data_secondary = catalog_instances[catalog_names[1]].get_quantities(quantities,return_iterator=False)

        data_rank={}
        recvbuf_secondary={}
        for quantity in quantities:
            data_rank[quantity] = catalog_data_secondary[quantity]
            if has_mpi:
                if rank==0:
                    kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
                else:
                    kind = ''
                kind = comm.bcast(kind, root=0)
                recvbuf_secondary[quantity] = send_to_master(data_rank[quantity],kind)
            else:
                recvbuf_secondary[quantity] = data_rank[quantity]



        # Here is where the test is actually being run
        if rank==0:
            test_score, test_passed = run_test_double(data = recvbuf, data2 = recvbuf_secondary, outdir = output_dir, bands = self.bands, models = self.models, truths=truths)
            self.test_score = test_score
            self.test_passed = test_passed
        else:
            self.test_score = 0
            self.test_passed = False
        if has_mpi:
            self.test_score = comm.bcast(self.test_score, root=0)
            self.test_passed = comm.bcast(self.test_passed, root=0)

        return TestResult(passed=self.test_passed, score=self.test_score)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        self.current_catalog_name = catalog_name
        self.test_score = 0
        self.test_passed = False
        version = getattr(catalog_instance, 'version', '') if not self.no_version else ''

        # create filter labels
        filters=[]
        for i, filt in enumerate(self.catalog_filters):
            filters = filt['filters']
   

        # quantities to read 

        if 'dp02' in catalog_name:
            truth = "MATCH"
        else:
            truth = "truth"

        quantities = get_quantity_labels(bands = self.bands, models = self.models, truth =truth)
        quantities = tuple(quantities)


        # reading in the data 
        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities,filters=filters,return_iterator=False)
        else:
            catalog_data = catalog_instance.get_quantities(quantities,return_iterator=False)
        
        data_rank={}
        recvbuf={}

        mask_tot = np.zeros(len(catalog_data[quantities[0]])).astype('bool')
        for quantity in quantities:
            if type(catalog_data[quantity])==np.ma.core.MaskedArray:
                mask_tot += catalog_data[quantity].mask

        for quantity in quantities:
            data_rank[quantity] = catalog_data[quantity][~mask_tot]
            #print(len(data_rank[quantity]),rank,flush=True)
            if has_mpi:
                if rank==0:
                    kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
                else:
                    kind = ''
                kind = comm.bcast(kind, root=0)
                recvbuf[quantity] = send_to_master(data_rank[quantity],kind)
            else:
                recvbuf[quantity] = data_rank[quantity]

        # Here is where the test is actually being run 
        if rank==0:
            test_score, test_passed = run_test(data = recvbuf, outdir = output_dir,bands = self.bands, models = self.models)
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
