import numpy as np

import GCRCatalogs
from halotools.mock_observables.alignments import ed_3d_one_two_halo_decomp, ed_3d
from astropy.table import Table
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['AlignmentValidationTest']

class AlignmentValidationTest(BaseValidationTest):

    def __init__(self, step=247, haloMassCut=12.1, **kwargs):
        """ Constructor
        """
        self.step = step
        self.haloMassCut = haloMassCut

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        """
        """
        # Extract relevant information
        cat = Table(catalog_instance.get_quantities(['x','y','z', 'step', 'is_central',
                            'step', 'hostHaloMass',
                            'hostHaloEigenVector3X', 'hostHaloEigenVector3Y', 'hostHaloEigenVector3Z',
                            'hostHaloEigenValue1', 'hostHaloEigenValue2','hostHaloEigenValue3']))

        # Apply selection
        m = (log10(cat['hostHaloMass']) > self.haloMassCut) & (cat['step'] == self.step)
        m &= (cat['is_central'] == 1)
        vals = cat[m]
        # Creates sample for correlation
        sample = np.zeros((len(vals),3))
        sample[:,0] = vals['x']; sample[:,1] = vals['y']; sample[:,2] = vals['z']
        orientations = zeros((len(vals),3))
        orientations[:,0] = vals['hostHaloEigenVector3X'] ; orientations[:,1] = vals['hostHaloEigenVector3Y']
        orientations[:,2] = vals['hostHaloEigenVector3Z']
        # Remove halos for which we might not have a shape
        m = sqrt(sum(orientations**2,axis=-1)) > 0.1
        sample = sample[m]
        orientations = orientations[m]
        # Compute orientation-direction correlation function
        r = logspace(-1,2,16)
        res = ed_3d(sample, orientations, sample, r)

        # Now do the same thing with MB2
        mb2 = Table.read('/global/homes/f/flanusse/mb2_z1_central.hdf5')
        m = (log10(mb2['halos.m_dm']*1e10) > self.haloMassCut) & (mb2['halos.central'] == 1.)
        vals = d[m]
        # Creates sample for correlation
        sample = np.zeros((len(vals),3))
        sample[:,0] = vals['halos.x']; sample[:,1] = vals['halos.y']; sample[:,2] = vals['halos.z']
        orientations = zeros((len(vals),3))
        orientations[:,0] = vals['shapesDM.a3d_x'] ; orientations[:,1] = vals['shapesDM.a3d_y']
        orientations[:,2] = vals['shapesDM.a3d_z']
        # Remove halos for which we might not have a shape
        m = sqrt(sum(orientations**2,axis=-1)) > 0.1
        sample = sample[m] *1e-3 # Convert distances into h^1Mpc
        orientations = orientations[m]
        # Compute orientation-direction correlation function
        res2 = ed_3d(sample, orientations, sample, r)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(0.5*(r[1:]+r[:-1]), res, label='protoDC2')
        ax.loglog(0.5*(r[1:]+r[:-1]), res2, label='MBII')
        ax.set_title('Halos Ellipticity-Direction correlation function ($M_{DM} > 10^{12}$ )');
        ax.legend(loc=1)
        ax.set_xlabel('R $[h^{-1} Mpc]$')

        fig.savefig(os.path.join(output_dir, 'alignment_test_halo_ED_{}.png'.format(catalog_name)))
        plt.close(fig)
        return TestResult(inspect_only=True)
