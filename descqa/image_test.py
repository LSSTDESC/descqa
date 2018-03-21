import numpy as np
import os
import GCRCatalogs
import pandas as pd
from desc.sims.GCRCatSimInterface import *
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import GalSimCelestialObject
from desc.sims.GCRCatSimInterface.InstanceCatalogWriter import PhoSimDESCQA_ICRS, PhoSimDESCQA
from desc.sims.GCRCatSimInterface import bulgeDESCQAObject, diskDESCQAObject, knotsDESCQAObject
from lsst.sims.catUtils.exampleCatalogDefinitions import DefaultPhoSimHeaderMap
import copy
import galsim
import desc.imsim
from .base import BaseValidationTest, TestResult
from .plotting import plt


class ImageVerificationTest(BaseValidationTest):

    def __init__(self,
                 imag_cut=25.,
                 fov=0.25,
                 obsHistID=1418971,
                 opsimdb='minion_1016_sqlite_new_dithers.db'):

        self.imag_cut = imag_cut

        # Create obs metadata
        obs_gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')
        self.obs_md = obs_gen.getObservationMetaData(obsHistID=obsHistID,
                                                         boundType='circle',
                                                         boundLength=fov)[0]

        self.camera = LsstSimMapper().camera

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # Create a galaxy instance catalog
        self._build_instance_catalog(catalog_name=catalog_name,
                                     output_dir=output_dir)

        # Parse the instance catalog and produces a dictionary of galsim objects
        galaxies = self._parse_instance_catalog(catalog_instance, output_dir,
                                                imag_cut=self.imag_cut)

        # Draws and analyse each galaxy

    def conclude_test(self, output_dir):
        pass

    def _parse_instance_catalog(self, catalog, output_dir, imag_cut=25.0):
        """
        Reads in an instance catalog and returns an ImSim Galaxy catalog.
        Applies an iband magnitude.
        """
        # load default imSim configuration
        config_path = os.path.dirname(desc.imsim.__file__)+'/data/default_imsim_configs'
        config = desc.imsim.read_config()

        catalog_contents = desc.imsim.parsePhoSimInstanceFile(os.path.join(output_dir, 'catalog.txt'))

        obs_md = catalog_contents.obs_metadata
        phot_params = catalog_contents.phot_params
        sources = catalog_contents.sources
        gs_object_arr = sources[0]
        gs_object_dict = sources[1]

        df = pd.DataFrame.from_dict(
            catalog.get_quantities(['galaxyID', 'LSST_filters/magnitude:LSST_i:observed']))

        # Loop over the objects to create GalSim objects
        gsobjects = {}
        for obj in gs_object_arr:

            gid = obj.uniqueId // 1024

            # if the input magnitude is lower than the cut, we skip
            if df.loc[df['galaxyID'] == gid]['LSST_filters/magnitude:LSST_i:observed'].iloc[0] > imag_cut:
                continue

            if obj.galSimType == 'sersic':
                gal = galsim.Sersic(obj.sindex, obj.halfLightRadiusArcsec)
            elif obj.galSimType == 'RandomWalk':
                rng = galsim.BaseDeviate(int(obj.uniqueId))
                gal = galsim.RandomWalk(npoints=obj.npoints,
                                        half_light_radius=obj.halfLightRadiusArcsec,
                                        rng=rng)
            # Define the flux
            gal = gal.withFlux(obj.flux('i'))

            # Apply ellipticity
            gal = gal.shear(q=obj.minorAxisRadians / obj.majorAxisRadians,
                            beta=(0.5 * np.pi + obj.positionAngleRadians) * galsim.radians)

            # Add the galaxy to the list
            if gid in gsobjects:
                gsobjects[gid] += gal
            else:
                gsobjects[gid] = gal

        return gsobjects

    def _build_instance_catalog(self, catalog_name, output_dir):
        """
        Generates a simple instance catalog from a given catalog
        """
        cat_bulge = PhoSimDESCQA_ICRS(bulgeDESCQAObject(catalog_name),
                                      obs_metadata=self.obs_md,
                                      cannot_be_null=['hasBulge'])
        cat_bulge.phoSimHeaderMap = copy.deepcopy(DefaultPhoSimHeaderMap)
        cat_bulge.phoSimHeaderMap['rawSeeing'] = ('rawSeeing', None)
        cat_bulge.phoSimHeaderMap['FWHMgeom'] = ('FWHMgeom', None)
        cat_bulge.phoSimHeaderMap['FWHMeff'] = ('FWHMeff', None)
        cat_bulge.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                                chunk_size=100000)

        cat_disk = PhoSimDESCQA_ICRS(diskDESCQAObject(catalog_name),
                                     obs_metadata=self.obs_md,
                                     cannot_be_null=['hasDisk'])
        cat_disk.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                               chunk_size=100000, write_header=False,
                               write_mode='a')

        cat_knots = PhoSimDESCQA_ICRS(knotsDESCQAObject(catalog_name),
                                      obs_metadata=self.obs_md,
                                      cannot_be_null=['hasKnots'])
        cat_knots.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                                chunk_size=100000, write_header=False,
                                write_mode='a')
