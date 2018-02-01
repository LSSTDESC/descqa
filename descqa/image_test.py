import numpy as np
import os
import GCRCatalogs
import pandas as pd
from desc.sims.GCRCatSimInterface import *
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import GalSimCelestialObject
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



    def _build_galsim_objects(self, imsimCatalog, gsobjects={}):
        """
        Creates a dictionary indexed by galaxyID which contains galsim objects
        """

        # Extract relevant columns
        objectNames = imsimCatalog.column_by_name('uniqueId')
        galaxyID = imsimCatalog.column_by_name('galaxyID')
        raICRS = imsimCatalog.column_by_name('raICRS')
        decICRS = imsimCatalog.column_by_name('decICRS')
        xPupil = imsimCatalog.column_by_name('x_pupil')
        yPupil = imsimCatalog.column_by_name('y_pupil')
        halfLight = imsimCatalog.column_by_name('halfLightRadius')
        minorAxis = imsimCatalog.column_by_name('minorAxis')
        majorAxis = imsimCatalog.column_by_name('majorAxis')
        positionAngle = imsimCatalog.column_by_name('positionAngle')
        sindex = imsimCatalog.column_by_name('sindex')
        gamma1 = imsimCatalog.column_by_name('gamma1')
        gamma2 = imsimCatalog.column_by_name('gamma2')
        kappa = imsimCatalog.column_by_name('kappa')

        sedList = imsimCatalog._calculateGalSimSeds()

        for (name, gid, ra, dec, xp, yp, hlr, minor, major, pa, ss, sn, gam1, gam2, kap) in \
            zip(objectNames, galaxyID, raICRS, decICRS, xPupil, yPupil, halfLight,
                minorAxis, majorAxis, positionAngle, sedList, sindex,
                gamma1, gamma2, kappa):

            gid = int(gid)
            flux_dict = {'i': ss.calcADU(imsimCatalog.bandpassDict['i'],
                                         imsimCatalog.photParams)}
            gsObj = GalSimCelestialObject(imsimCatalog.galsim_type, ss, ra, dec, xp, yp,
                                          hlr, minor, major, pa, sn, flux_dict, gam1, gam2, kap, uniqueId=name)

            if imsimCatalog.galsim_type == 'sersic':
                gal = galsim.Sersic(gsObj.sindex, gsObj.halfLightRadiusArcsec)
            elif imsimCatalog.galsim_type == 'RandomWalk':
                rng = galsim.BaseDeviate(int(gsObject.uniqueId))
                gal = galsim.RandomWalk(npoints=int(gsObj.sindex),
                                        half_light_radius=float(
                                            gsObj.halfLightRadiusArcsec),
                                        rng=rng)
            # Define the flux
            gal = gal * gsObj.flux('i')

            # Apply ellipticity
            gal = gal.shear(q=gsObj.minorAxisRadians / gsObj.majorAxisRadians,
                            beta=(0.5 * np.pi + gsObj.positionAngleRadians) * galsim.radians)

            if gid in gsobjects:
                gsobjects[gid] += gal
            else:
                gsobjects[gid] = gal

        return gsobjects

    def _parse_instance_catalog(self, catalog, output_dir, imag_cut=25.0):
        """
        Reads in an instance catalog and returns an ImSim Galaxy catalog.
        Applies an iband magnitude.
        """
        commands, phosim_objects = desc.imsim.parsePhoSimInstanceFile(
            os.path.join(output_dir, 'catalog.txt'), None)
        obs_md = desc.imsim.phosim_obs_metadata(commands)

        # Complement the list of objects with the data from the parent catalog
        phosim_objects['galaxyID'] = phosim_objects['uniqueId'] // 1024
        protoDC2_data = pd.DataFrame.from_dict(
            catalog.get_quantities(['galaxyID', 'mag_i_lsst']))
        phosim_objects = phosim_objects.join(
            protoDC2_data.set_index('galaxyID'), on='galaxyID')

        # Restrict objects to imag < 25
        m = phosim_objects['mag_i_lsst'] < imag_cut
        phosim_objects = phosim_objects[m]

        # Clean up catalog
        phosim_objects = desc.imsim.validate_phosim_object_list(
            phosim_objects).accepted

        # Separate out sersic from random walk components
        sersicDataBase = phosim_objects.query("galSimType=='sersic'")
        knotsDataBase = phosim_objects.query("galSimType=='RandomWalk'")

        # Create imsim catalogs
        config = desc.imsim.read_config(None)
        # Switching to the i-band
        commands['bandpass'] = 'i'
        obs_md._bandpass = 'i'

        # Define the photometric params for COSMOS
        phot_params = desc.imsim.photometricParameters(commands)
        phot_params._gain = 1.
        phot_params._nexp = 1
        phot_params._exptime = 1.
        phot_params._effarea = 2.4**2 * (1.-0.33**2)
        phot_params._exptime = 1.

        # Create a catalog with ImSim to load the SEDs and stuf
        sersicCatalog = desc.imsim.ImSimGalaxies(sersicDataBase, obs_md)
        sersicCatalog.photParams = phot_params
        sersicCatalog.camera = LsstSimMapper().camera
        sersicCatalog.camera_wrapper = LSSTCameraWrapper()
        sersicCatalog._initializeGalSimCatalog()

        randomWalkCatalog = desc.imsim.ImSimGalaxies(knotsDataBase, obs_md)
        randomWalkCatalog.photParams = phot_params
        randomWalkCatalog.camera = LsstSimMapper().camera
        randomWalkCatalog.camera_wrapper = LSSTCameraWrapper()
        randomWalkCatalog._initializeGalSimCatalog()

        # Build galsim objects
        gsobjects = self._build_galsim_objects(sersicCatalog)
        gsobjects = self._build_galsim_objects(randomWalkCatalog, gsobjects)
        return gsobjects

    def _build_instance_catalog(self, catalog_name, output_dir):
        """
        Generates a simple instance catalog from a given catalog
        """
        cat_bulge = PhoSimDESCQA_ICRS(bulgeDESCQAObject(catalog_name),
                                      obs_metadata=obs_md,
                                      cannot_be_null=['hasBulge'])
        cat_bulge.phoSimHeaderMap = copy.deepcopy(DefaultPhoSimHeaderMap)
        cat_bulge.phoSimHeaderMap['rawSeeing'] = ('rawSeeing', None)
        cat_bulge.phoSimHeaderMap['FWHMgeom'] = ('FWHMgeom', None)
        cat_bulge.phoSimHeaderMap['FWHMeff'] = ('FWHMeff', None)
        cat_bulge.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                                chunk_size=100000)

        cat_disk = PhoSimDESCQA_ICRS(diskDESCQAObject(catalog_name),
                                     obs_metadata=obs_md,
                                     cannot_be_null=['hasDisk'])
        cat_disk.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                               chunk_size=100000, write_header=False,
                               write_mode='a')

        cat_knots = PhoSimDESCQA_ICRS(knotsDESCQAObject(catalog_name),
                                      obs_metadata=obs_md,
                                      cannot_be_null=['hasKnots'])
        cat_knots.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                                chunk_size=100000, write_header=False,
                                write_mode='a')
