import numpy as np
import os
import GCRCatalogs
import pandas as pd
from desc.sims.GCRCatSimInterface import *
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.GalSimInterface import GalSimCelestialObject
from desc.sims.GCRCatSimInterface.InstanceCatalogWriter import PhoSimDESCQA_ICRS, PhoSimDESCQA
from desc.sims.GCRCatSimInterface import bulgeDESCQAObject, diskDESCQAObject, knotsDESCQAObject
from lsst.sims.catUtils.exampleCatalogDefinitions import DefaultPhoSimHeaderMap
import copy
import galsim
import desc.imsim
from .base import BaseValidationTest, TestResult
from .plotting import plt
from astropy.table import Table
from multiprocessing import Pool
from functools import partial

def _draw_galaxies(inds, cosmos_cat=None, cosmos_index=None, galaxies=None, cosmos_noise=None):
    """ Function to draw the galaxies into postage stamps
    """
    i,k = inds
    im_sims = galsim.ImageF(64, 64, scale=0.03)
    im_cosmos = galsim.ImageF(64, 64, scale=0.03)
    flag=True

    try:
        cosmos_gal = cosmos_cat.makeGalaxy(cosmos_index[i])
        psf = cosmos_gal.original_psf

        sims_gal = galsim.Convolve(galaxies[k], psf)
        sims_gal.drawImage(im_sims,method='no_pixel')
        im_sims.addNoise(cosmos_noise)

        cosmos_gal = galsim.Convolve(cosmos_gal, psf)
        cosmos_gal.drawImage(im_cosmos, method='no_pixel')
    except:
        flag=False

    return (k, im_sims, i, im_cosmos, flag)



class ImageVerificationTest(BaseValidationTest):

    def __init__(self,
                 imag_cut=24.,
                 fov=0.25,
                 obsHistID=1418971,
                 opsimdb='minion_1016_sqlite_new_dithers.db',
                 galsim_cosmos_dir='/global/homes/f/flanusse/repo/GalSim/share/COSMOS_25.2_training_sample',
                 pool_size=None):

        self.imag_cut = imag_cut
        self.galsim_cosmos_dir = galsim_cosmos_dir
        self.pool_size=pool_size

        # Create obs metadata
        obs_gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')
        self.obs_md = obs_gen.getObservationMetaData(obsHistID=obsHistID,
                                                         boundType='circle',
                                                         boundLength=fov)[0]

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # Create a galaxy instance catalog
        self._build_instance_catalog(catalog_name=catalog_name,
                                     output_dir=output_dir)

        # Parse the instance catalog and produces a dictionary of galsim objects
        galaxies = self._parse_instance_catalog(catalog_instance, output_dir,
                                                imag_cut=self.imag_cut)

        cosmos_cat = galsim.COSMOSCatalog(dir=self.galsim_cosmos_dir)

        m = cosmos_cat.param_cat['mag_auto'][cosmos_cat.orig_index] < self.imag_cut
        cosmos_index = np.array(range(cosmos_cat.getNObjects()))[m]
        np.random.shuffle(cosmos_index)

        cosmos_noise = galsim.getCOSMOSNoise()

        print("Processing %d galaxies"%len(galaxies))
        indices = [(i,k) for i,k in enumerate(galaxies)]

        engine = partial(_draw_galaxies, cosmos_cat=cosmos_cat, cosmos_index=cosmos_index,
                         galaxies=galaxies, cosmos_noise=cosmos_noise)
        
        if self.pool_size is None:
            res = []
            for inds in indices:
                res.append(_draw_galaxies(inds, cosmos_cat, cosmos_index, galaxies, cosmos_noise))
        else:
            with Pool(self.pool_size) as p:
                res = p.map(engine, indices)

        # Extract the postage stamps into separate lists, discarding the ones
        # that failed
        ims = {}
        imc = {}
        for k, im_sims, i, im_cosmos, flag in res:
            if flag:
                ims[k] = im_sims
                imc[i] = im_cosmos

        # Computes moments of sims and real images
        m_cosmos = self._moments(imc)
        m_sims = self._moments(ims)

        return m_cosmos, m_sims, imc, ims



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

        commands = desc.imsim.metadata_from_file(os.path.join(output_dir, 'catalog.txt'))
        # Switching to the i-band, no matter what obsHistID was used
        commands['bandpass'] = 'i'
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        phot_params = desc.imsim.photometricParameters(commands)
        # Define the photometric params for COSMOS
        phot_params._gain = 1.
        phot_params._nexp = 1
        phot_params._exptime = 1.
        phot_params._effarea = 2.4**2 * (1.-0.33**2) * 100**2 # in cm2
        print("Loading sources")
        sources = desc.imsim.sources_from_file(os.path.join(output_dir, 'catalog.txt'),
                                obs_md,
                                phot_params)

        gs_object_arr = sources[0]
        gs_object_dict = sources[1]

        df = pd.DataFrame.from_dict(
            catalog.get_quantities(['galaxyID', 'LSST_filters/magnitude:LSST_i:observed']))
        print("Building GalSim objects")
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

        if 'addon_knots' in catalog_name:
            cat_knots = PhoSimDESCQA_ICRS(knotsDESCQAObject(catalog_name),
                                          obs_metadata=self.obs_md,
                                          cannot_be_null=['hasKnots'])
            cat_knots.write_catalog(os.path.join(output_dir, 'catalog.txt'),
                                    chunk_size=100000, write_header=False,
                                    write_mode='a')

    def _moments(self, images):
        """
        Computes HSM moments for a set of galsim images
        """
        sigma = []
        e  = []
        e1 = []
        e2 = []
        g  = []
        g1 = []
        g2 = []
        flag = []
        amp = []

        for k in images:
            shape = images[k].FindAdaptiveMom(guess_centroid=galsim.PositionD(32,32), strict=False)
            amp.append(shape.moments_amp)
            sigma.append(shape.moments_sigma)
            e.append(shape.observed_shape.e)
            e1.append(shape.observed_shape.e1)
            e2.append(shape.observed_shape.e2)
            g.append(shape.observed_shape.g)
            g1.append(shape.observed_shape.g1)
            g2.append(shape.observed_shape.g2)
            if shape.error_message is not '':
                flag.append(False)
            else:
                flag.append(True)

        return Table({'amp': amp,
                      'sigma_e': sigma,
                      'e': e,
                      'e1': e1,
                      'e2': e2,
                      'g': g,
                      'g1': g1,
                      'g2': g2,
                      'flag': flag})
