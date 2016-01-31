#! /usr/bin/env python

from GalaxyCatalogInterface import GalaxyCatalog
import h5py
import numpy as np
import astropy.cosmology
import astropy.units as u


class SAGGalaxyCatalog(GalaxyCatalog):
    """
    Semi-Analytic Galaxies (SAG) model galaxy catalog class. Uses generic 
    quantity and filter mechanisms defined by GalaxyCatalog base class. In
    addition, implements the use of 'stored' vs. 'derived' quantity getter
    methods.
    Additional data structures:
    
    catalog       A dictionary whose keys are the names of the various stored
                  properties, and whose values are arrays containing the values
                  of these quantities for each of the galaxies in the catalog.
    
    derived       A dictionary whose keys are the names of derived quantities
                  and whose values are tuples containing the string name of a
                  corresponding stored quantity (actually present in the file)
                  and a pointer to the function used to compute the derived
                  quantity from the stored one. Some catalogs may support
                  having the stored quantity be a tuple of stored quantity
                  names.
    """

    def __init__(self, fn=None):
        self.type_ext   = 'hdf5'
        self.filters    = { 'zlo':          True,
                            'zhi':          True
                          }
        self.quantities = { 'redshift'    : self._get_stored_property,
                            'M_star_disk' : self._get_stored_property,
                            'M_star_bulge': self._get_stored_property,
                            'stellar_mass': self._get_derived_property,
                          }

        self.derived    = {'stellar_mass' : ('M_star_disk', 'M_star_bulge', self._add)} 
        self.catalog    = {}
        self.sky_area   = 4.0 * np.pi * u.sr # All sky by default
        self.cosmology = None
        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        self.catalog = self.SAGcollection(fn)
        
        self.cosmology = astropy.cosmology.LambdaCDM(H0 = self.catalog.readAttr('H0')[0],
                                                     Om0 = self.catalog.readAttr('Omega')[0],
                                                     Ode0 = self.catalog.readAttr('OmegaLambda')[0])

        return self

    def _get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        zrange=[filters['zlo'], filters['zhi']]
        return self.catalog.readDataset(dsname=quantity, zrange=zrange)

    def _get_derived_property(self, quantity, filters):
        """
        Return a derived halo property. These properties aren't stored
        in the catalog but can be computed from properties that are via
        a simple function call.
        """
        zrange = [filters['zlo'], filters['zhi']]
        stored_qty_rec = self.derived[quantity]
        stored_qty_name_1 = stored_qty_rec[0]
        stored_qty_name_2 = stored_qty_rec[1]
        stored_qty_func = stored_qty_rec[2]
        qty_array_1 = self.catalog.readDataset(dsname=stored_qty_name_1,\
                                               zrange=zrange)
        qty_array_2 = self.catalog.readDataset(dsname=stored_qty_name_2,\
                                               zrange=zrange)

        propList = [qty_array_1, qty_array_2]

        return stored_qty_func(propList=propList)

    def _add(self, propList):
        """
        Routine that returns element-wise addition of two arrays.
        """
        return sum(propList)


    class SAGcollection():
        """
        A collection of SAGdata objects of SAG. It assumes the outputs are ordered in
        different directories (one per snapshot/redshift).
        """

        def __init__(self, filename, boxSizeMpc=0):
            """
            It creates a new catalog collection from a specified directory
            """
            self.dataList  = []
            self.snaptag   = []
            self.nfiles    = []
            self.redshift  = []
            self.nz        = 0
            self.boxSizeMpc = 0
            self.zminidx    = -1
            import os
            filename = os.path.split(os.path.split(filename)[0])[0] 
            print(filename)

            if 0 != boxSizeMpc:
                simname = 'SAG_sim'
                self.boxSizeMpc = boxSizeMpc
            else:
                # This file must exist!
                simdat = open(filename+"/simdata.txt")
                simname = simdat.readline()
                self.boxSizeMpc = simdat.readline()
                simdat.close()

            import os
            ls = os.listdir(filename)
            ls.sort()
            for name in ls:
                if name.split("_")[0] == "snapshot":
                    snap = name.split("_")[1]
                    lss = os.listdir(filename+"/"+name)
                    lss.sort()
                    filesindir = 0
                    for h5name in lss:
                        if h5name.split(".")[-1] in ['hdf5','h5'] and \
                            h5name.split("_")[0] == 'gal':
                            # This is a SAG file!:
                            filesindir += 1

                            if filesindir == 1: ## add a new redshift to the collection:
                                self.dataList.append(self.SAGdata(simname, self.boxSizeMpc))
                                self.snaptag.append(snap)
                                self.nfiles.append(0)
                                self.nz += 1

                            sagname = filename+'/'+name+'/'+h5name
                            print('Opening file '+sagname)
                            self.dataList[self.nz-1].addFile(sagname)
                            self.nfiles[self.nz-1] += 1
            # and the corresponding redshifts: 
            for i in range(self.nz):
                self.redshift.append(float(self.dataList[i].readAttr('Redshift')))


            # If the outputs are not ordered in subfolders, then they are mixed up:
                filesindir = 0
                if 0 == self.nz:
                    for name in ls:
                        if name.split(".")[-1] in ['hdf5','h5'] and \
                           name.split("_")[0] == 'gal':

                           filesindir += 1
                           snap =  name.split("_")[1]
                           if snap == 'itf':
                              snap = name.split("_")[2]
                              try:
                                  idx = self.snaptag.index(snap)
                              except ValueError:
                                  self.dataList.append(self.SAGdata(simname, self.boxSizeMpc))
                                  self.snaptag.append(snap)
                                  self.nfiles.append(0)
                                  idx = self.nz
                                  self.nz += 1

                              sagname = filename+'/'+name
                              print('Opening file '+sagname)
                              self.dataList[idx].addFile(sagname)
                              self.nfiles[idx] += 1

                              # and the redshift:
                              if 1 == self.nfiles[idx]:
                                  self.redshift.append(float(self.dataList[idx].readAttr('Redshift')))

                self.zminidx = self.redshift.index(min(self.redshift))
                self.reduced = self.dataList[0].reduced


        def clear(self):
            del self.snaptag[:]
            del self.nfiles[:]
            del self.redshift[:]
            self.nz = 0
            self.boxSizeMpc = 0
            self.zminidx = -1
            for sagd in self.dataList:
               sagd.clear()


        def _lookup_z(self, zlow, zhigh):
            """
            It returns a list with the redshifts of the collection that are in the
            range zlow <= Z <= zhigh.
            """
            zl = []
            for z in self.redshift:
                if zlow <= z <= zhigh:
                    zl.append(z)
            return zl

        def readDataset(self, dsname, multiSnaps=False, zrange=None, **kwargs):
            """
            It searches for an unique or a set of redshifts or boxes and returns the 
            requested datasets.
            """
            for key in kwargs.keys():
                if key not in ['idxfilter']:
                    raise KeyError(key)

            if multiSnaps: print("Warning: requesting multiple snaps!")

            if not zrange:
                if not multiSnaps:
                    # the lowest redshift (hopefully z=0):
                    iarr = [self.zminidx]
                else:
                    iarr = [i for i in range(self.nz)]
            else:
                zl = zrange[0]
                zh = zrange[1]
                if multiSnaps:
                    iarr = []
                    for z in self._lookup_z(zl, zh):
                        iarr.append(self.redshift.index(z))
                else:
                    # search for the lowest match:
                    zarr = self._lookup_z(zl, zh)
                    iarr = [self.redshift.index(min(zarr))]

            if 'idxfilter' in kwargs.keys():
                flt = kwargs['idxfilter']
            else:
                flt = []
            # Now we have the list of redshifts we are going to use, let's concatenate the
            # datasets:
            for k, i in enumerate(iarr):
                dsarr = self.dataList[i].readDataset(dsname)
                if 0 == k:
                    nparr = dsarr
                else:
                    nparr = np.concatenate([nparr, dsarr])

            if 0 != len(flt):
                tmp = nparr[flt]
                del nparr
                nparr = tmp

            return nparr
        # All the files should have the same attributes and units, so these return
        # the ones found in the first file at the lowest redshift.
        def readAttr(self, attname):
            """
            It returns a requested attribute.
            """
            return self.dataList[self.zminidx].readAttr(attname)

        def readUnits(self):
            """ 
            It return an 'Units' object with the unit conversions of the catalog.
            """
            return self.dataList[self.zminidx].readUnits()

        def datasetList(self):
            """
            It returns the dataset list of the files, following the groups recursively.
            """
            return self.dataList[self.zminidx].datasetList()


        def getGalaxies_by_ids(self, ids, dslist='all', multiSnaps=False, zrange=None):
            """
            It returns a dictionary with the different datasets for all the requested
            galaxies.
            """
            if multiSnaps: print("Warning: requesting multiple snaps!")

            if not zrange:
                if not multiSnaps:
                    # the lowest redshift (hopefully z=0):
                    iarr = [self.zminidx]
                else:
                    iarr = [i for i in range(self.nz)]
            else:
                zl = zrange[0]
                zh = zrange[1]
                if multiSnaps:
                    iarr = []
                    for z in self._lookup_z(zl, zh):
                        iarr.append(self.redshift.index(z))
                else:
                    # search for the lowest match:
                    zarr = self._lookup_z(zl, zh)
                    iarr = [self.redshift.index(min(zarr))]

            if dslist == 'all':
                dslist = self.dataList[iarr[0]].datasetList()
            if 'Histories/DeltaT_List' in dslist:
                dslist.remove('Histories/DeltaT_List')

            # here we need to verify the datasets are present in all the snaps.
            if multiSnaps:
                if len(iarr) > 1:
                    ds1 = set(dslist)
                    ds2 = set(self.dataList[iarr[1]].datasetList())
                    dslist = list(ds1.intersection(ds2))

            # now we collect the galaxies:
            gal = { 'ngal': np.array([0]) }
            for i in iarr:
                gtmp = self.dataList[i].getGalaxies_by_ids(ids, dslist)
                if 0 == gal['ngal']:
                    gal.update(gtmp)
                    gal['ngal'] += len(gal[dslist[0]])
                    if len(iarr) > 1:
                        gal['redshift'] = np.repeat(self.redshift[i],len(gal[dslist[0]]))
                else:
                    for ds in dslist:
                        gal[ds] = np.concatenate([gal[ds], gtmp[ds]])
                    gal['ngal'] += len(gtmp[dslist[0]])
                    tmp = np.repeat(self.redshift[i],len(gtmp[dslist[0]]))
                    gal['redshift'] = np.concatenate([gal['redshift'], tmp])

            return gal




        class SAGdata():
           """
           The class 'SAGdata' stores a collection of hdf5 output files
           created by the SAG code. It can extract a particular array from
           all the stored files and returns a unique np array with the 
           requested data.
           """
           def __init__(self, simname, boxSizeMpc):
              """
              It creates an empty collection of files.
              """
              self.simname = str(simname)
              self.filenames = []
              self.dataList = []
              self.nfiles = 0
              self.boxSizeMpc = boxSizeMpc
              self.reduced = False


           def clear(self):
              self.simname = ""
              del self.filenames[:]
              self.nfiles = self.boxSizeMpc = 0
              self.reduced = False
              for fsag in self.dataList:
                 fsag.close()
            

           def addFile(self, filename):
              """
              It adds an hdf5 file to the object.
              """
              try:
                 sag = h5py.File(filename, "r")
              except IOError:
                 print("Cannot load file: '"+filename+"'")
                 return
              self.filenames.append(filename)
              self.dataList.append(sag)
              self.nfiles += 1
         
              if 1 == self.nfiles:
                 try:
                    attr = self.dataList[0].attrs['REDUCED_HDF5']
                    if attr == 'YES':
                       self.reduced = True
                 except KeyError:
                    pass


           def readDataset(self, dsname, idxfilter=[]):
              """
              It returns a unique np array of the requested dataset only
              if it exists in all loaded SAG files.

              The idxfilter can be created with np.where(condition), for example:
              >>> types = d.readDataset("Type")
              >>> row, col = np.where(types == 0)
              >>> discMass  = d.readDataset("DiscMass", idxfilter=row)
              >>> pos = d.readDataset("Pos", idxfilter=row)
              """
              for i, sag in enumerate(self.dataList):
                 dsarr = np.array(sag.get(dsname))
                 if None == dsarr.all():
                    print("Dataset '"+dsname+"' not present in "+self.filenames[i])
                    return None
                 if 0 == i:
                    nparr = dsarr
                 else:
                    nparr = np.concatenate([nparr, dsarr])
              if 0 != len(idxfilter):
                 tmp = nparr[idxfilter]
                 del nparr
                 nparr = tmp
            
              return nparr


           def readAttr(self, attname, fnum=0):
              """
              It returns the value of the requested attribute from a particular
              file of the list.
              """
              try:
                 attr = self.dataList[fnum].attrs[attname]
                 return attr
              except KeyError:
                 print("Attribute '"+attname+"' not present in "
                       +self.filenames[fnum])
                 return None


           def readUnits(self, fnum=0):
              """
              It returns an instance of the 'Units' class, with all the unit conversions
              of the data found in the firts hdf5 file of the list.
              """
              if 0 < self.nfiles:
                 if not self.reduced:
                    m_in_g = float(self.dataList[fnum].attrs["UnitMass_in_g"])
                    l_in_cm = float(self.dataList[fnum].attrs["UnitLength_in_cm"])
                    vel_in_cm_s = float(self.dataList[fnum].attrs["UnitVelocity_in_cm_per_s"])
                 else:
                    m_in_g = 1.989e33  # Msun
                    l_in_cm = 3.085678e21  # kpc
                    vel_in_cm_s = 1e5      # km/s
    
                 h = float(self.readAttr('Hubble_h'))
         
                 units = Units(l_in_cm, m_in_g, vel_in_cm_s, h)
                 return units
              else:
                 return None
    
           def datasetList(self, fnum=0, group="/"):
              ks = []
              for tag in self.dataList[fnum][group].keys():
                 if type(self.dataList[fnum][group+tag]) is h5py._hl.dataset.Dataset:
                    ks.append(group+tag)
                 elif type(self.dataList[fnum][group+tag]) is h5py._hl.group.Group:
                    tmp = self.datasetList(fnum, group=group+tag+"/")
                    ks += tmp
              return ks


           def _gal_idxs(self, ids, dsname):

              if type(ids) != list: ids = [ids]
              idxs = []
              boxes = []
              for i in range(self.nfiles):
                 dset = self.dataList[i][dsname]
                 tmp = np.where(np.in1d(dset, ids, assume_unique=True))[0]
                 idxs += tmp.tolist()
                 for _ in range(len(tmp)): boxes.append(i)
              return np.array(idxs), np.array(boxes) 
  

           def getGalaxies(self, dslist='all'):
              if dslist == 'all':
                 dslist = self.datasetList()
              gal = {}
              for dstag in dslist:
                 if type(self.dataList[0][dstag]) is h5py._hl.dataset.Dataset:
                    gal[dstag] = self.readDataset(dstag)
              return gal


           def getGalaxies_by_ids(self, ids, dslist='all'):
              """
              It returns a dictionary with the different datasets for all the requested
              galaxies.
              """
              if dslist == 'all':
                 dslist = self.datasetList()
                 dslist.remove('Histories/DeltaT_List')
              # retrieve indexes of the galaxies:
              idname = 'GalaxyID' if self.reduced else 'UID'
              idxs, boxes = self._gal_idxs(ids, idname)
              gal = {}
              for dstag in dslist:
                 if type(self.dataList[0][dstag]) is h5py._hl.dataset.Dataset:
                    dims = self.dataList[0][dstag].shape[1]
                    l = np.zeros((len(idxs),dims), dtype=self.dataList[0][dstag].dtype)
                    for i in range(self.nfiles):
                       l_idx = (boxes == i)
                       l[l_idx] = self.dataList[i][dstag][:][idxs[l_idx]]
                    gal[dstag] = l
              return gal


      
     
