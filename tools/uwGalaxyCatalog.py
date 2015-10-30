import numpy as np

class GalaxyCatalog():

    def __init__(self, catReader, dataLabels):
        self.catalogs = catReader.catalogs
        self.dataLabels = dataLabels

    def get_quantities(self, quantitiesList, **kwargs):

        for catType in self.catalogs.keys():
            if catType == 'UW':
                print 'here'
                uwData = load_UW_data(self.catalogs['UW'], quantitiesList, **kwargs).data
                return uwData
#        if dataLabels == None:
#            dataLabels = self.dataLabels
#            print dataLabels
#        for dataType in dataLabels:
#            for catType in self.catReader.keys():
#                if catType == 'UW':


class load_UW_data():

    def __init__(self, catalog, dataLabels, **kwargs):
        self.catalog = catalog
        self.dataLabels = dataLabels
        self.data = self.parse_labels(**kwargs)

    def parse_labels(self, **kwargs):
        for dataLabel in self.dataLabels:
            if dataLabel=='stellar_mass':
                data = self.load_stellar_masses(**kwargs)
        return data

    def load_stellar_masses(self, **kwargs):

        if 'zMin' in kwargs.keys():
            zMin = kwargs['zMin']
        else:
            zMin = None

        if 'zMax' in kwargs.keys():
            zMax = kwargs['zMax']
        else:
            zMax = None

        print zMin, zMax

        if (zMin is None) and (zMax is None):
            stellar_mass = self.catalog['mass_stellar']
        elif (zMin is not None) and (zMax is not None):
            stellar_mass = self.catalog['mass_stellar'][np.where((zMin < self.catalog['redshift']) & (self.catalog['redshift'] < zMax))]
        elif zMin is None:
            stellar_mass = self.catalog['mass_stellar'][np.where(self.catalog['redshift'] < zMax)]
        elif zMax is None:
            stellar_mass = self.catalog['mass_stellar'][np.where(zMin < self.catalog['redshift'])]

        return stellar_mass*1e10
