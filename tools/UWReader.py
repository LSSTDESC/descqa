import numpy as np

class UWReader():

    def __init__(self, filename):
        self.catalog = self.readCatalog(filename)

    def readCatalog(self, filename):
        fullCatalog = np.genfromtxt(filename, delimiter=',', names=True)
        return fullCatalog
