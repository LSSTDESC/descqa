from UWReader import UWReader
from ANLReader import ANLReader

class CatalogReader():
    """
    Reads
    """
    def __init__(self,filename=''):
        catType=self.gettype(filename)
        self.catalogs = {}
        if (catType=="ANL"):
            self.catalogs[catType] = ANLReader(filename).catalog
        elif(catType=="Chile"):
            self.catalogs[catType] = ChileReader(filename).catalog
        elif(catType=="UW"):
            self.catalogs[catType] = UWReader(filename).catalog
        else:
            return None
        #endif

    def gettype(self,filename):
        import os
        if (len(filename)==0) or (filename is None):
            raise ValueError('No filename provided')
        else:
            dirname,catfile=os.path.split(filename)
            subdirs=dirname.split('/')
            type=subdirs[len(subdirs)-1]
            return type
        #endif
