from UWReader import UWReader
from ANLReader import ANLReader

class CatalogReader():
    """
    Reads catalogs from different sources
    """
    ANL='ANL'
    UW='UW'
    Chile='Chile'
    def __init__(self,filename=''):
        catType=self.gettype(filename)
        if (catType==self.ANL):
            self.catalog = ANLReader(filename).catalog
        elif(catType==self.Chile):
            self.catalog = ChileReader(filename).catalog
        elif(catType==self.UW):
            self.catalog = UWReader(filename).catalog
        else:
            return None

        self.catType = catType
        #endif

    def gettype(self,filename):
        import os
        if (len(filename)==0) or (filename is None):
            raise NameError('No filename provided')
        else:
            dirname,catfile=os.path.split(filename)
            subdirs=dirname.split('/')
            type=subdirs[len(subdirs)-1]
            return type
        #endif
