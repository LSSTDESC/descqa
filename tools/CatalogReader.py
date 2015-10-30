from UWReader import UWReader
#from ANLReader import ANLReader

class CatalogReader():
    """
    Reads
    """
    def __init__(self,filename=''):
        type=self.gettype(filename)
        if (type=="ANL"):
            self.reader = ANLReader(filename)
        elif(type=="Chile"):
            self.reader = ChileReader(filename)
        elif(type=="UW"):
            self.reader = UWReader(filename)
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
