class CatalogReader():
    """
    Reads
    """
    def __init__(self,filename=''):
        type=self.gettype(filename)
        if (type=="ANL"):
            return ANLReader(filename)
        elif(type=="Chile"):
            return ChileReader(filename)
        elif(type=="UW"):
            return UWReader(filename)
        else:
            return None
        #endif

    def gettype(self,filename):
        import os
        if(len(filename)==0 or filename is None):
            return None
        else:
            dirname,catfile=os.path.split(filename)
            subdirs=dirname.split('/')
            type=subdirs[len(subdirs)-1]
            return type
        #endif
