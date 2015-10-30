class ANLReader(CatalogReader):
    """
    Input ANL Mock
    """
    #CONSTANTS

    def __init__(self,filename=''):




    def readmock(self,filename='',filetype=''):
        myname=self.readmock.__name__
        from numpy import loadtxt
        f=open(filename,"r")
        if(filetype==self.hdf5_filetype):
            import h5py
            mode='r'
            hdfFile=h5py.File(filename,'r')
            print "Opened hdf5 file:",hdfFile
            hdf5keys,hdf5attr=self.gethdf5group(hdfFile)
            mockcat={}
            print "hdf5groups are",hdf5keys
            for key in hdf5keys:
                if(key.find("Output")!=-1):
                    outgroup=hdfFile[key]
                    datakeys,datattr=self.gethdf5group(outgroup)
                    mockcat[key]=self.gethdf5arrays(outgroup)
                #endif                                                                                                     

            #endfor                                                                                                        
            hdfFile.close()
        else:
            self.errorlog(myname,self.unknown,filetype)
        #endif                                                                                                             
        print "Filled mock catalog dictionary with",len(mockcat),"keys:",mockcat.keys()
        return mockcat
