import os
import numpy as np

class ANLReader():

    """
    Input ANL Mock
    """
    #CONSTANTS
    Quiet='Quiet'
    unknown='unknown'
    hdf5_filetype='.hdf5'

    def __init__(self,filename=''):
        filestem,filetype=os.path.splitext(filename)
        self.catalog=self.readmock(filename=filename,filetype=filetype)
        #self.catalog=GalaxyCatalog.ANLGalaxyCatalog(mockcat)

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
            #print "hdf5groups are",hdf5keys
            for key in hdf5keys:
                if(key.find("Output")!=-1):
                    outgroup=hdfFile[key]
                    datakeys,datattr=self.gethdf5group(outgroup)
                    mockcat[key]=self.gethdf5arrays(outgroup)
                #endif                                                                                                     

            #endfor                                                                                                        
            hdfFile.close()
            print "Filled mock catalog dictionary with",len(mockcat),"keys:",mockcat.keys()
        else:
            print myname,self.unknown,filetype
            mockcat=None
        #endif                                                                                                             
        return mockcat

    ###hdf5 files                                                                                            
    def gethdf5keys(self,id,*args):
        if(len(args)>0):
            blurb=args[0]
        else:
            blurb=self.null
        #endif                                                                                               
        keys=id.keys()
        keylist=[str(x) for x in keys]
        if(blurb!=self.Quiet):
            print blurb," = ",keylist
        return keylist

    def gethdf5attributes(self,id,key,*args):
        #Return dictionary with group attributes and values                                                  
        if(len(args)>0):
            blurb=args[0]
        else:
            blurb=self.null
        #endif                                                                                               
        group=id[key]
        dict={}
        for item in group.attrs.items():
            attribute=str(item[0])
            dict[attribute]=item[1]

        #endfor                                                                                              
        if(blurb!=self.Quiet):
          print blurb,self.space+str(key),": {attributes, values} = ",dict
        #endif            

    def gethdf5group(self,group,*args):
        #return dictionary of (sub)group dictionaries                                                        
        groupkeys=self.gethdf5keys(group,self.Quiet)
        groupdict={}
        for key in groupkeys:
            mydict=self.gethdf5attributes(group,key,self.Quiet)
            groupdict[str(key)]=mydict

        #endfor                                                                                              
        return groupkeys,groupdict

    def gethdf5arrays(self,group,*args):
        groupkeys=self.gethdf5keys(group,self.Quiet)
        arraydict={}
        oldlen=-1
        for key in groupkeys:
            array=np.array(group[key])
            arraylen=len(array)
            if(oldlen>-1):         #check that array length is unchanged                                     
                if(oldlen!=arraylen):
                    print "Warning: hdf5 array length changed for key",key
                #endif                                                                                       
            else:
                oldlen=arraylen   #set to ist array length                                                   
            #endif                                                                                           
            arraydict[str(key)]=array

        #endfor                                                                                              
        return arraydict
