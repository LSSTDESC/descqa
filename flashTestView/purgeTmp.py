#!/usr/bin/env python
import sys, os
import time

ignoreList = [".svn"]

def purgeTmp(pathToTmp=os.path.join(os.getcwd(), "tmp")):
  # delete all files (really links to files) in testsuite/tmp
  # that have not been accessed in over 24 hours
  if os.path.isdir(pathToTmp):
    cwd = os.getcwd()
    os.chdir(pathToTmp)
    items = os.listdir(".")
    currentTime = time.time()
    for item in items:
      if item not in ignoreList:
        try:
          lastModified = os.stat(item)[8]
        except OSError:
          # link target is gone
          lastModified = -(24*60*60)
        if currentTime - lastModified >= (24*60*60):
          os.system("rm -rf %s" % item)
    os.chdir(cwd)
  else:
    print "\"%s\" either does not exist or is not a directory" % pathToTmp

if __name__==("__main__"):
  pathToFlashTestView = os.path.dirname(os.path.abspath(sys.argv[0]))
  pathToTmp = os.path.join(pathToFlashTestView, "tmp")
  purgeTmp(pathToTmp)
