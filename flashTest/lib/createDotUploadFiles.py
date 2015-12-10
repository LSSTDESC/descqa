#!/usr/bin/env python
import sys, re, os, shutil

# regular expressions for dirs we won't descend into
# to search for logfiles
avoidPatterns = ["\.svn", "2005-09-*", "2005-10-*", "2005-11-*"]

def matchesAvoidPattern(item):
  for avoidPattern in avoidPatterns:
    if re.match(avoidPattern, item):
      return True
  return False

def main():

  scriptDir = os.path.abspath(os.path.dirname(sys.argv[0]))
  user = "tester"
  experiment = "test-suite runs"
  template = ("user: %s\n" +
              "experiment: %s\n" +
              "grouptag: %s\n")

  results = []

  def __createDotUploadFiles(dir):
    cwd = os.getcwd()
    os.chdir(dir)
    items = os.listdir(".")
    for item in items:
      if item.endswith(".log"):
        # see if this logfile is from a successful run
        if os.path.isfile("errors"):
          errors = open("errors","r").read()
          if "1" in errors:
            continue
          # else all "0"s in "errors" means this was a successful run
          runName   = os.getcwd()
          buildName = os.path.dirname(runName)
          dateName  = os.path.dirname(buildName) # we're going to ignore this one
          siteName  = os.path.dirname(dateName)

          runName   = os.path.basename(runName)
          buildName = os.path.basename(buildName)
          dateName  = os.path.basename(dateName)
          siteName  = os.path.basename(siteName)
          groupTag  = siteName + "_" + buildName + "_" + runName
          try:
            outdir = os.path.join("/home/tester/flashTest/outputcopy",siteName,dateName,buildName,runName)
            if not os.path.isdir(outdir):
              os.makedirs(outdir)
              shutil.copy(item, outdir)
              open(os.path.join(outdir,".upload"),"w").write(template % (user, experiment, groupTag))
              results.append("wrote grouptag:\n\t%s\nin dir:\n\t%s" % (groupTag, outdir))
          except:
            results.append("error writing grouptag:\n\t%s\nin dir:\n\t%s" % (groupTag, outdir))
        else:
          # "errors" file is suspiciously not here...
          continue
      elif os.path.isdir(item) and not matchesAvoidPattern(item):
        __createDotUploadFiles(item)
    os.chdir(cwd)

  if sys.argv[1] == "-d":
    if len(sys.argv) != 3:
      usage()
    else:
      flashTestDir = sys.argv[2]
      cwd = os.getcwd()
      pathToFlashTestDir = os.path.join(cwd, flashTestDir)
      if not os.path.isdir(pathToFlashTestDir):
        print "error: no such directory: %s" % pathToFlashTestDir
        sys.exit(1)
      # else we know flashTestDir is really a directory
  else:
    flashTestDir = scriptDir
  
  __createDotUploadFiles(flashTestDir)
  for result in results:
    print result
  
def usage():
  print "Do this later"

if __name__ == '__main__':
  if len(sys.argv) == 1 or sys.argv[1] == "-h":
    usage()
  else:
    main()
