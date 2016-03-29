#!/usr/bin/env python
import sys, os
import re, socket
from time import asctime, strftime, localtime
from shutil import copy

def main():

  # some variables used below and what they represent:
  #
  #             version: the version number for this "FlashTest"
  #     pathToFlashTest: absolute path up to and including the directory
  #                      directly containing the flashTest.py script
  #        pathToOutdir: absolute path up to and including the directory
  #                      to which all FlashTest output will be directed
  #       pathToSiteDir: absolute path up to and including the directory
  #                      which contains all output for a given platform.
  #             siteDir: the name of the above directory; the basename of
  #                      'pathToSiteDir' and the fully qualified domain
  #                      name of the host computer
  # pathToInvocationDir: absolute path up to and including the directory
  #                      which contains all FlashTest output for a given
  #                      invocation.
  #       invocationDir: the name of the above directory; the basename of
  #                      'pathToInvocationDir'
  #          pathToInfo: absolute path up to and including the file
  #                      containing all "test.info" data for this platform
  #                 log: the Logfile instance for this invocation of
  #                      "FlashTest"

  version = "1.0"
  owner   = os.environ["LOGNAME"]
  pathToFlashTest = os.path.dirname(os.path.abspath(sys.argv[0]))

  def __abort(msg):
    """
    called for errors in the initial invocation of FlashTest

    Because no logfile is instantiated yet, 'msg' is recorded
    in a file "ERROR" in the top-level FlashTest directory
    """
    print msg
    open(os.path.join(pathToFlashTest, "ERROR"),"a").write("%s: %s\n" % (asctime(), msg))
    sys.exit(1)


  def __deleteFiles(startDir):
    """
    Recursively search each directory under 'startDir' for text
    files named "files_to_delete", which will have been written
    by the test objects during run time, and which name files in
    that directory which should be deleted after the invocation
    has been archived.

    If a "files_to_delete" file is present, another file called
    "deleted_files" will be created, listing all files that were
    successfully deleted as well as any that were not found.

    The original "files_to_delete" file will itself be deleted
    in all cases.
    """
    def __delete():
      items = os.listdir(".")
      cwd = os.getcwd()
      deletedFiles = []
      notFound     = []
      for item in items:
        if os.path.isdir(item):
          os.chdir(item)
          __delete()
          os.chdir(cwd)
        elif item == "files_to_delete":
          filesToDelete = parser.fileToList(item)
          os.remove("files_to_delete")

          for fileToDelete in filesToDelete:
            if os.path.isfile(fileToDelete):
              os.remove(fileToDelete)
              deletedFiles.append(fileToDelete)
            else:
              notFound.append(fileToDelete + " (not found or not a file) ")

          deletedFiles.extend(notFound)
          if deletedFiles:
            open("deleted_files","w").write("\n".join(deletedFiles))

    cwd = os.getcwd()
    os.chdir(startDir)
    __delete()
    os.chdir(cwd)

  def __incErrors(whichStage):
    """
    Take a path and a stage 0-3, which corresponds to the stage
    of testing where the error occurred (0=setup, 1=compilation,
    2=execution, 3=testing) and note the error in "errors" files
    at all appropriate levels.
    """
    # masterDict["changedFromPrevious"] should be set by the test
    # object's tester component if we reached the testing stage and
    # found that the results of the test differed from those of the
    # previous invocation.
    changedFromPrevious = masterDict.has_key("changedFromPrevious")

    # note the error at the RUN LEVEL (if we've gotten that far)
    if masterDict.activeLayer >= 3:  # a run dir has been generated
                                     # with an "errors" file inside
      # This "errors" file has only 2 or 3 lines:
      #
      # 0/1: no error/error during execution stage
      # 0/1: no error/error during testing stage
      #  ! : result of test differs from
      #      those of previous invocation
      errorsFile = os.path.join(pathToRunDir, "errors")
      if whichStage == 2:
        errorLines = ["1","0"]
      elif whichStage == 3:
        errorLines = ["0","1"]
        if changedFromPrevious:
          errorLines.append("!")

      open(errorsFile, "w").write("\n".join(errorLines))

    # note the error at the BUILD LEVEL
    # This "errors" file has 5 or 6 lines:
    #
    # 0/1: no error/error during setup stage
    # 0/1: no error/error during compilation stage
    # 0/n: no errors/n errors during execution stage
    # 0/n: no errors/n errors during testing stage
    #  n : total number of runs attempted ( = num parfiles)
    #  ! : results of 1 or more tests in this build differed
    #      from those of the previous invocation.
    errorsFile = os.path.join(pathToBuildDir, "errors")
    errorLines = parser.fileToList(errorsFile)

    if len(errorLines) == 5:
      if changedFromPrevious:
        errorLines.append("!")
    else:  # can assume len(errorLines) is 6 with a "!" at the end
      # if this build has had *any* run whose results differed from
      # those of the previous invocation, set 'changedFromPrevious'
      # to True regardless of whether *this* run's results differed
      # or not. This will make sure the "!" gets put on the end of
      # the appropriate line at the invocation level (see below)
      changedFromPrevious = True

    # we actually need these a few lines down
    # for the increment at the invocation level
    numExecErrs = int(errorLines[2])
    numTestErrs = int(errorLines[3])
    totalRuns   = int(errorLines[4])

    # increment the value in stage 'whichStage' by 1
    errorLines[whichStage] = str(int(errorLines[whichStage])+1)
    open(errorsFile,"w").write("\n".join(errorLines))

    # note the error at the INVOCATION LEVEL
    # This "errors" file contains names of all builds which yielded
    # some kind of error, noting the number of failed runs or tests
    # if the test failed in the execution or testing stages.
    errorsFile = os.path.join(pathToInvocationDir, "errors")
    errorLines = parser.fileToList(errorsFile)

    # compose err message
    if whichStage == 0:
      errMsg = "failed in setup"
    elif whichStage == 1:
      errMsg = "failed in compilation"
    elif whichStage == 2:
      errMsg = "%s/%s failed in execution" % ((numExecErrs+1), totalRuns)
      if numTestErrs > 0:
        errMsg = "%s; %s/%s failed in testing" % (errMsg, numTestErrs, totalRuns)
    elif whichStage == 3:
      errMsg = "%s/%s failed in testing" % ((numTestErrs+1), totalRuns)
      if numExecErrs > 0:
        errMsg = "%s/%s failed in execution; %s" % (numExecErrs, totalRuns, errMsg)
    errMsg = "%s - %s" % (buildDir, errMsg)
    if changedFromPrevious:
      # add a "!" if changed from previous invocation's results
      errMsg = errMsg + " - !"    

    for i in range(len(errorLines)):
      # if the invocation-level errors file already has
      # data about this build, replace it with 'errMsg'
      if errorLines[i].split(" - ",1)[0] == buildDir:
        errorLines[i] = errMsg
        break
    else:
      # otherwise append it
      errorLines.append(errMsg)
      errorLines.sort()

    open(errorsFile,"w").write("\n".join(errorLines))


  ##########################
  ##  PARSE COMMAND LINE  ##
  ##########################

  # options to FlashTest which take no arguments
  # DEV should be able to adjust this in Config
  # especially now that "-u" is a custom option
  standAloneOpts = ["-t","-u","-v"]

  # Retrieve a two-tuple for this invocation where the first element is
  # a dictionary of options (with arguments, if applicable) to FlashTest,
  # and the second element is a list of two-tuples, each containing a
  # test-path and a dictionary of options specific to that test.
  #
  # Graphically, the structure of the return value of 'parseCommandLine'
  # looks like:
  # ( { FlashTest-opt1: FlashTest-val1,
  #     FlashTest-opt2: FlashTest-val2 },
  #   [ ( test-path1, { test-path1-opt1: test-path1-val1,
  #                     test-path1-opt2: test-path1-val2 } ),
  #     ( test-path2, { test-path2-opt1: test-path2-val1 } ) ] )
  flashTestOpts, pathsAndOpts = parser.parseCommandLine(sys.argv[1:], standAloneOpts)

  # see if user has specified a separate "config"
  if flashTestOpts.has_key("-c"):
    pathToConfig = flashTestOpts["-c"]
  else:
    pathToConfig = os.path.join(pathToFlashTest, "config")

  #########################
  ##  PARSE CONFIG FILE  ##
  #########################

  if os.path.isfile(pathToConfig):
    try:
      configDict = parser.parseFile(pathToConfig)
    except Exception, e:
      __abort("Error opening \"config\" file\n" + str(e))
  else:
    __abort("Configuration file \"%s\" does not exist or is not a file." % pathToConfig)

  #########################
  ##  MASTER DICTIONARY  ##
  #########################

  # The master dictionary which we'll pass to the test object begins
  # life as a LayeredDict with initial keys and values copied from
  # 'configDict' (the copy is implicit in the initialization)
  masterDict = LayeredDict(configDict)

  masterDict["version"]         = version
  masterDict["owner"]           = owner
  masterDict["pathToFlashTest"] = pathToFlashTest
  masterDict["flashTestOpts"]   = flashTestOpts

  ###########################
  ##  CHECK FOR -f OPTION  ##
  ###########################

  if flashTestOpts.has_key("-f"):
    pathToJobFile = flashTestOpts["-f"]
    if not pathToJobFile:
      __abort("The '-f' option must be followed by a path to a text file\n" +
              "consisting of a newline-delimited list of tests")
    # else
    if not os.path.isfile(pathToJobFile):
      __abort("\"%s\" does not exist or is not a file." % pathToJobFile)

    # Extend 'pathsAndOpts' to include test-paths read in from a file.
    # See comments to "parseCommandLine()" above for more details
    lines = parser.fileToList(pathToJobFile)

    # briefly re-join the 'lines' list into a single string,
    # with a single space inserted between list elements, then
    # re-split the new string by breaking on all whitespace.
    if lines: lines = re.split("\s*", " ".join(lines))

    pathsAndOpts.extend(parser.getPathsAndOpts(lines))


  if len(pathsAndOpts) == 0:
    __abort("There are no tests to run. You must provide the name of at\n" +
            "least one test or path to at least one \"test.info\" file.")

  # else
  masterDict["pathsAndOpts"] = pathsAndOpts

  # get "test.info" data, if necessary
  masterInfoNode = None

  # Do a preliminary run through all our tests to see if any will require
  # data from a "test.info" file. If so, parse that data into an instance
  # of class XmlNode for use in all tests that require it, copying the
  # "test.info" file to the local machine from a remote host if necessary.
  if flashTestOpts.has_key("-i"):
    # get 'pathToInfo' value from command-lines
    pathToInfo = flashTestOpts["-i"]
    if not os.path.isfile(pathToInfo):
      __abort("\"%s\" does not exist or is not a file." % pathToInfo)
  else:
    # get 'pathToInfo' value from "config" or, if it's not there,
    # from the default location under 'pathToFlashTest'
    pathToInfo = masterDict.get("pathToInfo", os.path.join(pathToFlashTest, "test.info"))

  # split 'pathToInfo' into its hostname and path segments
  if pathToInfo.count(":") > 0:
    infoHost, pathToInfo = pathToInfo.split(":",1)
    if infoHost == "localhost":
      infoHost = ""
  else:
    infoHost = ""

  masterDict["infohost"]   = infoHost
  masterDict["pathToInfo"] = pathToInfo

  for testPath, overrideOptions in pathsAndOpts:
    if testPath.count(os.sep) > 0:
      # At least one test in this invocation requires "test.info" data
      # so obtain the text of the "test.info" file whether it resides
      # on a remote computer or locally.
      if infoHost:
        # "test.info" file is on a remote computer
        pathToTmp = os.path.join(pathToFlashTest, "tmp")

        cmd = "scp %s:%s %s" % (infoHost, pathToInfo, pathToTmp)
        out, err, duration, exitStatus = getProcessResults(cmd)
        if err:
          __abort("Unable to retrieve \"%s:%s\"\n%s" %
                  (infoHost, pathToInfo, err))
        # else
        if exitStatus != 0:
          __abort("Exit status %s indicates error retrieving \"%s:%s\"" %
                  (exitStatus, infoHost, pathToInfo))
        # else the file should have been transfered
        filename = os.path.basename(pathToInfo)
        if not os.path.isfile(os.path.join(pathToTmp, filename)):
          __abort("File \"%s\" not found in \"%s\"" %
                  (filename, pathToTmp))
        # else
        try:
          masterInfoNode = parseXml(os.path.join(pathToTmp, filename))
        except Exception, e:
          __abort("Error parsing \"%s:%s\"\n%s" %
                  (infoHost, pathToInfo, e))

        # remove the copied file from "tmp"
        os.remove(os.path.join(pathToTmp, filename))

      else:
        # "test.info" file is local
        if not os.path.isfile(pathToInfo):
          __abort("\"%s\" does not exist or is not a file" % pathToInfo)

        # else
        try:
          masterInfoNode = parseXml(pathToInfo)
        except Exception, e:
          __abort("Error parsing \"%s\"\n%s" %
                  (pathToInfo, e))

      # 'masterInfoNode' now encapsulates all "test.info" data.
      # Break out of the for-loop.
      break


  #############################
  ##  GENERATE pathToOutdir  ##
  #############################

  pathToOutdir = flashTestOpts.get("-o", "")
  if not pathToOutdir:
    # check if we got a path to the outdir from the config file,
    # else use directory named "output" inside FlashTest directory
    pathToOutdir = masterDict.get("pathToOutdir", os.path.join(pathToFlashTest, "output"))

  # create it if it doesn't exist
  if not os.path.exists(pathToOutdir):
    try:
      os.mkdir(pathToOutdir)
    except:
      __abort("Unable to create directory %s" % pathToOutdir)

  # make sure we have an absolute path
  pathToOutdir = os.path.abspath(pathToOutdir)

  masterDict["pathToOutdir"] = pathToOutdir

  ########################
  ##  GENERATE siteDir  ##
  ########################

  # try to determine the fully qualified domain name of the host
  if masterDict.has_key("FQHostname"):
    FQHostname = masterDict["FQHostname"]
  else:
    # DEV must allow a mechanism for dealing with aliases
    # DEV need a better mechanism for figuring out computer name
    # DEV when on a wireless network
    # DEV Also, if gethostbyaddr doesn't work, flashTest.py will
    # crash even tho it's not necessary - FQHostname only needs
    # to be defined if 'site' is not (see below)
    FQHostname = socket.gethostbyaddr(socket.gethostname())[0]
    masterDict["FQHostname"] = FQHostname

  # Determine value of "site". This will be the first element of
  # 'FQHostname' unless the user has specified it with the "-s"
  # command-line option or by declaring "site" in "config"
  if flashTestOpts.has_key("-s"):
    # get the site from the command line
    siteDir = flashTestOpts["-s"]
  elif masterDict.has_key("site"):
    # get the site from the config file
    siteDir = masterDict["site"]
  else:
    # 'siteDir' will be the first element of the fully-qualified
    # hostname as returned by socket.gethostname(). i.e. if the
    # fully-qualified hostname has a "." in it, 'siteDir' will
    # be everything that comes before that "."
    siteDir = FQHostname.split(".",1)[0]

  pathToSiteDir = os.path.join(pathToOutdir, siteDir)

  # create it if it doesn't exist
  if not os.path.exists(pathToSiteDir):
    try:
      os.mkdir(pathToSiteDir)
    except:
      __abort("Unable to create directory %s" % pathToSiteDir)

  masterDict["pathToSiteDir"] = pathToSiteDir
  masterDict["siteDir"] = siteDir


  ##############################
  ##  GENERATE invocationDir  ##
  ##############################

  # Generate a directory to hold all output from this invocation of FlashTest
  # Its name will be the current day's date, plus a suffix if necessary

  # start by getting the date in YYYY-MM-DD format
  dateStr = strftime("%Y-%m-%d",localtime())

  # get a list of any dirs in "output" that already start with dateStr
  dirs = [item for item in os.listdir(pathToSiteDir) if (
          os.path.isdir(os.path.join(pathToSiteDir, item)) and
          item.startswith(dateStr))]

  if len(dirs) > 0:
    suffixes = [dir[len(dateStr):].strip("_") for dir in dirs]
    intsuffixes=[]
    for d in dirs:
        try:
            intsuffixes.append( int(d.split('_')[1] ) )
        except:
            intsuffixes.append(0)
    intsuffixes.sort()
    highestSuffix=intsuffixes[len(intsuffixes)-1]

    if highestSuffix == "":
      newSuffix = 2
    else:
      newSuffix = int(highestSuffix) + 1
    # there has already been at least one invocation
    # of FlashTest on this date, so add a suffix
    dateStr = "%s_%s" % (dateStr, newSuffix)

  invocationDir = dateStr
  pathToInvocationDir = os.path.join(pathToSiteDir, invocationDir)

  # create invocationDir
  try:
    os.mkdir(pathToInvocationDir)
  except:
    __abort("Unable to create directory %s" % pathToInvocationDir)
    

  # create a ".lock" file so that FlashTestView
  # will ignore it until the invocation is complete
  open(os.path.join(pathToInvocationDir, ".lock"),"w").write("")

  # Create a file "errors" which will record the details
  # of all errors that happen during this invocation.
  #
  # This file will be the basis of the floating statistics
  # box that appears in FlashTestView when the user hovers
  # over an invocation link (i.e. a date).
  #
  # See the comments to "__incErrors()" for more details
  open(os.path.join(pathToInvocationDir, "errors"),"w").write("")

  # Write a copy of the level-0 dictionary of 'masterDict'
  # (which at this point contains everything in 'masterDict')
  # into the invocation directory as a newline-delimited list
  # of key-value pairs
  keysAndVals = ["%s: %s" % (key, masterDict.dicts[0][key]) for key in masterDict.dicts[0]]
  open(os.path.join(pathToInvocationDir, "masterDict"),"w").write("\n".join(keysAndVals))

  masterDict["pathToInvocationDir"] = pathToInvocationDir
  masterDict["invocationDir"] = invocationDir

  ###########################
  ##  INSTANTIATE LOGFILE  ##
  ###########################

  log = Logfile(pathToInvocationDir, "flash_test.log", flashTestOpts.has_key("-v"))
  log.info("FlashTest v%s started by %s on %s\n" % (version, owner, asctime()), False)
  log.info("Original command-line: %s %s" % (sys.argv[0], " ".join(sys.argv[1:])), False)
  log.info("This invocation: %s" % invocationDir, False)
  log.brk()

  masterDict["log"] = log

  #######################################################
  ##  CHECK FOR PREVIOUS INVOCATION'S ARCHIVE RESULTS  ##
  #######################################################

  if not flashTestOpts.has_key("-t"):
    previousArchiveLog = os.path.join(pathToSiteDir, "archive.log")
    if os.path.isfile(previousArchiveLog):
      archiveText = open(previousArchiveLog, "r").read()
      log.info(archiveText, False)
      log.brk()

  ######################
  ##  ARCHIVE OBJECT  ##
  ######################
  # provides methods for communication with the remote archive

  pathToLocalArchive = masterDict.get("pathToLocalArchive", os.path.join(pathToFlashTest, "localArchive"))

  if not os.path.isabs(pathToLocalArchive):
    pathToLocalArchive = os.path.join(pathToFlashTest, pathToLocalArchive)
    log.note("Using \"%s\" as path to local archive." % pathToLocalArchive)

  masterDict["pathToLocalArchive"] = pathToLocalArchive    

  mainArchiveHost = ""
  pathToMainArchive = masterDict.get("pathToMainArchive", "")
  if pathToMainArchive:
    # work out if the main archive resides on a remote computer
    if pathToMainArchive.count(":") > 0:
      mainArchiveHost, pathToMainArchive = pathToMainArchive.split(":",1)
      if mainArchiveHost == "localhost":
        mainArchiveHost = ""

  masterDict["mainArchiveHost"]   = mainArchiveHost
  masterDict["pathToMainArchive"] = pathToMainArchive

  viewArchiveHost = ""
  pathToViewArchive = masterDict.get("pathToViewArchive", "")
  if pathToViewArchive:
    # work out if the view archive resides on a remote computer
    if pathToViewArchive.count(":") > 0:
      viewArchiveHost, pathToViewArchive = pathToViewArchive.split(":",1)
      if viewArchiveHost == "localhost":
        viewArchiveHost = ""

  masterDict["viewArchiveHost"]   = viewArchiveHost
  masterDict["pathToViewArchive"] = pathToViewArchive

  arch = Archive(log, masterDict)
  masterDict["arch"] = arch

  #######################
  ##  BEGIN FLASHTEST  ##
  #######################

  try:
    testObject = TestObject(masterDict)
  except Exception, e:
    __abort(str(e))

  # else testObject is now successfully instantiated
  if testObject.entryPoint1() == False:  # user's 1st chance to interrupt
    __incErrors(0)
  else:
    # outer loop over all test tuples, where each test tuple contains a
    # test name and a list of secondary "build tuples". Each build tuple,
    # in turn, consists of a short path to the relevant "test.info" file
    # (or the empty string if no info file was indicated), and the list of
    # override-options to the test, if any
    for testPath, overrideOptions in pathsAndOpts:
      # some variables used below and what they represent:
      # (see also comments at top of file)
      #
      #        testPath: xml-path to a node in "test.info" which contains
      #                  data relevant to a single FlashTest build
      #  pathToBuildDir: absolute path up to and including the directory
      #                  which contains all output for a single build, and
      #                  which contains one or more instances of 'runDir'
      #                  (see below).
      #        buildDir: The basename of 'pathToBuildDir', an immedidate
      #                  sub-directory of 'invocationDir'
      #    pathToRunDir: absolute path up to and including the directory
      #                  which directly contains all output for one run
      #                  of the executable against a single set of runtime
      #                  parameters.
      #          runDir: The basename of 'pathToRunDir', an immediate sub-
      #                  directory of 'buildDir'
      masterDict.setActiveLayer(1)
      masterDict["testPath"] = testPath
      masterDict["overrideOptions"] = overrideOptions

      # start log entry for this build
      log.brk()
      shortPathToNode = os.path.normpath(os.path.join(siteDir, testPath))

      if testPath.count(os.sep) > 0:
        thisNode = masterInfoNode.findChild(shortPathToNode)

        if not thisNode:
          # 'testPath' led to a non-existant node,
          if infoHost:
            log.err("%s does not exist in \"%s:%s\"\n" % (shortPathToNode, infoHost, pathToInfo) +
                    "Aborting this build.")
          else:
            log.err("%s does not exist in \"%s\"\n" % (shortPathToNode, pathToInfo) +
                    "Aborting this build.")
          continue

        # else
        infoData = "\n".join([line.strip() for line in thisNode.text if len(line.strip()) > 0])

        if len(infoData) == 0:
          if infoHost:
            log.err("%s exists in \"%s:%s\", but contains no info data.\n" % (shortPathToNode, infoHost, pathToInfo) +
                    "Aborting this build.")
          else:
            log.err("%s exists in \"%s\", but contains no info data.\n" % (shortPathToNode, pathToInfo) +
                    "Aborting this build.")
          continue

        infoOptions = parser.parseLines(thisNode.text)
        log.stp("Parsed \"%s\"" % shortPathToNode)
      else:
        log.info("No \"test.info\" data provided")
        infoOptions = {}

      # determine 'buildDir', which will hold output from this individual build
      buildDir = os.path.normpath(testPath).replace(os.sep, "_")

      # determine absolute path to buildDir
      pathToBuildDir = os.path.join(pathToInvocationDir, buildDir)
      log.stp("Creating directory \"%s\"" % pathToBuildDir)

      if os.path.isdir(pathToBuildDir):
        # a directory of this name already exists[0]
        log.err("A directory called \"%s\" already exists.\n" % pathToBuildDir +
                "Skipping this build.")
        continue

      # else
      try:
        os.mkdir(pathToBuildDir)
      except Exception, e:
        log.err("%s\n" % str(e) +
                "Skipping this build.")
        continue

      masterDict["pathToBuildDir"] = pathToBuildDir
      masterDict["buildDir"]       = buildDir

      # enter override-options and "test.info" options into 'masterDict'
      # "test.info" options will temporarily overwrite override-options,
      # for the sake of constructing a good message to the user
      masterDict.update(overrideOptions)
      masterDict.update(infoOptions)

      # override-options take precedence over options with the same
      # name from the "test.info" file. Note in logfile if this happens.
      for key in overrideOptions:
        if masterDict.has_key(key) and masterDict[key] != overrideOptions[key]:
          log.note("key \"%s\" with value \"%s\" overridden by new value \"%s\"" %
                   (key, masterDict[key], overrideOptions[key]))
          masterDict[key] = overrideOptions[key]

      if testPath.count(os.sep) > 0:
        # Record the text of the original "test.info" data in the logfile
        log.info("****** \"test.info\" ******")
        log.info(infoData)
        log.info("*************************")

        # Record a copy of 'infoData' in this build's output directory.
        # The first line is an xml node-path to the original "test.info"
        # data, which FlashTestView uses to generate a link to the info-
        # file manager.
        msg  = shortPathToNode + "\n\n"
        msg += infoData + "\n"
        open(os.path.join(pathToBuildDir, "test.info"),"w").write(msg)

      #####################
      ##  "errors" File  ##
      #####################

      # Creates a file "errors" which will record the number of
      # errors encountered for one build/series of runs as well
      # as the total number of parfiles intended to be run. The
      # file is a text file with 5 lines: one line each for the
      # number of errors in setup and compilation (will be 0 or
      # 1), one line each for the number of errors occurring in
      # runtime and in testing, and a final line for the total
      # number of parfiles that were to be tried in this build.
      startVals = "0\n0\n0\n0\n0"
      open(os.path.join(pathToBuildDir, "errors"),"w").write(startVals)

      # The user can interfere again here if he/she wants, e.g.
      # to control what object is put in place for 'testObject's'
      # "setupper" and "compiler" members or to add an instruction
      # instruction to FlashTestView regarding what kind of a menu
      # item to generate on the "view builds" screen. 
      testObject.entryPoint2()

      # Now that information from the "test.info" file has been
      # read in, add instances of strategy classes for setup,
      # compilation, execution, and testing stages.
      for keyword in ["setupper", "compiler"]:
        testObject.installComponent(keyword)

      #############
      ##  SETUP  ##
      #############
      log.info("** setup phase **")
      if testObject.setup() == False:
        __incErrors(0)
        continue

      ###################
      ##  COMPILATION  ##
      ###################
      log.info("** compilation phase **")
      if testObject.compile() == False:
        __incErrors(1)
        continue

      #################
      ##  TRANSFERS  ##
      #################
      transfers = masterDict.get("transfers",[])
      if len(transfers) > 0:
        # change space-delimited 'transfers' into a python list
        transfers = parser.stringToList(transfers)

      # make copies in output directory of any files in 'transfers'
      transferedFiles = []
      for path in transfers:
        # 'path' is assumed to be a relative path to the file
        # from the top-level directory of the BCC QA source
        # Split this into path and basename
        path, file = os.path.split(path)
        log.info("Transfering file \"%s\"" % file)
        if os.path.isfile(os.path.join(pathToBCC, path, file)):
          copy(os.path.join(pathToBCC, path, file), pathToBuildDir)
          transferedFiles.append(file)
        else:
          log.warn("File \"%s\" was not found in \"%s\"" % (file, path))

      ################
      ##  PARFILES  ##
      ################
      parfiles = parser.stringToList(masterDict.get("parfiles", ""))
      masterDict["parfiles"] = parfiles
 
      if len(parfiles) > 0:
        # set number of parfiles to be run on bottom line of "errors" file
        errorsFile = os.path.join(pathToBuildDir, "errors")
        lines = open(errorsFile,"r").readlines()
        lines[4] = str(len(parfiles)) + "\n"
        open(errorsFile,"w").write("".join(lines))

      # DEV another entry point here for changing transfers and parfiles?

      #################
      ##  EXECUTION  ##
      #################
      log.info("** execution phase **")

      # inner loop over each parfile listed for a single executable
      for parfile in parfiles:

        if os.path.isabs(parfile):
          if os.path.isfile(parfile):
            parfileText = open(parfile).read()
            parfile = os.path.basename(parfile)
          else:
            __incErrors(2)
            log.err("Parfile \"%s\" not found\n" % parfile +
                    "Skipping this run.")
            continue

        else:
          __incErrors(2)
          log.err("Parfiles specified in a \"test.info\" file such as \"%s\"\n" % parfile +
                  "must be declared as absolute paths. Skipping this run.")
          continue

        # define the name of the directory that will hold all results from this run
        firstDot = parfile.find(".")
        if firstDot > 0:
          runDir = parfile[:firstDot]  # truncate any dot and following chars
                                       # (essentially to truncate the '.par')
        pathToRunDir = os.path.join(pathToBuildDir, runDir)

        if os.path.isdir(pathToRunDir):
          # a directory of this name already exists in 'buildDir'
          __incErrors(2)
          log.err("There is already a directory called '%s' in %s\n" % (runDir, pathToBuildDir) +
                  "Skipping this run.")
          continue
        # else all is well - The last level of nested directories will be
        # for the output of the run against this parfile
        os.mkdir(pathToRunDir)

        # set 'masterDict' layer to 2 and erase any keys previously set
        # at this layer, i.e. in an earlier iteration of the loop
        masterDict.setActiveLayer(2)

        masterDict["runDir"]       = runDir
        masterDict["pathToRunDir"] = pathToRunDir

        # make links in 'runDir' to any transfers we had
        for transferedFile in transferedFiles:
          os.link(os.path.join(pathToBuildDir, transferedFile), os.path.join(pathToRunDir, transferedFile))

        # write the parfile into our new directory
        open(os.path.join(pathToRunDir, parfile), "w").write(parfileText)

        masterDict["pathToParfile"] = os.path.join(pathToRunDir, parfile)
        masterDict["parfile"] = parfile

        # create an "errors" file at the run level whose value will be
        # "00" (no errors), "10" (error during runtime) or "01" (error
        # during the testing phase)
        open(os.path.join(pathToRunDir, "errors"),"w").write("0\n0")

        # The user can interfere again here if he/she wants, e.g.
        # to control what object is put in place for 'testObject's'
        # "executer" and "tester" members
        testObject.entryPoint3()

        for keyword in ["executer", "tester"]:
          testObject.installComponent(keyword)

        # run the executable
        if testObject.execute() == False:
          __incErrors(2)
          continue

        ############
        ##  TEST  ##
        ############
        log.info("** testing phase **")
        if testObject.test() == False:
          __incErrors(3)


  #############################
  ##  ARCHIVING AND CLEANUP  ##
  #############################

  log.stp("All tests completed.")

  if flashTestOpts.has_key("-t"):
    log.stp("No archiving done for test-run. FlashTest complete. End of Logfile.")
    os.remove(os.path.join(pathToInvocationDir, ".lock"))
  elif ((not pathToMainArchive) and (not pathToViewArchive)):
    log.stp("FlashTest complete. End of Logfile.")
    os.remove(os.path.join(pathToInvocationDir, ".lock"))
  else:
    log.stp("Preparing data for archiving. Outcome of archiving attempt will be written to\n" +
            "\"%s\"\n" % os.path.join(pathToSiteDir, "archive.log") +
            "and will be incorporated into the regular logfile of the next invocation.\n" +
            "End of Logfile.")
            
    # instantiate archive logfile - this data will appear
    # in the next invocation's regular logfile
    archiveLog = Logfile(pathToSiteDir, "archive.log", flashTestOpts.has_key("-v"))
    archiveLog.info("Archiving results for invocation %s:" % invocationDir, False)

    errorCreatingTarFile = False

    os.remove(os.path.join(pathToInvocationDir, ".lock"))

    if pathToMainArchive:
      archiveLog.stp("Creating local tarball \"%s.tar.gz\" for main archive..." % pathToInvocationDir)
      try:
        pathToTarFile = arch.makeTarFile()
      except Exception, e:
        errorCreatingTarFile = True
        archiveLog.err("Error creating tarball\n%s\n" % e +
                       "No files will be deleted from local copy.")
      else:
        try:
          arch.sendTarFileToMainArchive(pathToTarFile)
        except Exception, e:
          archiveLog.err("Unable to send tarball to main archive\n%s\n" % e +
                         "Tarfile still exists at \"%s\"" % pathToTarFile)
        else:
          archiveLog.stp("Tarball sent to main archive.")
          # remove local copy of tarball
          os.remove(pathToTarFile)

        archiveLog.stp("Deleting specified files for slim copy...")
        # the files "deleted_files" that this method generates will obviously
        # only be present in the copy of the output sent to the view archive
        __deleteFiles(pathToInvocationDir)
        archiveLog.stp("Local files deleted.")

    if pathToViewArchive:
      if errorCreatingTarFile:
        archiveLog.stp("Sending fat copy of output to view archive...")
      else:
        archiveLog.stp("Sending slim copy of output to view archive...")

      try:
        arch.sendToViewArchive()
      except Exception, e:
        archiveLog.err("%s\n" % e +
                       "No copy of output sent to view archive.")
      else:
        if errorCreatingTarFile:
          archiveLog.stp("Fat copy of output sent to view archive.")
        else:
          archiveLog.stp("Slim copy of output sent to view archive.")

    archiveLog.stp("FlashTest complete.")


def usage():
  print "Usage for \"flashTest.py\":"
  print "  ./flashTest.py [general opts] [test-path#1] [override opts] \\"
  print "                                [test-path#2] [override opts] \\"
  print "                                [test-path#n] [override opts]"
  print "or:"
  print "  ./flashTest.py [general opts] -f [path/to/job/file] \\"
  print "                                   [test-path#1][override opts]"
  print ""
  print "General options to FlashTest are:"
  print "-c <file>: use <file> as \"config\" instead of default"
  print "-i <file>: use <file> as source of \"test.info\" data"
  print "-f <file>: read <file> (a \"job-file\") for list of test-paths"
  print "-o <dir> : direct output to <dir>"
  print "-s <name>: <name> is the name of this site (host)"
  print "-t       : test run (no archiving)"
  print "-u       : update BCC QA before run"
  print "-v       : verbose output (print logfile as it is written)"
  print "-z <dir> : BCC QA source rooted at <dir>"
  print ""
  print "Each option must be entered with its own preceding dash. For example"
  print ""
  print "  $ ./flashTest.py -t -v -f path/to/job/file"
  print ""
  print "will work, but"
  print ""
  print "  $ ./flashTest.py -tvf path/to/job/file"
  print ""
  print "will not."
  print ""
  print "-i, -o, and -z options will override 'pathToInfo', 'pathToOutdir', and"
  print "'pathToBCC' in the \"config\" file."
  print ""
  print "Variables that are set in the \"config\" file or in a \"test.info\" file"
  print "may be overridden on the command line, or in a job-file, by so-called"
  print "\"override options\" e.g.:"
  print ""
  print "  $ ./flashTest.py Comparison/Sod key1=val1 key2=val2"
  print ""
  print "If an option passed in this way, for example, 'key1', requires no value"
  print "the syntax looks like:"
  print ""
  print "  $ ./flashTest.py Comparison/Sod key1= key2=val2"
  print ""
  print "Different entryPoint, setupper, compiler, executer, and tester components"
  print "can be assigned to different test-species in the \"config\" file. See the"
  print "notes in \"config\" for more details or see the FlashTest User's Guide."


if __name__ == '__main__':
  # add "lib" to our sys.path. It will be in the
  # same directory where the executable is found
  sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),"lib"))
  import flashTestParser as parser
  from archive import Archive
  from getProcessResults import getProcessResults
  from layeredDict import LayeredDict
  from logfile import Logfile
  from testObject import TestObject
  from xmlNode import parseXml
  if len(sys.argv) == 1 or sys.argv[1] == "-h":
    usage()
  else:
    main()
