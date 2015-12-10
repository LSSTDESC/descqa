import os, re, time
import secondsToHuman
from getProcessResults import getProcessResults
from strategyTemplates import *

#########################################
##  FRAMEWORK ACTIONS FOR BCC QA TESTS ##
#########################################

class BCCEntryPoint(EntryPointTemplate):
  def entryPoint1(self):
    log           = self.masterDict["log"]            # guaranteed to exist by flashTest.py
    flashTestOpts = self.masterDict["flashTestOpts"]  # guaranteed to exist by flashTest.py

    ###############################
    ##  CHECK FOR BCC QA SOURCE  ##
    ###############################

    pathToBCC = flashTestOpts.get("-z","")
    if not pathToBCC:
      # check if we got a path to the BCC QA source from the config file
      pathToBCC = self.masterDict.get("pathToBCC","")
      # make sure we have an absolute path
      if pathToBCC and not os.path.isabs(pathToBCC):
        pathToBCC = os.path.join(pathToFlashTest, pathToBCC)
    else:
      # make sure we have an absolute path
      if not os.path.isabs(pathToBCC):
        pathToBCC = os.path.join(os.getcwd(), pathToBCC)

    if not pathToBCC:
      log.err("You must provide a path to a copy of the BCC QA source\n" +
              "either in a \"config\" file or with the \"-z\" option.")
      return False
    elif not os.path.isdir(pathToBCC):
      log.err("\"%s\" does not exist or is not a directory." % pathToBCC)

    self.masterDict["pathToBCC"] = pathToBCC

    ####################################
    ##  UPDATE BCC QA SOURCE IF ASKED  ##
    ####################################

    pathToInvocationDir = self.masterDict["pathToInvocationDir"]  # guaranteed to exist by flashTest.py

    if flashTestOpts.has_key("-u"):
      updateScript = self.masterDict.get("updateScript","").strip()
      if updateScript:
        log.stp("Attempting to update BCC QA source at \"%s\" with \"%s\"" % (pathToBCC, updateScript))

        cwd = os.getcwd()
        os.chdir(pathToBCC)

        updateOutput = ""
        out, err, duration, exitStatus = getProcessResults(updateScript)
        if err:
          log.err("Unable to update BCC QA source.\n" + err)
          return False
        elif exitStatus != 0:
          log.err("Exit status %s indicates error updating BCC QA source" % exitStatus)
          return False
        else:
          log.info("BCC QA source was successfully updated")
          open(os.path.join(pathToInvocationDir, "update_output"),"w").write(out)
  
        os.chdir(cwd)
      else:
        log.err("\"-u\" passed to command line but no key \"updateScript\"\n" +
                "found in \"config\". Unable to update BCC QA source")
        return False
    else:
      log.warn("BCC QA source at \"%s\" was not updated" % pathToBCC)
    
    return True


  def entryPoint2(self):
    testPath = self.masterDict["testPath"]  # guaranteed to exist by flashTest.py
    firstElement = testPath.split("/",1)[0]

#EXAMINE BEGIN
    # Write some info into "linkAttributes" for use by FlashTestView
    if firstElement == "L2DiffComparison":
      pathToBuildDir = self.masterDict["pathToBuildDir"]  # guaranteed to exist by flashTest.py
      tester = self.masterDict.get("tester", "L2DiffTester")
      text = "testerClass: %s" % tester
      open(os.path.join(pathToBuildDir, "linkAttributes"),"w").write(text)
    elif firstElement == "FitComparison":
      pathToBuildDir = self.masterDict["pathToBuildDir"]  # guaranteed to exist by flashTest.py
      tester = self.masterDict.get("tester", "FitTester")
      text = "testerClass: %s" % tester
      open(os.path.join(pathToBuildDir, "linkAttributes"),"w").write(text)


  def entryPoint3(self):
    pathToBCC       = self.masterDict["pathToBCC"]        # guaranteed to exist by flashTest.py
    pathToRunDir    = self.masterDict["pathToRunDir"]     # guaranteed to exist by flashTest.py
    parfile         = self.masterDict.get("parfile","")
    setupName       = self.masterDict.get("setupName","")

    script          = "%s/setup_parfile.py %s %s %s" % (pathToBCC, setupName, parfile, pathToRunDir)
    os.chdir(pathToRunDir)
    out, err, duration, exitStatus = getProcessResults(script)

    testPath = self.masterDict["testPath"]  # guaranteed to exist by flashTest.py
    firstElement = testPath.split("/",1)[0]

    # give this test an appropriate executer component (will be
    # be automatically installed based on value in 'masterDict')
    if firstElement == "Inspection":
      self.masterDict["executer"] = "InspectionExecuter"
    elif firstElement == "L2DiffComparison":
      self.masterDict["executer"] = "ComparisonExecuter"
    elif firstElement == "FitComparison":
      self.masterDict["executer"] = "ComparisonExecuter"

    # give this test an appropriate tester component (will be
    # be automatically installed based on value in 'masterDict')
    if firstElement == "Inspection":
      tester = self.masterDict.get("tester")
      self.masterDict["tester"] = "InspectionTester"
    elif firstElement == "L2DiffComparison":
      tester = self.masterDict.get("tester")
      self.masterDict["tester"] = "L2DiffTester"
    elif firstElement == "FitComparison":
      tester = self.masterDict.get("tester")
      self.masterDict["tester"] = "FitTester"
#EXAMINE END


class BCCSetupper(SetupperTemplate):
  def setup(self):
    """
    Set up the BCC QA tools to run a particular test
    """
    log             = self.masterDict["log"]              # guaranteed to exist by flashTest.py
    pathToBCC       = self.masterDict["pathToBCC"]        # guaranteed to exist by flashTest.py
    pathToBuildDir  = self.masterDict["pathToBuildDir"]   # guaranteed to exist by flashTest.py
    pathToFlashTest = self.masterDict["pathToFlashTest"]  # guaranteed to exist by flashTest.py

    pathToSimData   = self.masterDict["pathToSimData"]
    pathToObsData   = self.masterDict["pathToObsData"]

    setupName      = self.masterDict.get("setupName","")
    setupOptions   = self.masterDict.get("setupOptions","")
    

    if len(setupName) == 0:
      log.err("No setup name provided.\n" +
              "Skipping this build.")
      return False

    self.masterDict["pathToFlashExe"] = os.path.join(pathToBuildDir, "descqa.sh")

    # setup script
    script = "%s/setup.py %s %s %s %s %s" % \
      (pathToBCC, setupName, setupOptions, pathToBuildDir, pathToSimData, pathToObsData)

    # record setup invocation
    open(os.path.join(pathToBuildDir, "setup_call"),"w").write(script)

    # log timestamp of command
    log.stp(script)

    # get stdout/stderr and duration of setup and write to file
    out, err, duration, exitStatus = getProcessResults(script)
    if out:
      open(os.path.join(pathToBuildDir, "setup_output"),"w").write(out)
    else:
      open(os.path.join(pathToBuildDir, "setup_output"),"w").write("No setup output")
    if len(err) > 0:
      open(os.path.join(pathToBuildDir, "setup_error"),"w").write(err)

    # return the success or failure of the setup
    if exitStatus == 0:
      log.stp("setup was successful")
      return True
    else:
      log.stp("setup was not successful")
      return False


class BCCCompiler(CompilerTemplate):
  def compile(self):
    """
    does nothing at present -- the BCC QA scripts don't require compilation
    """
    pathToBuildDir  = self.masterDict["pathToBuildDir"]   # guaranteed to exist by flashTest.py
    open(os.path.join(pathToBuildDir, "gmake_output"),"w").write("No gmake output")
    return True

#PMR  def getDeletePatterns(self):
#PMR    return [self.masterDict.get("exeName", "flash-exe")]


class BCCExecuter(ExecuterTemplate):
  def execute(self, timeout=None):
    """
    run a BCC QA test, piping output and other data into 'runDir'

    pathToRunDir: abs path to output dir for this unique test
        numProcs: number of processors used for this run
    """
    log             = self.masterDict["log"]              # guaranteed to exist by flashTest.py
    pathToBCC       = self.masterDict["pathToBCC"]        # guaranteed to exist by flashTest.py
    pathToRunDir    = self.masterDict["pathToRunDir"]     # guaranteed to exist by flashTest.py
    pathToFlashTest = self.masterDict["pathToFlashTest"]  # guaranteed to exist by flashTest.py

    # read the execution script from "exeScript"
    exeScriptFile = os.path.join(pathToFlashTest, "exeScript")
    if not os.path.isfile(exeScriptFile):
      log.err("File \"exeScript\" not found. Unable to run executable.\n" +
              "Skipping all runs.")
      return False

    lines = open(exeScriptFile).read().split("\n")
    lines = [line.strip() for line in lines
             if len(line.strip()) > 0 and not line.strip().startswith("#")]
    script = "\n".join(lines)
    self.masterDict["script"] = script
    script = self.masterDict["script"]  # do it this way so that any angle-bracket variables
                                        # in "exeScript" will be filled in by self.masterDict
    # determine 'pathToRunSummary'
    pathToRunSummary = os.path.join(pathToRunDir, "run_summary")

    # cd to output directory to run executable
    os.chdir(pathToRunDir)

    # obtain and record number of processors
    if not self.masterDict.has_key("numProcs"):
      self.masterDict["numProcs"] = 1
    open(pathToRunSummary,"a").write("numProcs: %s\n" % self.masterDict["numProcs"])

    # record mpirun invocation in "bcc_call" file and in log
    open(os.path.join(pathToRunDir, "bcc_call"), "w").write(script)
    log.stp(script)

    # get stdout/stderr and duration of execution and write to file
    out, err, duration, exitStatus = getProcessResults(script, timeout)

    open(os.path.join(pathToRunDir, "bcc_output"),"a").write(out)
    if len(err) > 0:
      open(os.path.join(pathToRunDir, "bcc_error"),"a").write(err)

    # record execution time in the run summary and logfile in human-readable form
    duration = secondsToHuman.convert(duration)
    open(pathToRunSummary,"a").write("wallClockTime: %s\n" % duration)
    log.info("duration of execution: %s" % duration)

    # search the parfile output directory for output files
    outputFiles = []
    items = os.listdir(pathToRunDir)
    for item in items:
      if re.match(".*?plot_output.*?.txt$", item):
        outputFiles.append(item)

    # record number and names of output files in the run summary
    open(pathToRunSummary,"a").write("numOutputFiles: %s\n" % len(outputFiles))
    for outputFile in outputFiles:
      open(pathToRunSummary,"a").write("outputFile: %s\n" % outputFile)

    # search the parfile output directory for theory/fit files
    theoryFiles = []
    items = os.listdir(pathToRunDir)
    for item in items:
      if re.match(".*?theory_output.*?.txt$", item):
        theoryFiles.append(item)

    # record number and names of theory/fit files in the run summary
    open(pathToRunSummary,"a").write("numTheoryFiles: %s\n" % len(theoryFiles))
    for theoryFile in theoryFiles:
      open(pathToRunSummary,"a").write("theoryFile: %s\n" % theoryFile)

    # An exit status of 0 means a normal termination without errors.
    if exitStatus == 0:
      log.stp("Process exit-status reports execution successful")
      runSucceeded = True
    else:
      log.stp("Process exit-status reports execution failed")
      runSucceeded = False

    # set entries in masterDict to facilitate testing
    self.masterDict["numOutputFiles"] = len(outputFiles)
    self.masterDict["outputFiles"]    = outputFiles
    self.masterDict["numTheoryFiles"] = len(theoryFiles)
    self.masterDict["theoryFiles"]    = theoryFiles

    # cd back to flashTest
    os.chdir(pathToFlashTest)

    return runSucceeded

#PMR  def getDeletePatterns(self):
#PMR    return [".*_chk_\d+$", ".*_plt_cnt_\d+$"]

# InspectionExecuter doesn't do anything special (yet)
class InspectionExecuter(BCCExecuter):
    pass

# ComparisonExecuter doesn't do anything special (yet)
class ComparisonExecuter(BCCExecuter):
    pass

#PMR   def adjustFilesToDelete(self, filesToDelete):
#PMR     """
#PMR     Determine the highest-numbered checkpoint file and create an
#PMR     entry "chkMax" in masterDict whose value is the name of this
#PMR     file. We'll use this value later to do our sfocu comparison.

#PMR     Then remove this file's name from 'filesToDelete', which will
#PMR     later be used to determine which files will be deleted before
#PMR     creation of the slim copy of the invocation's output.
#PMR     """
#PMR     pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py
#PMR     chkFiles = []

#PMR     # Search 'runDir' for checkpoint files. This method will also
#PMR     # be called for GridDumpComparison problems that do not generate
#PMR     # checkpoint files, but nothing will happen in that case.
#PMR     items = os.listdir(pathToRunDir)
#PMR     for item in items:
#PMR       if re.match(".*?_chk_\d+$", item):
#PMR         chkFiles.append(item)

#PMR     # sorting and reversing will put the highest-numbered
#PMR     # checkpoint file at index 0
#PMR     chkFiles.sort()
#PMR     chkFiles.reverse()

#PMR     if len(chkFiles) > 0:
#PMR       chkMax = chkFiles[0]
#PMR       self.masterDict["chkMax"] = chkMax
#PMR       for fileToDelete in filesToDelete[:]:
#PMR         if fileToDelete == chkMax:
#PMR           filesToDelete.remove(fileToDelete)


class InspectionTester(TesterTemplate):

  def test(self):
    log          = self.masterDict["log"]           # guaranteed to exist by flashTest.py
    outfile      = self.masterDict["outfile"]       # guaranteed to exist by flashTest.py
    log.stp("Inspection test; will rely on human inspection to judge test results.")
    outfile.write("Inspection test; will rely on human inspection to judge test results.")
    return True


class ComparisonTester(TesterTemplate):

  def compare(self, pathToFileA, pathToFileB, cmd):
    log                = self.masterDict["log"]                # guaranteed to exist by flashTest.py
    arch               = self.masterDict["arch"]               # guaranteed to exist by flashTest.py
    outfile            = self.masterDict["outfile"]            # guaranteed to exist by flashTest.py
    pathToRunDir       = self.masterDict["pathToRunDir"]       # guaranteed to exist by flashTest.py
    pathToLocalArchive = self.masterDict["pathToLocalArchive"] # guaranteed to exist by flashTest.py

    pathToFileA = os.path.normpath(pathToFileA)
    pathToFileB = os.path.normpath(pathToFileB)

    if not os.path.isabs(pathToFileA):
      pathToFileA = os.path.join(pathToRunDir, pathToFileA)
    if not os.path.isabs(pathToFileB):
      pathToFileA = os.path.join(pathToRunDir, pathToFileB)

    if pathToFileA.startswith(pathToLocalArchive):
      try:
        arch.confirmInLocalArchive(pathToFileA)
      except Exception, e:
        log.err("%s\n" % e +
                "Aborting this test.")
        outfile.write(str(e))
        return False
    elif not os.path.isfile(pathToFileA):
      log.stp("\"%s\" does not exist." % pathToFileA)
      outfile("\"%s\" does not exist.\n" % pathToFileA)
      return False

    if pathToFileB.startswith(pathToLocalArchive):
      try:
        arch.confirmInLocalArchive(pathToFileB)
      except Exception, e:
        log.err("%s\n" % e +
                "Aborting this test.")
        outfile.write(str(e))
        return False
    elif not os.path.isfile(pathToFileB):
      log.stp("\"%s\" does not exist." % pathToFileB)
      outfile.write("\"%s\" does not exist.\n" % pathToFileB)
      return False

    log.stp("FileA: \"%s\"\n" % pathToFileA +
            "FileB: \"%s\""   % pathToFileB)
    outfile.write("FileA: \"%s\"\n" % pathToFileA +
                  "FileB: \"%s\"\n\n" % pathToFileB)

    outfile.write("script: %s\n" % cmd)
    return getProcessResults(cmd)


#PMR  def compareToYesterday(self, pathToFile, pathToCompareExecutable):
#PMR    yesterDate = time.strftime("%Y-%m-%d", time.localtime(time.time()-24*60*60))

#PMR    pat1 = re.compile("\/\d\d\d\d-\d\d-\d\d.*?\/")
#PMR    pathToYesterFile = pat1.sub("/%s/" % yesterDate, pathToFile)

#PMR    cmd = "%s %s %s" % (pathToCompareExecutable, pathToFile, pathToYesterFile)
#PMR    return self.compare(pathToFile, pathToYesterFile, cmd)


class L2DiffTester(ComparisonTester):

  def compareToTheory(self, pathToTheoryFile, pathToOutputFile, errTol, l2diffScript):
    log          = self.masterDict["log"]           # guaranteed to exist by flashTest.py
    pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py
    outfile      = self.masterDict["outfile"]       # guaranteed to exist by flashTest.py

    pathToTheoryFile = os.path.normpath(pathToTheoryFile)
    pathToOutputFile = os.path.normpath(pathToOutputFile)

    if not os.path.isabs(pathToTheoryFile):
      pathToTheoryFile = os.path.join(pathToRunDir, pathToTheoryFile)
    if not os.path.isabs(pathToOutputFile):
      pathToOutputFile = os.path.join(pathToRunDir, pathToOutputFile)

    if not os.path.isfile(pathToTheoryFile):
      log.stp("\"%s\" does not exist." % pathToTheoryFile)
      outfile("\"%s\" does not exist.\n" % pathToTheoryFile)
      return False

    if not os.path.isfile(pathToOutputFile):
      log.stp("\"%s\" does not exist." % pathToOutputFile)
      outfile("\"%s\" does not exist.\n" % pathToOutputFile)
      return False

    # expect that plot data file will contain errors in third column
    cmd = " ".join([l2diffScript, "-t", str(errTol), "-e", "3", \
                    pathToTheoryFile, pathToOutputFile])

    outfile.write("command: %s\n" % cmd)
    return getProcessResults(cmd)

  def test(self):
    log          = self.masterDict["log"]           # guaranteed to exist by flashTest.py
    pathToBCC    = self.masterDict["pathToBCC"]     # guaranteed to exist by flashTest.py
    pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py
    outfile      = self.masterDict["outfile"]       # guaranteed to exist by flashTest.py

    if self.masterDict["numOutputFiles"] == 0:
      log.stp("No output files were produced, so no comparisons can be made.")
      outfile.write("No output files were produced, so no comparisons can be made.\n")
      return False

    if self.masterDict["numTheoryFiles"] == 0:
      log.stp("No theory/fit files were produced, so no comparisons can be made.")
      outfile.write("No theory/fit files were produced, so no comparisons can be made.\n")
      return False

    # must be one theory/fit file for each output file
    if self.masterDict["numOutputFiles"] <> self.masterDict["numTheoryFiles"]:
      log.stp("Number of output files does not match number of theory/fit files!")
      outfile.write("Number of output files does not match number of theory/fit files!\n")
      return False

    # else

    if self.masterDict.has_key("errTol"):
      errTol = self.masterDict["errTol"]
    else:
      errTol = 0.

    pathToL2Diff = self.masterDict.get("pathToL2Diff", os.path.join(pathToBCC, "tools", "l2diff.py"))
    l2diffScript = self.masterDict.get("l2diffScript", pathToL2Diff)

    trueIfAllSucceeded = True

    for outputFile, theoryFile in zip(self.masterDict["outputFiles"], self.masterDict["theoryFiles"]):

      pathToOutputFile = os.path.join(pathToRunDir, outputFile)
      pathToTheoryFile = os.path.join(pathToRunDir, theoryFile)

      # compare to theory/fit result
      log.stp("Compare result %s to theory/fit %s " % (outputFile, theoryFile))
      outfile.write("Compare result %s to theory/fit %s\n" % (outputFile, theoryFile))

      retval = self.compareToTheory(pathToTheoryFile, pathToOutputFile, errTol, l2diffScript)

      if retval:
        # unpack the tuple
        out, err, duration, exitStatus = retval

        # An exit status of 0 means a normal termination without errors.
        if exitStatus == 0:
          log.stp("Process exit-status reports l2diff ran successfully")
          outfile.write("<b>l2diff output:</b>\n"
                        + out.strip() + "\n\n")

          # Even if l2diff ran fine, the test might still have failed
          # if the two files were not equivalent      
          if out.strip().endswith("SUCCESS"):
            log.stp("comparison of files yielded: SUCCESS")
          else:
            log.stp("comparison of files yielded: FAILURE")
            # Set key "disagreesWithTheory" in masterDict (the value doesn't
            # matter) which is recognized by flashTest.py as a signal to
            # add a "!" to the ends of the "errors" files at the run,
            # build, and invocation levels.
            self.masterDict["disagreesWithTheory"] = True
            trueIfAllSucceeded = False
        else:
          log.stp("Process exit-status reports l2diff encountered an error")
          # record whatever we got anyway
          outfile.write("Process exit-status reports l2diff encountered an error\n" +
                        "<b>l2diff output:</b>\n" +
                        out.strip() + "\n\n")
          trueIfAllSucceeded = False
      else:
        trueIfAllSucceeded = False

    return trueIfAllSucceeded

class FitTester(ComparisonTester):

  def compareToFit(self, pathToFitFile, pathToOutputFile, fitdiffScript):
    log          = self.masterDict["log"]           # guaranteed to exist by flashTest.py
    pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py
    outfile      = self.masterDict["outfile"]       # guaranteed to exist by flashTest.py

    pathToFitFile    = os.path.normpath(pathToFitFile)
    pathToOutputFile = os.path.normpath(pathToOutputFile)

    if not os.path.isabs(pathToFitFile):
      pathToFitFile = os.path.join(pathToRunDir, pathToFitFile)
    if not os.path.isabs(pathToOutputFile):
      pathToOutputFile = os.path.join(pathToRunDir, pathToOutputFile)

    if not os.path.isfile(pathToFitFile):
      log.stp("\"%s\" does not exist." % pathToFitFile)
      outfile("\"%s\" does not exist.\n" % pathToFitFile)
      return False

    if not os.path.isfile(pathToOutputFile):
      log.stp("\"%s\" does not exist." % pathToOutputFile)
      outfile("\"%s\" does not exist.\n" % pathToOutputFile)
      return False

    # At the moment we just compare using the first two columns of the output
    # file and don't use error bars
    cmd = " ".join([fitdiffScript, pathToOutputFile, "1 2", pathToFitFile])

    outfile.write("command: %s\n" % cmd)
    return getProcessResults(cmd)

  def test(self):
    log          = self.masterDict["log"]           # guaranteed to exist by flashTest.py
    pathToBCC    = self.masterDict["pathToBCC"]     # guaranteed to exist by flashTest.py
    pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py
    outfile      = self.masterDict["outfile"]       # guaranteed to exist by flashTest.py

    if self.masterDict["numOutputFiles"] == 0:
      log.stp("No output files were produced, so no comparisons can be made.")
      outfile.write("No output files were produced, so no comparisons can be made.\n")
      return False

    if self.masterDict["numTheoryFiles"] == 0:
      log.stp("No theory/fit files were produced, so no comparisons can be made.")
      outfile.write("No theory/fit files were produced, so no comparisons can be made.\n")
      return False

    # must be one theory/fit file for each output file
    if self.masterDict["numOutputFiles"] <> self.masterDict["numTheoryFiles"]:
      log.stp("Number of output files does not match number of theory/fit files!")
      outfile.write("Number of output files does not match number of theory/fit files!\n")
      return False

    # else

    pathToFitDiff = self.masterDict.get("pathToFitDiff", os.path.join(pathToBCC, "tools", "fitdiff.py"))
    fitdiffScript = self.masterDict.get("fitdiffScript", pathToFitDiff)

    trueIfAllSucceeded = True

    for outputFile, fitFile in zip(self.masterDict["outputFiles"], self.masterDict["theoryFiles"]):

      pathToOutputFile = os.path.join(pathToRunDir, outputFile)
      pathToFitFile    = os.path.join(pathToRunDir, fitFile)

      # compare to theory/fit result
      log.stp("Compare result %s to theory/fit %s " % (outputFile, fitFile))
      outfile.write("Compare result %s to theory/fit %s\n" % (outputFile, fitFile))

      retval = self.compareToFit(pathToFitFile, pathToOutputFile, fitdiffScript)

      if retval:
        # unpack the tuple
        out, err, duration, exitStatus = retval

        # An exit status of 0 means a normal termination without errors.
        if exitStatus == 0:
          log.stp("Process exit-status reports fitdiff ran successfully")
          outfile.write("<b>fitdiff output:</b>\n"
                        + out.strip() + "\n\n")

          # Even if fitdiff ran fine, the test might still have failed
          # if the two files were not equivalent      
          if out.strip().endswith("SUCCESS"):
            log.stp("comparison of files yielded: SUCCESS")
          else:
            log.stp("comparison of files yielded: FAILURE")
            # Set key "disagreesWithTheory" in masterDict (the value doesn't
            # matter) which is recognized by flashTest.py as a signal to
            # add a "!" to the ends of the "errors" files at the run,
            # build, and invocation levels.
            self.masterDict["disagreesWithTheory"] = True
            trueIfAllSucceeded = False
        else:
          log.stp("Process exit-status reports fitdiff encountered an error")
          # record whatever we got anyway
          outfile.write("Process exit-status reports fitdiff encountered an error\n" +
                        "<b>fitdiff output:</b>\n" +
                        out.strip() + "\n\n")
          trueIfAllSucceeded = False
      else:
        trueIfAllSucceeded = False

    return trueIfAllSucceeded
