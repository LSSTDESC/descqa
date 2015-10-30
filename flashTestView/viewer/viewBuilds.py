#!/usr/bin/env python
import sys, os
import cgi
sys.path.insert(0, "../lib")
import customMenuClasses, littleParser, ezt

class LinkAttribute:
  """
  encapsulates a key/value pair for inclusion in the
  hyperlink element associated with a single build
  """
  def __init__(self, key, value):
    self.key = key
    self.value = value

class BuildTuple:
  """
  encapsulates one FLASH build. A list of BuildTuples
  will form part of the data dictionary passed to the
  ezt template
  """
  def __init__(self, infoSource, testName, failed, pathToBuild, linkAttributes, build, exitStatus):
    self.infoSource     = infoSource      # path to "test.info", if any
    self.testName       = testName        # e.g. Comparison
    self.failed         = failed          # None if test succeeded, else True
    self.pathToBuild    = pathToBuild     # e.g. /.../output/Comparison_Advect...
    self.linkAttributes = linkAttributes  # list of LinkAttribute instances
    self.build          = build           # e.g. Comparison_Advect_pm3_hdf5...
    self.exitStatus     = exitStatus      # status summary

class CustomMenuBox:
  """
  encapsulates the divHeader, insertJs, and insertHtml
  members of an instantiated TestObject
  """
  def __init__(self, testName, divHeader, insertJs, insertHtml):
    self.testName   = testName
    self.divHeader  = divHeader
    self.insertJs   = insertJs
    self.insertHtml = insertHtml

buildTuples   = []

allTests      = {} # dictionary of all tests represented in this batch
                   # of builds, with each test name keyed to a list of
                   # paths to that test's associated build directories.

infoFileTests = {} # subset of 'allTests', with test names keyed to lists
                   # of associated build directories, but only when those
                   # directories have an associated "test.info" file.

links         = [] # links to certain text files in the build directory.

# -------------- form data ---------------- #
form = cgi.FieldStorage()
pathToTargetDir = form.getvalue("target_dir")
displayExitStat = form.getvalue("displayExitStat")
displayTestName = form.getvalue("displayTestName")
orderBy         = form.getvalue("orderBy")

exitStatConditions = []
testNameConditions = []
orderByConditions  = []

if displayExitStat:
  testNameConditions.append("displayExitStat=%s" % displayExitStat)
  orderByConditions.append("displayExitStat=%s" % displayExitStat)
if displayTestName:
  exitStatConditions.append("displayTestName=%s" % displayTestName)
  orderByConditions.append("displayTestName=%s" % displayTestName)
if orderBy:
  exitStatConditions.append("orderBy=%s" % orderBy)
  testNameConditions.append("orderBy=%s" % orderBy)

pathToInfo = ""
pathToMasterDict = os.path.join(pathToTargetDir, "masterDict")
if os.path.isfile(pathToMasterDict):
  masterDict = littleParser.parseFile(pathToMasterDict)
  # flashTest.py guarantees that 'masterDict' will contain a value for
  # "pathToInfo".
  pathToInfo = masterDict["pathToInfo"]

# the data dictionary we will pass to the ezt template
templateData = {}

templateData["thisBuild"]          = "viewBuilds.py?target_dir=%s" % pathToTargetDir
templateData["pathToInfo"]         = pathToInfo
templateData["date"]               = os.path.basename(pathToTargetDir)
templateData["site"]               = os.path.basename(os.path.dirname(pathToTargetDir))
templateData["testNameConditions"] = "&".join(testNameConditions)
templateData["exitStatConditions"] = "&".join(exitStatConditions)
templateData["orderByConditions"]  = "&".join(orderByConditions)

# DEV If target dir has been deleted since the time the big board
# was last regenerated, this line will just cause an error. Instead
# there should be a "dir no longer exists message" and a link back
# to the big board. Also there should be a "regenerate" button on
# the big board to the right of "FlashTest Invocations"
items = os.listdir(pathToTargetDir)

# sort list by timestamp or alphabetically
if orderBy == "timestamp":
  # DEV!!
  # this will not be accurate if any modifications are later made by hand
  # to the output directories, better to have a text file showing explicit
  # order in which the builds were done
  items.sort(lambda x,y: cmp(os.stat(os.path.join(pathToTargetDir, x))[8], os.stat(os.path.join(pathToTargetDir, y))[8]))
else:
  items.sort()

for item in items:
  # if the item is a dir, it should represent one build, i.e. setup,
  # compilation, and run of executable against 1 thru n parfiles
  pathToItem = os.path.join(pathToTargetDir, item)
  if os.path.isdir(pathToItem):
    
    # check for a copy of a "test.info" block
    infoFile   = os.path.join(pathToItem, "test.info")
    infoSource = ""  # we'll overwrite this value if we can
    if os.path.isfile(infoFile):
      # the first line will be a path to the original "test.info" file
      infoSource = open(infoFile,"r").readline().strip()

    # DEV! This means tests may not have an underscore in their name
    # DEV This is also bad because it assigns special meaning to the
    # first element in the original test-path, which I don't want to
    # do anymore
    testName = item.split("_",1)[0]
    if allTests.has_key(testName):
      # append this path to an already extant entry for 'testName' in 'allTests'
      allTests[testName].append(pathToItem)
    else:
      # add a new entry for 'testName' in 'allTests'
      allTests[testName] = [pathToItem]

    # assume exit status is good until proven otherwise
    failed = None
    failedSetup=failedCompilation=failedExecution=failedTesting = None
    exitStatus = "all runs completed successfully"

    # read this build's "errors" file
    errorsFile = os.path.join(pathToItem, "errors")
    changedFromPrevious = False

    if os.path.isfile(errorsFile):
      errorLines = open(errorsFile).read().strip().split("\n")

      # A "!" at the end of the errors file means at least
      # one run in this build had testing results different
      # from the same run from the previous invocation.
      if len(errorLines) == 7 and errorLines[6].strip() == "!": # new format with 6(+1) lines
        changedFromPrevious = True
        del errorLines[6]
      elif len(errorLines) == 6 and errorLines[5].strip() == "!": # old format with 5(+1) lines
        changedFromPrevious = True
        del errorLines[5]

      if len(errorLines) == 5:          # old format errors file with (now) 5 lines:
        errorLines.append("0")          # assume 0 for number of "as before" errors

      # change values to ints
      errorLines = [int(errorLine.strip()) for errorLine in errorLines]
      numTestSameErrs = errorLines[5] 
      if errorLines[0] > 0:
        failed = True
	failedSetup=True
        exitStatus = "failed in setup"
      elif errorLines[1] > 0:
        failed = True
	failedCompilation=True
        exitStatus = "failed in compilation"
      else:
        if errorLines[2] > 0:
          failed = True
	  failedExecution=True
          exitStatus = "%s/%s runs failed in execution" % (errorLines[2], errorLines[4])
        if errorLines[3] > 0:
          failed = True
	  failedTesting=True
          if errorLines[2] > 0:
            exitStatus += "; %s/%s failed in testing" % (errorLines[3], errorLines[4])
          else:
            exitStatus = "%s/%s runs failed in testing" % (errorLines[3], errorLines[4])
          if numTestSameErrs > 0:
            if numTestSameErrs == errorLines[3]:
              exitStatus = exitStatus + " as before"
            else:
              exitStatus = "%s, %s as before" % (exitStatus, numTestSameErrs)
    else:
      exitStatus = "\"errors\" file not found"

    if changedFromPrevious:
      exitStatus += ", changed from previous invocation"

    # Custom key/value pairs can be attached to the link for a build
    # via a "linkAttributes" file in the corresponding build directory.
    # These can then be read by a menu element's javascript.
    linkAttributes = []
    pathToLinkAttributesFile = os.path.join(pathToItem, "linkAttributes")
    if os.path.isfile(pathToLinkAttributesFile):
      linkAttributesDict = littleParser.parseFile(pathToLinkAttributesFile)
      for key in linkAttributesDict:
        linkAttributes.append(LinkAttribute(key, linkAttributesDict[key]))

    # skip past tests which are filtered out because the user
    # selected to display only a specific test name and/or only
    # tests which succeeded / failed
    if ((displayTestName) and (displayTestName != testName)):
      continue
    elif ((displayExitStat) and
          ((displayExitStat == "s" and failed) or
           (displayExitStat == "f" and not failed)or
	   (displayExitStat == "fs" and not failedSetup)or
	   (displayExitStat == "fc" and not failedCompilation)or
	   (displayExitStat == "fe" and not failedExecution)or
	   (displayExitStat == "ft" and not failedTesting)	)):
      continue

    # else
    buildTuples.append(BuildTuple(infoSource, testName, failed, pathToItem, linkAttributes, item, exitStatus))

    # represent this test in 'infoFileTests'
    # if it has an associated "test.info" file
    if infoSource:
      infoFileTests[testName] = None

  elif os.path.basename(item) == "flash_test.log":
    # check for errors and warnings in logfile
    totalErrors = 0
    totalWarnings = 0
    lines = open(pathToItem,"r").readlines()
    for line in lines:
      if line.startswith("ERROR:"):
        totalErrors += 1
      elif line.startswith("WARNING:"):
        totalWarnings += 1
    queryStr = "target_file=%s" % os.path.join (pathToTargetDir, "flash_test.log")
    link = "<a href=\"viewTextFile.py?%s\" target=\"_blank\">FlashTest Log</a>" % queryStr
    if totalErrors > 0:
      link += "<font color=\"red\">"
      if totalErrors == 1:
        link += " <i>recorded 1 error</i>"
      else:
        link += " <i>recorded %s errors</i>" % totalErrors
      link += "</font>"
    if totalWarnings > 0:
      link += " <font color=\"orange\">"
      if totalErrors > 0:
        if totalWarnings == 1:
          link += " <i>, 1 warning</i>"
        else:
          link += " <i>, %s warnings</i>" % totalWarnings
      else:
        if totalWarnings == 1:
          link += " <i>recorded 1 warning</i>"
        else:
          link += " <i>recorded %s warnings</i>" % totalWarnings
      link += "</font>"
    # make sure logfile link is the last in the list so
    # messages about errors and warnings can extend to
    # the right without interference
    links.append(link)

  # put this in at the beginning of the list to make sure
  # the logfile stays at the end (see note above)
  elif os.path.basename(item) == "update_output":
    links.insert(0, "<a href=\"viewTextFile.py?target_file=%s\" target=\"_blank\">%s</a>" % (pathToItem, item))


allTestsList      = allTests.keys()
infoFileTestsList = infoFileTests.keys()
allTestsList.sort()
infoFileTestsList.sort()

templateData["allTests"]      = allTestsList
templateData["infoFileTests"] = infoFileTestsList
templateData["links"]         = " | ".join(links)
templateData["buildTuples"]   = buildTuples

customMenuBoxes = []
for infoFileTest in infoFileTestsList:
  if hasattr(customMenuClasses, infoFileTest):
    # 'testObject' is a quick instantion of the class named 'infoFileTest'
    try:
      testObject = getattr(customMenuClasses, infoFileTest)()
      divHeader  = testObject.divHeader
      insertJs   = testObject.insertJs(pathToTargetDir)
      insertHtml = testObject.insertHtml(pathToTargetDir)

      if insertHtml:
        customMenuBoxes.append(CustomMenuBox(infoFileTest, divHeader, insertJs, insertHtml))
    except Exception, e:
      # give the user a clue as to what went wrong
      customMenuBoxes.append(CustomMenuBox(infoFileTest, "error", "", str(e)))
    else:
      del testObject

templateData["customMenuBoxes"] = customMenuBoxes

if os.path.isfile("../config"):
  configDict = littleParser.parseFile("../config")
  siteTitle = configDict.get("siteTitle", [])

print "Content-type: text/html\n"
print "<html>"
print "<head>"
print "<title>%s</title>" % siteTitle
print open("style.css","r").read()
print "<script>"
print open("viewBuilds.js","r").read()

# insert user javascript into <head>

for customMenuBox in customMenuBoxes:
  js = customMenuBox.insertJs
  if js:
    print js + "\n"

  # If the javascript produced by the "insertJs()" method of a custom menu object
  # defines a function <test-name>Submit(), where <test-name> is the name of the
  # the custom menu class, that function will be called when a user clicks that
  # menu box's "submit" button. The function can then decide to proceed or abort
  # as apporpriate.
  #
  # But if the javascript produced by the "insertJs()" method of a custom menu
  # object does *not* define such a function, we supply one which alerts the user
  # to the fact.
  if js.find("%sSubmit" % customMenuBox.testName) < 0:
    print "function %sSubmit() {" % customMenuBox.testName
    print ("  alert('You have not supplied a javascript function %sSubmit\\n" % customMenuBox.testName +
           "to define behavior when the user clicks \"submit\" on this menu box.');")
    print "  return false;"
    print "}"
    print ""

print "</script>"
print "</head>"
# print the html generated by ezt templates
ezt.Template("viewBuildsTemplate.ezt").generate(sys.stdout, templateData)
print "</html>"
