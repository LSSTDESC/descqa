import os, re
import littleParser, xmlNode

class BigBoard:
  """
  Encapsulates all data related to a top-level visualization of a
  FlashTest output directory, that is, the big board with the red
  and green lights.  The path to the directory whose contents are
  thus represented is contained in the 'pathToOutdir' member.

  A call to a BigBoard instance's 'quickRegenerate()' method will
  notice site directories added to or removed from 'pathToOutdir'
  since the last time the table was generated and incorporate the
  necessary changes into the table. It will also check known site
  directories for newly-added or removed invocation directories,
  again making changes if necessary. It will *not* notice changes
  made inside an invocation directory of a known site directory,
  because this does not affect the site directory's last-modified
  time. Still, a call to quickRegenerate() should ensure that all
  ordinary changes to a FlashTest output directory are accurately
  reflected in the output of the corresponding BigBoard instance.

  A call to a BigBoard instance's 'spewHtml()' method with a file
  object as an input parameter will write to that file object the
  html that produces the table. 
  """

  def __init__(self, pathToOutdir):
    self.needsSorting     = False  # this BigBoard's columns need re-alphabetizing
    self.needsRealignment = False  # this BigBoard's rows need matching up
    self.invocationLists  = []     # contains instances of class InvocationList
    self.lastModified     = None   # will be set in quickRegenerate()
    self.numRows          = 0
    self.pathToOutdir     = pathToOutdir
    self.quickRegenerate()

  def fullRegenerate(self):
    # eliminate all previous structure inside this instance so
    # that each site directory will be regenerated from scratch
    self.__init__(self.pathToOutdir)

  def findMatchingInvocationList(self, siteDir):
    for invocationList in self.invocationLists:
      if invocationList.siteDir == siteDir:
        return invocationList
    return None

  def getInvocationName(self, index):
    if index >= self.numRows:
      index = self.numRows-1
    return self.invocationLists[0].invocations[index].name

  def isOutOfDate(self):
    lastModified = str(os.stat(self.pathToOutdir)[8])
    if lastModified != self.lastModified:
      # a directory has been added or removed
      return True
    # Now we know self.invocationLists is still an accurate
    # representation of the directories in 'pathToOutdir'
    for invocationList in self.invocationLists:
      if invocationList.isOutOfDate():
        return True
    return False

  def realignGrid(self):
    # put all unique names of invocations from all sites into one dictionary
    bigDict = {}
    for invocationList in self.invocationLists:
      for invocation in invocationList.invocations:
        # the key's associated value doesn't matter
        bigDict[invocation.name] = None

    allInvocations = bigDict.keys()
    allInvocations.sort(lambda x,y:-cmp(x,y))  # sort in reverse order

    i=0
    while i < len(allInvocations):
      for invocationList in self.invocationLists:
        if ((i >= len(invocationList.invocations)) or
            (invocationList.invocations[i].name != allInvocations[i])):
          # insert an Invocation instance with the correct name, so
          # it will stay in the correct place when 'invocationList'
          # is sorted, but with no html, so it will produce no link
          # on the big board
          invocationList.invocations.insert(i, Invocation(allInvocations[i], ""))
          # If an invocation directory is removed, it may result in
          # an entire row of these "empty" Invocation instances, so
          # we'll check for any such empty row and delete it.
          for invocationList2 in self.invocationLists:
            if ((i >= len(invocationList2.invocations)) or
                (invocationList2.invocations[i].name != allInvocations[i]) or
                (invocationList2.invocations[i].html != "")):
              break
          else:
            # every invocation list has an invocation with this name
            # and each of these invocations has the empty string for
            # its 'html' member - it's an empty row
            for invocationList2 in self.invocationLists:
              del invocationList2.invocations[i]
            del allInvocations[i]
            i -= 1  # since we've just deleted a row, kludgily set the
                    # index variable back to counter the upcoming increment
      i+=1

    self.numRows = len(allInvocations)

  def quickRegenerate(self):

    if self.isOutOfDate():
      self.lastModified = str(os.stat(self.pathToOutdir)[8])

      # Get list of all directories in 'pathToOutdir'. These are assumed
      # to be site directories, i.e. directories holding all output from
      # a single platform. The name of the site will appear in the green
      # stripe along the top edge of the big board.
      siteDirs = [item for item in os.listdir(self.pathToOutdir)
                  if os.path.isdir(os.path.join(self.pathToOutdir, item))]

      # eliminate any sites in 'invocationLists' no longer in 'pathToOutdir'
      for invocationList in self.invocationLists[:]:
        if invocationList.siteDir not in siteDirs:
          self.invocationLists.remove(invocationList)

      # add any sites newly found in 'pathToOutdir'
      for siteDir in siteDirs:
        invocationList = self.findMatchingInvocationList(siteDir)
        if invocationList:
          # Quick-regenerate this list. A return value of True means
          # that changes were made, and the big board needs realignment.
          if invocationList.quickRegenerate() == True:
            self.needsRealignment = True
        else:
          # this site has been added since the last quickRegenerate()
          self.invocationLists.append(InvocationList(self.pathToOutdir, siteDir))
          self.needsSorting = True

      if self.needsSorting:
        self.invocationLists.sort()
        self.needsSorting = False
        self.needsRealignment = True

      if self.needsRealignment:
        self.realignGrid()
        self.needsRealignment = False

      return True  # changes were made

    else:
      # No changes have been made to the contents of 'self.pathToOutdir' or to
      # any of its immediate subdirectories. Return False for "no changes made"
      return False

  def spewHtml(self, outfile, startRow, endRow):
    """
    Generate html corresponding to this big board instance
    """
    # Make a list of invocation lists that contain at least one invocation
    # in the range we're working with (from 'startRow' to 'endRow'). We'll
    # use this to make a display with no empty columns.
    representedInvocationLists = []
    for invocationList in self.invocationLists:
      for i in range(startRow, endRow+1):
        if len(invocationList.invocations[i].html) > 0:
          representedInvocationLists.append(invocationList)
          break  # we found one example, so we don't need to search anymore

    # figure out which "test.info" file is responisble for each platform
    # by examining the copy of 'masterDict' in the latest invocation
    masterNode = None
    pathToInfo = None

    lines = []
    lines.append("<table border=\"0\" width=\"100%\" cellspacing=\"0\">")

    # make the green row with names of the sites
    lines.append("<tr>")

    for invocationList in representedInvocationLists:
      # find first row that's not a blank
      firstNonBlankRow = -1
      for i in range(startRow, endRow+1):
        if invocationList.invocations[i].html:
          firstNonBlankRow = i
          break

      if firstNonBlankRow >= 0:
        pathToMasterDict = os.path.join(invocationList.pathToOutdir, invocationList.siteDir,
                                        invocationList.invocations[firstNonBlankRow].name, "masterDict")
        if os.path.isfile(pathToMasterDict):
          masterDict = littleParser.parseFile(pathToMasterDict)
          # flashTest.py guarantees that 'masterDict' will contain a value for
          # "pathToInfo".
          masterDictPathToInfo = masterDict["pathToInfo"]
          if not os.path.isabs(masterDictPathToInfo) and os.name == "posix":
            # assume that person who originally ran this invocation of FlashTest
            # used an info-file in his/her own home directory on this machine
            owner = masterDict.get("owner","")
            masterDictPathToInfo = os.path.join(os.path.expanduser("~" + owner), masterDictPathToInfo)

          if masterDictPathToInfo != pathToInfo:
            # this is a new 'pathToInfo' from what we had before, so parse up a
            # new 'masterNode'. Usually, all invocations on one big-board will
            # derive from one "test.info" and we'll only need to parse it once.
            pathToInfo = masterDictPathToInfo
            try:
              masterNode = xmlNode.parseXml(pathToInfo)
            except:
              masterNode = None
          else:
            pass  # 'masterNode' remains the same as in the last pass of the for loop

      lines.append("<td class=\"green\">")

      siteDir = invocationList.siteDir
      if masterNode and masterNode.findChild(siteDir):
        # clicking on this site's name will take you to that
        # site's info files in the info file manager
        link = "%s?path_to_info=%s&start_node=%s" % (os.path.join("infoFileManager", "home.py"), pathToInfo, siteDir)
        if invocationList.floatLines:
          statsHeader = "stats for %s" % siteDir
          statsBody   = "<br>".join(invocationList.floatLines)
          html = ("<a href=\"%s\"" % link +
                  "onMouseOver=\"appear('%s','%s')\" " % (statsHeader, statsBody) +
                  "onMouseOut=\"disappear()\">%s</a>" % siteDir)
          lines.append(html)
        else:
          lines.append("<a href=\"%s\" title=\"edit &quot;test.info&quot; files for %s\">%s</a>" % (link, siteDir, siteDir))
      else:
        if invocationList.floatLines:
          statsHeader = "stats for %s" % siteDir
          statsBody   = "<br>".join(invocationList.floatLines)
          html = ("<span onMouseOver=\"appear('%s','%s')\" " % (statsHeader, statsBody) +
                  "onMouseOut=\"disappear()\">%s</span>" % siteDir)
          lines.append(html)
        else:
          lines.append(siteDir)

      lines.append("</td>")

    lines.append("</tr>")

    # make the white and blue rows with the links to individual invocations
    i=startRow
    while i <= endRow:
      if i%2 == 0:
        rowClass = "white"
      else:
        rowClass = "blue"
      lines.append("<tr>")
      for invocationList in representedInvocationLists:
        lines.append("<td class=\"%s\">" % rowClass)
        lines.append(invocationList.invocations[i].html)
        lines.append("</td>")
      lines.append("</tr>")
      i+=1

    lines.append("</table>")
    outfile.write("\n".join(lines))


class InvocationList:
  """
  Encapsulates all data specific to FlashTest invocations on a given platform.
  Instances of InvocationList populate a BigBoard instance's 'invocationLists'
  member. When the big board is visualized, each InvocationList instance will
  produce a single column of data with the name of the platform at the top and
  data from each invocation on that platform listed beneath.
  """
  def __init__(self, pathToOutdir, siteDir):
    self.invocations   = []
    self.lastModified  = None      # will be set in quickRegenerate()
    self.pathToOutdir  = pathToOutdir
    self.siteDir       = siteDir   # name of site connected to these invocations
    self.pathToSiteDir = os.path.join(self.pathToOutdir, self.siteDir)
    self.floatLines    = ""
    self.quickRegenerate()

  def __cmp__(self, other):
    """
    When InvocationList instances are sorted by a BigBoard instance,
    they will be sorted according to the name of their siteDir member.
    """
    return cmp(self.siteDir, other.siteDir)

  def isOutOfDate(self):
    """
    Compare this instance's 'lastModified' member to the time of
    last modification on the site directory.
    """
    if self.lastModified != str(os.stat(self.pathToSiteDir)[8]):
      return True
    # else
    return False

  def quickRegenerate(self):
    """
    Adust the contents of this instance's invocations list if
    its 'lastModified' member doesn't match the last modified
    time of its associcated site directory.
    """

    if self.isOutOfDate():
      self.lastModified = str(os.stat(self.pathToSiteDir)[8])

      datePat = re.compile("^\d\d\d\d-\d\d-\d\d.*")
      GREEN   = 0
      YELLOW  = 1
      RED     = 2

      pathToFloatText = os.path.join(self.pathToSiteDir, "float_text")
      if os.path.isfile(pathToFloatText):
        self.floatLines = open(pathToFloatText).read().split("\n")

      invocationDirs = [item for item in os.listdir(self.pathToSiteDir)
                        if os.path.isdir(os.path.join(self.pathToSiteDir, item))]

      # cull out directories that don't fit the pattern
      invocationDirs = [invocationDir for invocationDir in invocationDirs
                        if datePat.match(invocationDir)]

      # cull out directories that contain a ".lock" file, which
      # means that this invocation is currently being written to
      invocationDirs = [invocationDir for invocationDir in invocationDirs
                        if not os.path.isfile(os.path.join(self.pathToSiteDir, invocationDir, ".lock"))]

      # check for any Invocation instances in 'self.invocations'
      # not present in 'invocationDirs' and delete them
      for invocation in self.invocations[:]:
        if invocation.name not in invocationDirs:
          self.invocations.remove(invocation)

      # check for new invocation dirs not present in 'self.invocations'
      # and append a new Invocation instance encapsulating that data

      newInvocations = []  # we'll temporarily hold any added invocations in here,
                           # then tack them all on at once when we're finished. If
                           # we put them in self.invocations right away, each new
                           # invocation would be pointlessly examined over and over
                           # again in the inner for-loop two lines down from here
      if self.invocations == []:
        mostRecent = ""
      else:
        mostRecent=sorted(self.invocations)[0]	
      	
      for invocationDir in invocationDirs:
       try:
        for invocation in self.invocations[:]:
          if invocation.name == invocationDir:
            if invocation.html == "" or invocation.name == mostRecent.name:
              # this is a blank "place-holder" invocation that
              # needs to be replaced by a real one, so remove it
	      # Also remove most recent just in case that testrun is updated on the view site incrementally
              self.invocations.remove(invocation)
            else:
              # there's already an invocation with this name, and
              # it's not an empty "place-holder", so break out of
              # this inner for-loop (skipping the 'else' clause below)
              break
        else:
          errorLines  = []  # lines of text which will populate the floating stats window
          totalErrors = 0   # number of builds that had an error at some stage
          totalRedErrors = 0 # number of builds that had an error except for
                             # failure in testing "as before"
          totalBuilds = 0   # total number of builds attempted on this invocation

          logErrors   = 0   # number of times the invocation logfile recorded an error
          logWarnings = 0   # number of times the invocation logfile recorded a warning

          lightColor  = GREEN          # Assume a green light and no changes from
          changedFromPrevious = False  # previous invocation unless proven otherwise

          statsHeader = ""  # The darker, upper portion of the floating stats window,
                            # it holds total numbers of failed builds (setup & compile
                            # phases) and runs (execution and testing phases)
          statsBody   = ""  # The lighter, lower portion of the floating stats window,
                            # it reports total errors found in the log and expands on
                            # errors during all four phases of testing.

          pathToInvocationDir = os.path.join(self.pathToSiteDir, invocationDir)

          # we assume a directory inside 'pathToInvocationDir' represents a build
          items = os.listdir(pathToInvocationDir)
          totalBuilds = len([item for item in items
                             if os.path.isdir(os.path.join(pathToInvocationDir, item))])

          errorsFile = os.path.join(pathToInvocationDir, "errors")

          if os.path.isfile(errorsFile):
            errorLines = open(errorsFile).read().strip().split("\n")
            errorLines = [errorLine.strip() for errorLine in errorLines if len(errorLine.strip()) > 0]

            # An exclamation mark at the end of any build information means that
            # that build showed some kind of change from the previous invocation.
            for errorLine in errorLines:
              if errorLine.endswith("!"):
                changedFromPrevious = True
              if not ((errorLine.find(" failed in testing as before") > 0) and
                      (errorLine.find(" failed in execution") < 0)):
                totalRedErrors = totalRedErrors + 1
              if changedFromPrevious and (totalRedErrors > 0):
                break  # we only need one "!" to generate a "!" on the big board, and
                       # one without " as before" to be sure the light should be
                       # red not yellow.

            totalErrors = len(errorLines)
            if totalErrors == 0:
              statsHeader = "All %s tests completed successfully" % totalBuilds
            else:
              statsHeader = "%s/%s tests had some error" % (totalErrors, totalBuilds)
          else:
            statsHeader = "No errors file found for this invocation"

          logFile = os.path.join(pathToInvocationDir, "flash_test.log")
          if os.path.isfile(logFile):
            logLines = open(logFile).readlines()
            for logLine in logLines:
              if logLine.startswith("ERROR:"):
                logErrors+=1
              elif logLine.startswith("WARNING:"):
                logWarnings+=1

            if logErrors + logWarnings == 0:
              logMsg = "Logfile recorded no errors or warnings"
            else:
              logMsg = ""
              if logErrors > 0:
                if logErrors == 1:
                  logMsg += "Logfile recorded 1 error"
                else:
                  logMsg += "Logfile recorded %s errors" % logErrors
              if logWarnings > 0:
                if logErrors > 0:
                  if logWarnings == 1:
                    logMsg += ", 1 warning"
                  else:
                    logMsg += ", %s warnings" % logWarnings
                else:
                  if logWarnings == 1:
                    logMsg += "Logfile recorded 1 warning"
                  else:
                    logMsg += "Logfile recorded %s warnings" % logWarnings
          else:
            logMsg = "No logfile found for this invocation"

          if logErrors + len(errorLines) > 0:
            if logErrors + totalRedErrors == 0:
              lightColor = YELLOW
            else:
              lightColor = RED

          errorLines.append(logMsg)
          statsBody = "<br>".join(errorLines)

          html = ("<a href=viewer/viewBuilds.cgi?target_dir=%s " % pathToInvocationDir +
                  "onMouseOver=\"appear('%s','%s')\" " % (statsHeader, statsBody) +
                  "onMouseOut=\"disappear()\">%s</a>" % invocationDir)
          if lightColor == GREEN:
            html += "&nbsp;<img src=\"images/green.gif\">"
          elif lightColor == YELLOW:
            html += "&nbsp;<img src=\"images/yellow.gif\">"
          else:
            html += "&nbsp;<img src=\"images/red.gif\">"

          if changedFromPrevious:
            html += "&nbsp;<b>!</b>"

          newInvocations.append(Invocation(invocationDir, html))

       except Exception,e:
        print "Exception:",e,"; continuing"


      if len(newInvocations) > 0:
        self.invocations.extend(newInvocations)
        self.invocations.sort()

      return True  # indicates changes were made
    else:
      # No changes have been made to the immediate contents of 'self.pathToSiteDir'
      return False


class Invocation:
  """
  encapsulates data visible for a single FlashTest invocation, (a
  date, possibly with a suffix) at the top level of FlashTestView
  """
  def __init__(self, name, html):
    self.name = name
    self.html = html  # all html and javascript that goes inside
                      # the table cell for this invocation

  def __cmp__(self, other):
    return -cmp(self.name, other.name)  # Invocation instances will sort so that the
                                        # names appear in reverse lexicographical order
                                        # (b/c we want the most recent date at the top)
