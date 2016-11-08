#!/usr/bin/env python
try:
  open('x','w').write('0')
  import sys, os
  import cgi, pickle, tempfile
  open('x','w').write('0.1')
  import purgeTmp
  sys.path.insert(0, "lib")
  import invocations, littleParser
  open('x','w').write('0.5')

  fileMap = "fileMap"
except:
  import traceback
  traceback.print_exc(file=open('x','w'))
  sys.exit(1)

def abort(msg):
  print "<body>"
  print msg
  print "</body>"
  print "</html>"
  open('x','w').write(msg)
  sys.exit(0)

def rewriteFileMap(fileMapDict):
  # rewrite "fileMap" with new additions/deletions
  text = "# THIS IS A COMPUTER GENERATED FILE! DO NOT EDIT!\n"
  text += "\n".join(["%s: %s" % (key, fileMapDict[key]) for key in fileMapDict.keys()])
  open(fileMap,"w").write(text)


open('x','w').write('1')

# FlashTest's main results board showing red or green lights
# for FlashTest invocations with failures or no failures.
try:

    # first purge all files over 24 hours old from "tmp"
    #purgeTmp.purgeTmp()

    if os.path.isfile("config"):
      configDict = littleParser.parseFile("config")
      siteTitle = configDict.get("siteTitle", [])

    print "Content-type: text/html\n"
    print "<head>"
    print "<title>%s</title>" % siteTitle

    # next three lines ensure browsers don't cache, as caching can cause false
    # appearances of the "please wait while the table is being regenerated" if
    # the user uses the browser's "back" button.
    print "<meta http-equiv=\"cache-control\" content=\"no-cache\">"
    print "<meta http-equiv=\"Pragma\" content=\"no-cache\">"
    print "<meta http-equiv=\"Expires\" content=\"-1\">"

    print open("style.css","r").read()
    print "<script src=\"lib/vanishPleaseWait.js\"></script>"
    print "<script src=\"lib/statsWindow.js\"></script>"
    print "<script src=\"lib/redirect.js\"></script>"
    print "</head>"

    open('x','w').write('3')

    # make sure website has write permissions in this folder
    cwd = os.getcwd()
    if not os.access(cwd, os.W_OK):
      msg = ("The web-server does not have write permissions in directory \"%s\"<br>" % cwd +
             "This permission must be granted for DESCQA to function correctly.")
      open('x','w').write(msg)
      abort(msg)

    # Generate fileMapDict from "fileMap", a text file that maps
    # paths to output directories to their associated ".pick" files.
    if os.path.isfile(fileMap):
      # Paul, uncomment the lines below and delete this line when you've fixed the file permissions
      if not os.access(fileMap, os.R_OK + os.W_OK):
        msg = ("The web-server does not have read and/or write permissions on file \"%s\"<br>" % os.path.join(cwd, fileMap) +
               "This permission must be granted for DESCQA to function correctly.")
        open('x','w').write(msg)
        abort(msg)
      else:
        fileMapDict = littleParser.parseFile(fileMap)
    else:
      fileMapDict = {}

    open('x','w').write('4')
    if os.path.isfile("config"):
      configDict = littleParser.parseFile("config")
      pathsToOutdirs = configDict.get("pathToOutdir", [])  # returns a string if only one value
                                                           # associated with key, else a list

      # Make 'pathToOutdir' into a list if it's not one already.
      # If the list has more than one element, we'll eventually
      # use it to make the drop-down menu that lets the user
      # visualize different collections of FlashTest data
      if not isinstance(pathsToOutdirs, list):
        pathsToOutdirs = [pathsToOutdirs]

      # delete any .pick files whose corresponding path
      # no longer appears in 'configDict' and eliminate
      # the appropriate entry in 'fileMapDict'
      fileMapNeedsRewrite = False
      try:
          for key in fileMapDict.keys()[:]:
            if key not in pathsToOutdirs:
              try:
                os.remove(fileMapDict[key])
              except:
                pass
              del fileMapDict[key]
              fileMapNeedsRewrite = True
      except Exception,e:
        print "exception: ", e

      if fileMapNeedsRewrite:
        rewriteFileMap(fileMapDict)

    else:
      configDict = {}
      pathsToOutdirs = []

    pickFile = ""
    form = cgi.FieldStorage()
    pathToTargetDir = form.getvalue("target_dir")
    thisPageNum = form.getvalue("page")

    if pathToTargetDir:
      if configDict:
        if pathToTargetDir in pathsToOutdirs:
          if not os.path.isdir(pathToTargetDir):
            if fileMapDict.has_key(pathToTargetDir):
              del fileMapDict[pathToTargetDir]
              rewriteFileMap(fileMapDict)
            abort("\"%s\" does not exist or is not a directory." % pathToTargetDir)
        else:
          abort("Directory \"%s\" not listed as a value for key \"pathToOutdir\" in \"config\".<br>" % pathToTargetDir +
                "Add this directory to \"config\" and reload this page.")
      else:
        abort("File \"config\" either does not exist or does not contain any values.<br>"+
              "Create a \"config\" file if necessary and add the following text:<br><br>" +
              "pathToOutdir: %s<br><br>" % pathToTargetDir +
              "Then reload this page.")
    else:
      if configDict:
        if pathsToOutdirs:
          pathToTargetDir = pathsToOutdirs[0]
          if not os.path.isdir(pathToTargetDir):
            if fileMapDict.has_key(pathToTargetDir):
              del fileMapDict[pathToTargetDir]
              rewriteFileMap(fileMapDict)
            abort("\"%s\" as listed in \"config\"<br>" % pathToTargetDir +
                  "does not exist or is not a directory.")
        else:
          abort("You must add at least one value to the key \"pathToOutdir\" in \"config\"<br>" +
                "where that value is a path to a top-level DESCQA output directory.")
      else:
        abort("File \"config\" either does not exist or does not contain any values.<br>" +
              "Create a \"config\" file if necessary and add the following text:<br><br>" +
              "pathToOutdir: [path/to/outdir]<br><br>" +
              "where [path/to/outdir] is an absolute path to a top-level DESCQA  output directory.<br>" +
          "Then reload this page.")
except Exception,e :   
    print "Exception! ", e
    import traceback
    traceback.print_exc(file=sys.stdout)

# At this point we know that 'pathToTargetDir' is defined, that it is
# an extant directory, and that that directory is listed in "config"
try:
    shouldGenerate=False
    #shouldGenerate=True
    if fileMapDict.has_key(pathToTargetDir):
      pickFile = fileMapDict[pathToTargetDir]
      if os.path.exists(pickFile):
        bigBoard = pickle.load(open(pickFile))
        if bigBoard.isOutOfDate():
          print "<body onLoad=\"vanishPleaseWait(); statsWindowInit()\">"
          print "<div id=\"pleasewait\">"
          print "DESCQA has generated new data since the last time this page was viewed.<br>"
          print "Please wait while the table is being regenerated."
          print "</div>"
          sys.stdout.flush()
          bigBoard.quickRegenerate()
          pickle.dump(bigBoard, open(pickFile, "w"))
        else:
          print "<body onLoad=\"statsWindowInit()\">"
      else:
          # pickFile does not exist; generate
          shouldGenerate=True

    if shouldGenerate:
      print "<body onLoad=\"vanishPleaseWait(); statsWindowInit()\">"
      print "<div id=\"pleasewait\">"
      print "Please wait while DESCQA generates a table for \"%s\"." % pathToTargetDir
      print "</div>"
      sys.stdout.flush()
      bigBoard = invocations.BigBoard(pathToTargetDir)
      newFile, newFileName = tempfile.mkstemp(suffix=".pick", prefix="", dir=os.getcwd())
      os.chmod(newFileName, 256 + 32 + 4 + 128 + 16)  # make 'newFile' readable by all,
                                                      # writeable by owner and group
      pickle.dump(bigBoard, os.fdopen(newFile, "w"))
      fileMapDict[pathToTargetDir] = newFileName
      rewriteFileMap(fileMapDict)

    # At this point, 'bigBoard' exists, and is updated.

    # floating div which will be populated with the stats
    # from one invocation when user hovers over a datestamp
    print "<div id=\"statsWindow\">"
    print "<div id=\"statsHeader\"></div>"
    print "<div id=\"statsBody\"></div>"
    print "</div>"

    # start main page
    print "<div id=\"readmeDiv\">"
    print "<!-- <a href=\"/website/codesupport/flash_howtos/home.py?submit=flashTest-HOWTO.txt\">DESCQA HOW-TO</a> -->"
    print "</div>"
    print "<div class=\"clearBlock\">&nbsp;</div>"
    print "<div id=\"titleDiv\">"
    print "<h1>DESCQA Invocations</h1>"
    print "<h3>Sorting is alphabetical; I'll fix that later today -turam</h3>"
    print "</div>"

    # make bar with navigation to other "pages" of results.
    invocationsPerPage = int(configDict.get("invocationsPerPage", 50))

    numRows = bigBoard.numRows

    if numRows > invocationsPerPage:
      lastPageNum = ((numRows-1) / invocationsPerPage) + 1
      try:
        thisPageNum = int(thisPageNum)
      except:
        # No page number in query-string, so 'thisPageNum' was None.
        # Either that or some joker entered a non-numerical value in URL bar.
        thisPageNum = lastPageNum
      else:
        if thisPageNum < 1:
          # some joker entered '0', probably
          thisPageNum = 1
        elif thisPageNum > lastPageNum:
          # some joker entered something too high
          thisPageNum = lastPageNum

      # This is tricky because the *smaller* the value of 'thisPageNum',
      # the further we reach back in time, and the *greater* the indices
      # of the invocations we need to examine. Therefore, to help with the
      # arithmetic, we create 'reversedPageNum', whose value gets higher
      # with the indices (but not the dates) of the invocations.
      reversedPageNum = (lastPageNum + 1) - thisPageNum

      print "<div class=\"clearBlock\">&nbsp;</div>"
      print "<div id=\"pagesDiv\">"
      if thisPageNum > 1:
        # print a "<<" (previous page link)
        endRow  = bigBoard.getInvocationName(reversedPageNum*invocationsPerPage)
        startRow = bigBoard.getInvocationName(((reversedPageNum+1)*invocationsPerPage)-1)
        print ("<a class=\"everblue\" " +
               "href=\"./home.cgi?target_dir=%s&page=%s\" " % (pathToTargetDir, thisPageNum-1) +
               "title=\"%s thru %s\">&lt;&lt;</a>" % (startRow, endRow))
      else:
        # print a "dummy link"
        print "<span style=\"color: gray\">&lt;&lt;</span>"

      for i in range(1, lastPageNum + 1):
        if i == thisPageNum:
          print "<span style=\"color: gray\">%s</span>" % i # not a link, since we're already on this page
        else:
          # see comment regarding 'reversedPageNum' above
          reversedI = (lastPageNum + 1) - i
          endRow  = bigBoard.getInvocationName((reversedI-1)*invocationsPerPage)
          startRow = bigBoard.getInvocationName((reversedI*invocationsPerPage)-1)
          print ("<a class=\"everblue\" " +
                 "href=\"./home.cgi?target_dir=%s&page=%s\" " % (pathToTargetDir, i) +
                 "title=\"%s thru %s\">%s</a>" % (startRow, endRow, i))

      if thisPageNum < lastPageNum:
        # print a ">>" (next page link)
        endRow  = bigBoard.getInvocationName((reversedPageNum-2)*invocationsPerPage)
        startRow = bigBoard.getInvocationName(((reversedPageNum-1)*invocationsPerPage)-1)
        print ("<a class=\"everblue\" " +
               "href=\"./home.cgi?target_dir=%s&page=%s\" " % (pathToTargetDir, thisPageNum+1) +
               "title=\"%s thru %s\">&gt;&gt;</a>" % (startRow, endRow))
      else:
        # print a "dummy link"
        print "<span style=\"color: gray\">&gt;&gt;</span>"

      print "</div>"

      startRow = (reversedPageNum - 1) * invocationsPerPage
      endRow   = min((startRow + invocationsPerPage - 1), numRows-1)
    else:
      startRow = 0
      endRow   = numRows-1

    # generate drop-down menu for easy switching between
    # FlashTest output directories if more than 1 available.
    if len(pathsToOutdirs) > 1:
      print "<div id=\"menuDiv\">"
      print "<select onchange=\"javascript: redirect(this)\">"
      print "<option>&nbsp;</option>"
      for pathToOutdir in pathsToOutdirs:
        if pathToOutdir != pathToTargetDir:
          print "<option value=\"%s\">%s</option>" % (pathToOutdir, pathToOutdir)
      print "</select>"
      print "</div>"

    print "<div class=\"clearBlock\">&nbsp;</div>"

    bigBoard.spewHtml(sys.stdout, startRow, endRow)

    print "</body>"
    print "</html>"
except Exception,e:
    pass
    import traceback
    traceback.print_exc(file=sys.stdout)
    print "Exception: ", e

