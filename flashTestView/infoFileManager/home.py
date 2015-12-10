#!/usr/bin/env python
import sys, os, cgi
cwdParent = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(cwdParent, "lib"))
import littleParser

def abort(msg):
  print "<body>"
  print msg
  print "</body>"
  print "</html>"
  sys.exit(0)

print "Content-type: text/html\n\n"
print "<html>"
print "<head>"
print "<title>Flash Center for Computational Science</title>"
print "</head>"

form = cgi.FieldStorage()
pathToInfo = form.getvalue("path_to_info")

if pathToInfo:
  if not os.path.isfile(pathToInfo):
    abort("Error: info-file \"%s\" does not exist or is not a file." % pathToInfo)
  else:
    pass  # "leftFrame.py" will use this info-file to build its tree
else:
  # no 'path_to_info' was passed in via cgi, so look for a "config" file.
  pathToConfig = os.path.join(cwdParent, "config")
  if not os.path.isfile(pathToConfig):
    msg = ("Error: You must put a configuration file at \"%s\"<br>" % pathToConfig +
           "with at least one path specified for the key \"pathToInfo\" in the manner:<br>" +
           "pathToInfo: <path1>, <path2>, ... etc.")
    abort(msg)
  else:
    configDict = littleParser.parseFile(pathToConfig)
    pathToInfo = configDict.get("pathToInfo", "")
    if not pathToInfo:
      msg = ("Error: The configuration file at \"%s\" must specify<br>" % pathToConfig +
             "at least one path for the key \"pathToInfo\" in the manner:<br><br>" +
             "<i>pathToInfo: path1, path2, ...</i> etc.")
      abort(msg)
    else:
      if isinstance(pathToInfo, list):
        # check validity of all paths in config
        for altPathToInfo in pathToInfo:
          if not os.path.isfile(altPathToInfo):
            abort("Error: \"%s\" as specified in \"config\" does not exist or is not a file." % altPathToInfo)
        pathToInfo = pathToInfo[0]
      elif not os.path.isfile(pathToInfo):
        abort("Error: \"%s\" as specified in \"config\" does not exist or is not a file." % pathToInfo)

# At this point, we know that the file manager has an info-file to act
# upon. It is either specified in the query string, which we will pass
# to the left and right frames, or it is available in the "config" file.
queryStr = os.environ.get("QUERY_STRING", "")
if queryStr:
  queryStr = "?" + queryStr
print "<frameset cols=\"236,*\">"
print "  <frame src=\"leftFrame.py%s\" name=\"treeframe\">" % queryStr
print "  <frame src=\"rightFrame.py%s\" name=\"basefrm\">" % queryStr
print "</frameset>"
print "</html>"
