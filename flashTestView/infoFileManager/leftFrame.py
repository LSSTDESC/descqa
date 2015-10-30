#!/usr/bin/env python
#(Please keep all copyright notices.)
#This frameset document includes the Treeview script.
#Script found in: http://www.treeview.net
#Author: Marcelino Alves Martins

def _makeJavascript(rootNode, pathToStartNode, startChecked, startVisible):
  """
  generate the javascript 'infoNodes-xxxxxx.js' file
  from which Treeview will build the menu tree
  """
  lines = []
  def _scan(xmlNode):
    childsDepth = xmlNode.depth + 1 - depthOffset
    indent = "  " * childsDepth
    if len(xmlNode.subNodes) > 0:
      # Add all of this node's child-nodes
      for subNode in xmlNode.subNodes:
        thisNodesPath = subNode.getPathBelowRoot()
        lines.append(indent + "fld%d = insFld(fld%d, gFld(\"%s\", \"%s\", \"%s\"))" %
                     (childsDepth, childsDepth-1, subNode.name, subNode.name, thisNodesPath))
        lines.append(indent + "fld%d.prependHTML=\"<td valign='middle'><input type='checkbox'></td>\"" % childsDepth)
        if thisNodesPath in startChecked:
          lines.append(indent + "startChecked[startChecked.length] = fld%d" % childsDepth)
          startChecked.remove(thisNodesPath)
        if thisNodesPath in startVisible:
          lines.append(indent + "startVisible[startVisible.length] = fld%d" % childsDepth)
          startVisible.remove(thisNodesPath)

        _scan(subNode)

    return

  # Find node corresponding to path indicated by 'pathToStartNode', if given
  if pathToStartNode:
    startNode = rootNode.findChild(pathToStartNode)
  else:
    startNode = rootNode

  depthOffset = startNode.depth

  if startNode == rootNode:
    # If the externally-imposed "master node" which encloses all nodes coded
    # in the user's "test.info" file is at the root of the display, put its
    # name in italics, but if a "real" node is at the root, don't
    lines.append("fld0 = gFld(\"<i>%s</i>\", \"%s\", \"%s\")" %
                 (startNode.name, startNode.name, startNode.getPathBelowRoot()))
  else:
    lines.append("fld0 = gFld(\"%s\", \"%s\", \"%s\")" %
                 (startNode.name, startNode.name, startNode.getPathBelowRoot()))

  lines.append("fld0.prependHTML=\"<td valign='middle'><input type='checkbox'></td>\"")
  if startNode.getPathBelowRoot() in startChecked:
    lines.append("startChecked[startChecked.length] = fld0")
    startChecked.remove(startNode.getPathBelowRoot())

  _scan(startNode)  # this will populate the list 'lines'

  # Any paths left over in 'startChecked' or 'startVisible' were not present
  # in the info files tree. We'll make note of them and later alert the user
  for notFound in startChecked:
    lines.append("notFound[notFound.length] = \"%s\"" % notFound)
  for notFound in startVisible:
    lines.append("notFound[notFound.length] = \"%s\"" % notFound)

  return lines


import sys, os
import cgi, tempfile
from time import asctime
cwdParent = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(cwdParent, "lib"))
import littleParser, xmlNode

pathToConfig = os.path.join(cwdParent, "config")
if os.path.isfile(pathToConfig):
  configDict = littleParser.parseFile(pathToConfig)

# -------------- form data ---------------- #
# "start_checked" or "start_visible" keys will appear in the query string
# with no value if they are referring to the root node. We don't want to
# lose this information, so set python's 'keep_blank_values' keyword
form = cgi.FieldStorage(keep_blank_values=True)

pathToInfo   = form.getvalue("path_to_info", configDict.get("pathToInfo"))
               # "home.py" has guaranteed that if "path_to_info" was not
               # passed in the query-string, "config" exists and defines
               # at least one valid value for "pathToInfo"

if isinstance(pathToInfo, list):  # if "config" lists multiple values
  pathToInfo   = pathToInfo[0]

viewFiles       = form.getvalue("view_files")
editFiles       = form.getvalue("edit_files")
openPanel       = form.getvalue("open_panel")
undoChanges     = form.getvalue("undo_changes")
pathToStartNode = form.getvalue("start_node")

startChecked = form.getlist("start_checked")
startVisible = form.getlist("start_visible")

# DEV - get better, more general names for these next two coming from
# the Comparison test declaration via 'viewer/viewBuilds.js'.
# Not all panels have fields named "key" and "value", but right now
# I'm only worrying about the Comparison test opening the "addKeys" panel
key   = form.getvalue("key")
value = form.getvalue("value")

if undoChanges:
  if os.path.isfile(pathToInfo + ".back"):
    os.rename(pathToInfo + ".back", pathToInfo)

try:
  masterNode = xmlNode.parseXml(pathToInfo)
except Exception, e:
  print "Content-type: text/html\n\n"
  print "<html>"
  print "<head>"
  print "<title>left</title>"
  print "</head>"
  print "<body>"
  print "Error: %s" % str(e).replace("\n","<br>")
  print "</body>"
  print "</html>"
  sys.exit(1)
  
# else  

# see if the user wants to "zoom in" on a node
# below the master node, and if so, if it exists
startNodeDoesntExist = False
if pathToStartNode:
  if not masterNode.findChild(pathToStartNode):
    startNodeDoesntExist = pathToStartNode  # a little kludge to preserve
                                            # value of 'pathToStartNode'
                                            # for the error message
    pathToStartNode = None

# this is the text that will go into our "infoNodes-xxxxxx.js" file
text = ("/** This file is computer-generated by 'leftFrame.py'! Do not edit! **/\n" +
        "/** %s **/\n\n" % asctime() +
        open("leftFrame/infoNodesHeader.js").read() +
        "\n".join(_makeJavascript(masterNode, pathToStartNode, startChecked, startVisible)) +
        open("leftFrame/infoNodesFooter.js").read())

# make a new "infoNodes-xxxxxx.js" file and fill it with our new
# instructions for building the tree menu. We have to make a new
# file with a *new name* every time to prevent Internet Explorer
# from using a cached version of the javascript.  The javascript
# must be read anew each time for changes in the tree menu to be
# visible.
fd, brandNewFileName = tempfile.mkstemp(prefix="infoNodes-", suffix=".js", dir="leftFrame")
brandNewFileName = os.path.basename(brandNewFileName)

# remove all old "infoNodes-xxxxxx.js" files to keep things tidy
[os.remove(os.path.join("leftFrame", f)) for f in os.listdir("leftFrame")
  if f.startswith("infoNodes-") and f != brandNewFileName]

os.write(fd, text)
os.close(fd)
# make brand new file readable by all
os.chmod(os.path.join("leftFrame", brandNewFileName), 256 + 32 + 4 + 128 + 16)

print "Content-type: text/html\n\n"
print "<html>"
print "<head>"
print "<title>left</title>"

# next three lines ensure browsers don't cache, as caching makes
# changes to "test.info" files appear to have not gone through.
print "<meta http-equiv=\"cache-control\" content=\"no-cache\">"
print "<meta http-equiv=\"Pragma\" content=\"no-cache\">"
print "<meta http-equiv=\"Expires\" content=\"-1\">"

print "<link rel=\"stylesheet\" href=\"leftFrame/style.css\" type=\"text/css\"></link>"

# insert javascript into <head>
print "<script src=\"leftFrame/ua.js\"></script>"                  # browser detection
print "<script src=\"leftFrame/ftiens4.js\"></script>"             # infrastructure for tree
print "<script src=\"leftFrame/%s\"></script>" % brandNewFileName  # data for filesystem tree
print "<script src=\"leftFrame/auxfns.js\"></script>"              # js funcs for left frame
print "</head>"

# possibly perform some actions when finished loading
onLoadCommands = []
if startNodeDoesntExist:
  onLoadCommands.append("alert('Node &quot;%s&quot; not found in &quot;%s&quot;')" % (startNodeDoesntExist, pathToInfo))
if viewFiles:
  onLoadCommands.append("viewFiles()")
if editFiles:
  onLoadCommands.append("editFiles()")
if openPanel:
  openPanelCmds = []
  openPanelCmds.append("'%s'" % openPanel)
  if key:
    openPanelCmds.append("'%s'" % key)
  if value:
    openPanelCmds.append("'%s'" % value)
  openPanelStr = ",".join(openPanelCmds)
  onLoadCommands.append("openPanel(%s)" % openPanelStr)
onLoadStr = ";".join(onLoadCommands)
if len(onLoadStr) > 0:
  print "<body topmargin=16 marginheight=16 onLoad=\"javascript: %s\">" % onLoadStr
else:
  print "<body topmargin=16 marginheight=16>"

# re-insert some values into page so they can persist when "leftFrame.py"
# is reloaded by a call to "refreshTree()" in "rightFrame/auxfns.js"
print "<input type=\"hidden\" id=\"path_to_info\" value=\"%s\">" % pathToInfo
if pathToStartNode:
  print "<input type=\"hidden\" id=\"start_node\" value=\"%s\">" % pathToStartNode

# link to treeview site
print open("leftFrame/treeViewLink.html","r").read()

# Build the browser's objects and display default view of the tree.
# 'setInitialLayout()' (see infoNodesFooter.txt) is executed here.
print "<div id=\"treeBlock\">"
print "<script>initializeDocument()</script>"
print "</div>"

print "<noscript>"
print "You must enable JavaScript for this page to function."
print "</noscript>"

# this controls which control panel starts visible
if (openPanel == "addKeys" or openPanel == "appendValues" or
    openPanel == "removeKeys"):
  showHide = ("style=\"display:none\"","") # start with files control panel visible
else:
  showHide = ("","style=\"display:none\"") # start with nodes control panel visible
print open("leftFrame/controlPanel.html","r").read() % showHide

print "</body>"
print "</html>"
