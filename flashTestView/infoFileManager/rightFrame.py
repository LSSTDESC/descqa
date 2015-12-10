#!/usr/bin/env python
import sys, os, cgi, re, urllib
sys.path.insert(0, "rightFrame")
import auxfns
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

pathsToInfos  = configDict.get("pathToInfo")
pathToInfo    = form.getvalue("path_to_info")
newPathToInfo = form.getvalue("new_path_to_info")

if not pathToInfo:
  # "home.py" has guaranteed that if "path_to_info" was not passed in the
  # query-string, "config" exists and contains at least one valid value
  # for "pathToInfo".
  if isinstance(pathsToInfos, list):  # if "config" lists multiple values
    pathToInfo = pathsToInfos[0]
  else:  # "config" only lists one value, so 'pathsToInfos' is a string
    pathToInfo = pathsToInfos

try:
  masterNode = xmlNode.parseXml(pathToInfo)
  if newPathToInfo:
    newMasterNode = xmlNode.parseXml(newPathToInfo)
  else:
    newMasterNode = masterNode
except Exception, e:
  print "Content-type: text/html\n\n"
  print "<html>"
  print "<head>"
  print "<title>right</title>"
  print "</head>"
  print "<body>"
  print "Error: %s" % str(e).replace("\n","<br>")
  print "</body>"
  print "</html>"
  sys.exit(1)

# get command and array of relative paths from the
# xml-tree-root to all nodes checked by the user
cmd   = form.getvalue("cmd")
paths = form.getlist("path")

print "Content-type: text/html\n\n"
print "<html>"
print "<head>"
print "<title>right</title>"

# next three lines ensure browsers don't cache, as caching makes
# changes to "test.info" files appear to have not gone through.
print "<meta http-equiv=\"cache-control\" content=\"no-cache\">"
print "<meta http-equiv=\"Pragma\" content=\"no-cache\">"
print "<meta http-equiv=\"Expires\" content=\"-1\">"

print "<script src=\"rightFrame/auxfns.js\"></script>"
print "</head>"

if cmd == "view_files":
  ######################################
  ##  view all "test.info" data in    ##
  ##  a contiguous non-editable list  ##
  ######################################

  # get the leaf-nodes that correspond to the user's chosen paths
  leafNodes = auxfns.getLeafNodes(auxfns.getNodesFromPaths(masterNode, paths, cull=True))

  print "<body>"

  # re-insert 'paths' elements in case we reload this
  # page from itself via an 'edit' button below
  for path in paths:
    print "<input type=\"hidden\" name=\"path\" value=\"%s\">" % path

  for leafNode in leafNodes:
    print "<b>" + leafNode.getPathBelowRoot() + "</b><br>"
    for line in leafNode.text:
      print line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
      print "<br>"
    print "<input type=\"button\" value=\"edit\" onclick=\"javascript: loadFile('%s')\">" % leafNode.getPathBelowRoot()
    print "<br><br>"


elif cmd == "edit_files":
  #################################################
  ##  hand-edit a single run's "test.info" data  ##
  #################################################

  print "<body>"

  # re-insert 'paths' elements in case we reload this
  # page from itself via any of the buttons below
  for path in paths:
    print "<input type=\"hidden\" name=\"path\" value=\"%s\">" % path

  # get the leaf-nodes that correspond to the user's chosen paths
  leafNodes = auxfns.getLeafNodes(auxfns.getNodesFromPaths(masterNode, paths, cull=True))

  editPath = form.getvalue("edit_path")
  if editPath:
    editNode = masterNode.findChild(editPath)
  else:
    editNode = leafNodes[0]

  editNodeIndex = leafNodes.index(editNode)

  print "<b>%s</b><br>" % editNode.getPathBelowRoot()
  print "<table><tr><td colspan=\"2\">"

  # create the user-interactive editing-box
  print "<textarea id=\"text\" rows=\"20\" cols=\"100\" wrap=\"off\" onKeyUp=\"javascript: enableCommit()\">"
  for line in editNode.text:
    print line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
  print "</textarea>"
  print "</td></tr>"
  print "<tr>"

  # "prev" and "next" buttons
  print "<td align=\"left\">"  
  if editNodeIndex == 0:
    print "<input type=\"button\" value=\"prev\" disabled>"
  else:
    print "<input type=\"button\" value=\"prev\" onclick=\"javascript: if (changeOK()) loadFile('%s')\">" % (
      leafNodes[editNodeIndex-1].getPathBelowRoot())
  if editNodeIndex == len(leafNodes)-1:
    print "<input type=\"button\" value=\"next\" disabled>"
  else:
    print "<input type=\"button\" value=\"next\" onclick=\"javascript: if (changeOK()) loadFile('%s')\">" % (
      leafNodes[editNodeIndex+1].getPathBelowRoot())
  print "</td>"

  # "reload" and "commit" buttons
  print "<td align=\"right\">"
  print "<input type=\"button\" value=\"reload\" onclick=\"javascript: loadFile('%s')\">" % (
    editNode.getPathBelowRoot())
  print "<input type=\"button\" id=\"commit\" value=\"commit\" disabled onclick=\"javascript: writeFile('%s')\">" % (
    editNode.getPathBelowRoot())
  print "</td>"

  print "</tr></table>"


elif cmd == "write_file":
  ###################################
  ##  commit a hand-edited change  ##
  ###################################

  newText        = urllib.unquote(form.getvalue("text")).rstrip()
  writeNode      = masterNode.findChild(form.getvalue("write_path"))
  writeNode.text = [line.strip() for line in newText.split("\n") if len(line.strip()) > 0]

  try:
    auxfns.writeXml(pathToInfo, masterNode)
  except Exception, e:
    print "<body>"
    print e
  else:
    print "<body onLoad=\"javascript: loadFile('%s');\">" % writeNode.getPathBelowRoot()
    # re-insert 'paths' elements since we're going to reload
    # this page with an 'edit' command as soon as we're done.
    for path in paths:
      print "<input type=\"hidden\" name=\"path\" value=\"%s\">" % path


#######################
##  NODE MANAGEMENT  ##
#######################

elif cmd == "find_nodes":
  ################################
  ##  find nodes of given name  ##
  ################################

  # get a list of node objects corresponding to 'paths'
  selectedNodes = auxfns.getNodesFromPaths(masterNode, paths, cull=True)

  # name of node(s) we're looking for
  soughtNode = form.getvalue("new_node")

  foundNodes = []
  for selectedNode in selectedNodes:
    foundNodes.extend(selectedNode.findChildren(soughtNode))

  print "<body>"
  if len(foundNodes) > 0:
    print "<b>found:</b><br>"
    for foundNode in foundNodes:
      print "%s<br>" % foundNode.getPathBelowRoot()

  else:
    print "<b>No nodes found</b>"


elif cmd == "add_node":
  ##################
  ##  add a node  ##
  ##################
  if not os.access(pathToInfo, os.W_OK):
    print "<body>"
    print "Error: \"%s\" is not writable" % pathToInfo
  else:
    # get a list of node objects corresponding to 'paths'
    selectedNodes = auxfns.getNodesFromPaths(masterNode, paths)

    # name of new node(s) we're going to add
    newNodeName = form.getvalue("new_node")

    pathsToAddedNodes   = []
    pathsToUnaddedNodes = []  # i.e., paths that would have existed
                              # if the new node had been added
    for selectedNode in selectedNodes:
      try:
        newNode = selectedNode.add(newNodeName)
      except Exception, e:
        pathsToUnaddedNodes.append((os.path.join(selectedNode.getPathBelowRoot(), newNodeName), str(e)))
      else:
        pathsToAddedNodes.append(newNode.getPathBelowRoot())

    if pathsToAddedNodes:
      auxfns.writeXml(pathToInfo, masterNode)
      queryStr = ""
      queryStr +=  "start_checked=" + "&start_checked=".join([selectedNode.getPathBelowRoot() for selectedNode in selectedNodes])
      queryStr += "&start_visible=" + "&start_visible=".join([pathToAddedNode for pathToAddedNode in pathsToAddedNodes])
      print "<body onLoad=\"javascript: refreshTree('" + queryStr + "')\">"
      print "<b>Successfully added:</b><br>"
      for pathToAddedNode in pathsToAddedNodes:
        print "%s<br>" % pathToAddedNode
    if pathsToUnaddedNodes:
      if not pathsToAddedNodes:
        print "<body>"
      print "<b>Failed to add:</b><br>"
      for pathToUnaddedNode, e in pathsToUnaddedNodes:
        print "%s:<br>&nbsp;&nbsp;&nbsp;&nbsp;%s<br>" % (pathToUnaddedNode, e)
      

elif cmd == "remove_node":
  #####################
  ##  remove a node  ##
  #####################
  if not os.access(pathToInfo, os.W_OK):
    print "<body>"
    print "Error: \"%s\" is not writable" % pathToInfo
  else:
    if form.getvalue("new_node"):  # says whether to remove this
      fullRemove = True            # node's children as well
    else:
      fullRemove = False

    # get a list of node objects corresponding to 'paths'
    selectedNodes = auxfns.getNodesFromPaths(masterNode, paths)

    pathsToRemovedNodes   = []
    pathsToUnremovedNodes = []
    for selectedNode in selectedNodes:
      try:
        selectedNode.remove(fullRemove)
      except Exception, e:
        pathsToUnremovedNodes.append((selectedNode.getPathBelowRoot(), str(e)))
      else:
        pathsToRemovedNodes.append(selectedNode.getPathBelowRoot())

    auxfns.writeXml(pathToInfo, masterNode)
    if pathsToRemovedNodes:
      print "<body onLoad=\"javascript: refreshTree('')\">"
      print "<b>Successfully removed:</b><br>"
      for pathToRemovedNode in pathsToRemovedNodes:
        print "\"%s\"<br>" % pathToRemovedNode
    if pathsToUnremovedNodes:
      if not pathsToRemovedNodes:
        print "<body>"
      print "<b>Failed to remove:</b><br>"
      for pathToUnremovedNode, e in pathsToUnremovedNodes:
        print "\"%s\":<br>&nbsp;&nbsp;&nbsp;&nbsp;%s<br>" % (pathToUnremovedNode, e)


elif cmd == "rename_node":
  #####################
  ##  rename a node  ##
  #####################
  if not os.access(pathToInfo, os.W_OK):
    print "<body>"
    print "Error: \"%s\" is not writable" % pathToInfo
  else:
    # get a list of node objects corresponding to 'paths'
    selectedNodes = auxfns.getNodesFromPaths(masterNode, paths)

    # name of new node(s) we're going to add
    newName = form.getvalue("new_node")

    pathsToRenamedNodes   = []
    pathsToUnrenamedNodes = []
    for selectedNode in selectedNodes:
      if selectedNode.name == newName:
        msg = "Old name is same as new name."
        pathsToUnrenamedNodes.append((selectedNode.getPathBelowRoot(), msg))
      else:
        origPath = selectedNode.getPathBelowRoot()
        try:
          selectedNode.rename(newName)
        except Exception, e:
          pathsToUnrenamedNodes.append((selectedNode.getPathBelowRoot(), str(e)))
        else:
          pathsToRenamedNodes.append((origPath, selectedNode.getPathBelowRoot()))

    if pathsToRenamedNodes:
      auxfns.writeXml(pathToInfo, masterNode)
      queryStr = ""
      queryStr += "start_checked=" + "&start_checked=".join([selectedNode.getPathBelowRoot() for selectedNode in selectedNodes])
      print "<body onLoad=\"javascript: refreshTree('" + queryStr + "')\">"
      print "<b>Successfully renamed:</b><br>"
      for origPath, pathToRenamedNode in pathsToRenamedNodes:
        print "\"%s\"<br>to: \"%s\"<br>" % (origPath, pathToRenamedNode)
    if pathsToUnrenamedNodes:
      if not pathsToRenamedNodes:
        print "<body>"
      print "<b>Failed to rename:</b><br>"
      for pathToUnrenamedNode, e in pathsToUnrenamedNodes:
        print "\"%s\":<br>&nbsp;&nbsp;&nbsp;&nbsp;%s<br>" % (pathToUnrenamedNode, e)


elif cmd == "clone_branch":
  ################################################################
  ##  create new node as sibling to selected node, and give it  ##
  ##  sub-nodes which are copies of selected node's sub-nodes.  ##
  ################################################################
  if not os.access(pathToInfo, os.W_OK):
    print "<body>"
    print "Error: \"%s\" is not writable" % pathToInfo
  else:
    # get a list of node objects corresponding to 'paths'
    selectedNodes = auxfns.getNodesFromPaths(masterNode, paths)

    # name of node(s) we're going to add
    newNodeName = form.getvalue("new_node")

    pathsToClonedNodes   = []
    pathsToUnclonedNodes = []
    for selectedNode in selectedNodes:
      try:
        newNode = selectedNode.clone(newNodeName)
      except Exception, e:
        pathsToUnclonedNodes.append((selectedNode.getPathBelowRoot(), str(e)))
      else:
        pathsToClonedNodes.append((selectedNode.getPathBelowRoot(), newNode.getPathBelowRoot()))

    if pathsToClonedNodes:
      auxfns.writeXml(pathToInfo, masterNode)
      queryStr = ""
      queryStr +=  "start_checked=" + "&start_checked=".join([selectedNode.getPathBelowRoot() for selectedNode in selectedNodes])
      queryStr += "&start_visible=" + "&start_visible=".join([pathToClonedNode[1] for pathToClonedNode in pathsToClonedNodes])
      print "<body onLoad=\"javascript: refreshTree('" + queryStr + "')\">"
      print "<b>Successfully cloned:</b><br>"
      for selectedPath, pathToClonedNode in pathsToClonedNodes:
        print "\"%s\"<br>to: \"%s\"<br>" % (selectedPath, pathToClonedNode)
    if pathsToUnclonedNodes:
      if not pathsToClonedNodes:
        print "<body>"
      print "<b>Failed to clone:</b><br>"
      for pathToUnclonedNode, e in pathsToUnclonedNodes:
        print "\"%s\"<br>&nbsp;&nbsp;&nbsp;&nbsp;%s<br>" % (pathToUnclonedNode, e)


elif cmd == "extend_branch":
  ##################################################
  ##  new node 'B' becomes child of node 'A'      ##
  ##  old children of 'A' become children of 'B'  ##
  ##################################################
  if not os.access(pathToInfo, os.W_OK):
    print "<body>"
    print "Error: \"%s\" is not writable" % pathToInfo
  else:
    # get a list of node objects corresponding to 'paths'
    selectedNodes = auxfns.getNodesFromPaths(masterNode, paths)

    # name of node(s) we're going to add
    newNodeName = form.getvalue("new_node")

    pathsToAddedNodes   = []
    pathsToUnaddedNodes = []  # i.e., paths that would have existed
                              # if the new node had been added
    for selectedNode in selectedNodes:
      try:
        newNode = selectedNode.extend(newNodeName)
      except Exception, e:
        pathsToUnaddedNodes.append((os.path.join(selectedNode.getPathBelowRoot(), newNodeName), str(e)))
      else:
        pathsToAddedNodes.append(newNode.getPathBelowRoot())

    if pathsToAddedNodes:
      auxfns.writeXml(pathToInfo, masterNode)
      queryStr = ""
      queryStr +=  "start_checked=" + "&start_checked=".join([selectedNode.getPathBelowRoot() for selectedNode in selectedNodes])
      queryStr += "&start_visible=" + "&start_visible=".join([pathToAddedNode for pathToAddedNode in pathsToAddedNodes])
      print "<body onLoad=\"javascript: refreshTree('" + queryStr + "')\">"
      print "<b>Successfully added:</b><br>"
      for pathToAddedNode in pathsToAddedNodes:
        print "%s<br>" % pathToAddedNode

    if pathsToUnaddedNodes:
      if not pathsToAddedNodes:
        print "<body>"
      print "<b>Failed to add:</b><br>"
      for pathToUnaddedNode, e in pathsToAddedNodes:
        print "%s:<br>&nbsp;&nbsp;&nbsp;&nbsp;%s<br>" % (pathToUnaddedNode, e)


#######################
##  TEXT MANAGEMENT  ##
#######################

elif cmd == "regex_find":

  # get the leaf-nodes that correspond to the user's chosen paths
  leafNodes = auxfns.getLeafNodes(auxfns.getNodesFromPaths(masterNode, paths, cull=True))

  # regex we're looking for
  key = form.getvalue("key")

  foundNodes = []
  for leafNode in leafNodes:
    lines = leafNode.text
    for line in lines:
      if re.search(key, line):
        foundNodes.append(leafNode)
        break

  if len(foundNodes) > 0:
    queryStr = ""
    queryStr +=  "start_checked=" + "&start_checked=".join([foundNode.getPathBelowRoot() for foundNode in foundNodes])
    print "<body onLoad=\"javascript: refreshTree('" + queryStr + "')\">"
    for foundNode in foundNodes:
      print "found %s<br>" % foundNode.getPathBelowRoot()

  else:
    print "<body>"
    print "No nodes found"


elif (cmd == "add_key" or cmd == "append_val" or
      cmd == "remove_key" or cmd == "regex_replace"):
  ##################################
  ##  manipulate keys and values  ##
  ##  in "test.info" data         ##
  ##################################

  key = form.getvalue("key")
  val = form.getvalue("val")

  # get the leaf-nodes that correspond to the user's chosen paths
  leafNodes = auxfns.getLeafNodes(auxfns.getNodesFromPaths(masterNode, paths, cull=True))

  updatedNodes = []

  if cmd == "add_key" or cmd == "append_val" or cmd == "remove_key":
    for leafNode in leafNodes:
      lines = leafNode.text

      i = 0
      while i < len(lines):
        if lines[i].startswith(key + ":"):
          if cmd == "add_key":
            lines[i] = "%s: %s" % (key, val)
          elif cmd == "append_val":
            lines[i] = lines[i] + " " + val
          elif cmd == "remove_key":
            del lines[i]
            i -= 1
          updatedNodes.append(leafNode)

        i += 1

      if cmd == "add_key" and leafNode not in updatedNodes:
        lines.append("%s: %s" % (key, val))
        updatedNodes.append(leafNode)

  elif cmd == "regex_replace":
    for leafNode in leafNodes:
      lines = leafNode.text
      i=0
      while i < len(lines):
        if re.search(key, lines[i]):
          lines[i] = re.sub(key, val, lines[i])
          updatedNodes.append(leafNode)
        i += 1

  ####################
  ##  print output  ##
  ####################

  try:
    auxfns.writeXml(pathToInfo, masterNode)
  except Exception, e:
    print "<body>"
    print e
  else:
    print "<body>"

    if len(updatedNodes) > 0:
      print "<b>Successfully updated:</b><br>"
      for updatedNode in updatedNodes:
        print "%s<br>" % updatedNode.getPathBelowRoot()
    else:
      print "<b>No updates made</b>"

else:
  print "<body>"
  if newPathToInfo:
    print "<input id=\"new_path_to_info\" type=\"hidden\" value=\"%s\">" % newPathToInfo

  print "Working on info-file at \"%s\"" % pathToInfo

  if isinstance(pathsToInfos, list):
    print "<br><br>"
    # make drop down menu for choosing alternate info files
    print "<select onchange=\"javascript: pickNewPathToInfo(this)\">"
    print "<option>-- choose an alternate info-file --</option>"
    for altPathToInfo in pathsToInfos:
      if altPathToInfo != pathToInfo:
        if altPathToInfo == newPathToInfo:
          print "<option value=\"%s\" selected>%s</option>" % (altPathToInfo, altPathToInfo)
        else:
          print "<option value=\"%s\">%s</option>" % (altPathToInfo, altPathToInfo)

    print "</select>"

    print "<br><br>"
    # make drop down menu for choosing a start-node for this info-file
    print "<select onchange=\"javascript: redirect(this, '%s')\">" % pathToInfo
    print "<option>-- choose a root node --</option>"
    print "<option>/</option>"  # choose the all-encompassing master-node
    for subNode in newMasterNode:
      print "<option value=\"%s\">%s</option>" % (subNode.getPathBelowRoot(), subNode.getPathBelowRoot())     
    print "</select>"

print "</body>"
print "</html>"
