import os

## def cull(paths):
##   """
##   Take a list of paths and remove all which are duplicates
##   or substrings of other paths.  Since lists in python are
##   passed by reference, we don't need to return anything.
##   """
##   culledPaths = []
##   for path in paths[:]:
##     currentWorking
##   pathsCopy = paths[:]
##   pathsCopy.sort()
##   i = 0
##   while i < (len(pathsCopy)-1):
##     if pathsCopy[i+1].startswith(pathsCopy[i]):
##       paths.remove(pathsCopy[i+1])
##     i += 1

def __cull(nodes):
  """
  Take a list of nodes and return a new list which is a subset
  of the original list, and which contains only those nodes that
  are not sub-nodes of nodes already in the list
  """
  culledNodes = []
  for node in nodes:
    currentWorkingNode = node
    while currentWorkingNode.parent:
      if currentWorkingNode.parent in nodes:
        break
      else:
        currentWorkingNode = currentWorkingNode.parent
    else:
      culledNodes.append(node)

  return culledNodes

def getNodesFromPaths(masterNode, paths, cull=False):
  """
  Take a base node and a list of paths and return
  the list of node objects indicated by the paths.
  """
  selectedNodes = []
  for path in paths:
    childNode = masterNode.findChild(path)
    if childNode:
      selectedNodes.append(childNode)

  if cull:
    return __cull(selectedNodes)
  else:
    return selectedNodes

## def getLeafNodes(masterNode, paths):
##   """
##   Take a base node and a list of paths and return
##   the list of all leaf nodes below those paths.
##   """
##   leafNodes = []
##   def _getLeafNodes(node):
##     if len(node.subNodes) > 0:
##       for subNode in node.subNodes:
##         _getLeafNodes(subNode)
##     else:
##       leafNodes.append(node)

##   selectedNodes = getNodesFromPaths(masterNode, paths)

##   for selectedNode in selectedNodes:
##     _getLeafNodes(selectedNode)
##   return leafNodes

def getLeafNodes(nodes):
  """
  Take a list of nodes and return a list of all leaf-nodes
  (nodes with no sub-nodes) below those nodes. The returned
  list will include any members of the original list which
  themselves are leaf-nodes.
  """
  leafNodes = []
  for node in nodes:
    if len(node.subNodes) == 0:
      # one of the submitted nodes is already a leaf-node
      leafNodes.append(node)
    else:
      # this will loop through every descendent of node
      for subNode in node:
        if len(subNode.subNodes) == 0:
          leafNodes.append(subNode)
  return leafNodes

def writeXml(pathToXml, masterNode):
  # put the old version in a backup file
  # for a potential "undo" operation
  # DEV re-enable backups!
  #if os.path.isfile(pathToXml):
  #  os.rename(pathToXml, pathToXml + ".back")
  open(pathToXml, "w").write("\n".join(masterNode.getXml()))
