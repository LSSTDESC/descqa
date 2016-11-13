import re, os

class NodeError(Exception):
  pass

class ParseError(Exception):
  pass

class XmlNode:
  """
  Represents a single element in a multi-element categorization
  of a "test.info" description. For example, a single "path" to
  a "test.info" node might look like:

    Sod/pm3/hdf5/parallel/1d

  Each element in this path would be represented by a single
  node object. The node whose name was "parallel" would have
  a sub-node with the name "1d". The "1d" node would in turn
  have no sub-nodes, but its "text" member would contain the
  information necessary to run a FLASH simulation of type "Sod",
  using paramesh3, etc.

  Nodes are typically constructed by parsing an xml file with
  this modules "parseXml()" method. For example, a text file
  that contained the following:

    <Sod>
      <pm3>
        <hdf5>
          <parallel>
            <1d>
              setupName: Sod
              numProcs: 1
              parfiles: flash.par
            </1d>
          </parallel>
        </hdf5>
      </pm3>
    </Sod>

  would produce the nodes described above when parsed.

  Calling the "Sod" node's "getXml()" method would reproduce the
  above xml source again. Thus a node's sub-nodes and/or text can
  be easily manipulated in memory and re-converted into xml.

  I considered using pickled objects and avoiding xml altogether,
  but decided that the usefulness of having the ultimate source
  of the data encoded in a human-readable form was more important
  than the degree of inefficiency introduced by parsing the xml
  file into node objects and back again.
  """

  ########################
  ##  python functions  ##
  ########################
  
  def __init__(self, name):
    self.name     = name
    self.parent   = None
    self.depth    = 0
    self.subNodes = []
    self.text     = []

  def __iter__(self):
    # NB: this method returns a generator
    # object that can be iterated over
    return self.__nodeGenerator()

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name


  #########################
  ##  private functions  ##
  #########################

  def __adjustDepth(self):
    """
    Make sure the depth values of this
    node and all its children are correct
    """
    self.depth = self.parent.depth + 1
    for subNode in self.subNodes:
      subNode.__adjustDepth()

  def __adopt(self, child):
    """
    Give this node a new child-node, adjusting
    the child's parent and depth properties
    """
    # make sure the incoming child's name isn't
    # the same as an already present child's
    for subNode in self.subNodes:
      if subNode.name == child.name:
        msg = ("new node's name \"%s\" already used by sub-node of " % child.name +
               "parent \"%s\"" % self.getPathBelowRoot())
        raise NodeError(msg)

    # This node passes any text to its new child, assuming
    # the child has no text of its own. This can happen if
    # user tries an "add_node" command on a leaf node.
    if len(child.text) == 0:
      child.text = self.text[:]
    self.text = []
    self.subNodes.append(child)
    if child.parent and child in child.parent.subNodes:
      child.parent.subNodes.remove(child)
    child.parent = self
    child.__adjustDepth()

  def __copy(self):
    """
    Return a copy of this node with copies of
    all child-nodes and text
    """
    newNode = XmlNode(self.name)
    newNode.text = self.text[:]
    for subNode in self.subNodes:
      newNode.__adopt(subNode.__copy())
    return newNode

  def __nodeGenerator(self):
    """
    Returns a python generator object so that sub-nodes
    can be listed recursively (depth first) when a node
    is iterated over
    """
    for subNode in self.subNodes:
      yield subNode
      for subSubNode in subNode.__nodeGenerator():
        yield subSubNode


  ########################
  ##  public functions  ##
  ########################

  def add(self, newNodeName):
    """
    Create new node as child of this node with name 'newNodeName'
    """
    # check for a conflicting name
    childrenNames = [subNode.name for subNode in self.subNodes]
    if newNodeName in childrenNames:
      if self.isMasterNode():
        path = "masterNode"
      else:
        path = self.getPathBelowRoot()
      msg = ("Unable to create node with name \"%s\" " % newNodeName +
             "as sub-node of \"%s\", " % path +
             "as this node already has a sub-node of that name.")
      raise NodeError(msg)
    # else
    newNode = XmlNode(newNodeName)
    self.__adopt(newNode)
    return newNode

  def clone(self, newNodeName):
    """
    Create new sibling-node based on recursive copy
    of all sub-nodes and text.
    """
    # check for a conflicting name
    childrenNames = [subNode.name for subNode in self.parent.subNodes]
    if newNodeName in childrenNames:
      if self.parent.isMasterNode():
        parentPath = "master-node"
      else:
        parentPath = self.parent.getPathBelowRoot()
      msg = ("Unable to create node with name \"%s\" " % newNodeName +
             "as sub-node of \"%s\", " % parentPath +
             "as this node already has a sub-node of that name.")
      raise NodeError(msg)
    # else
    newNode = self.__copy()
    newNode.name = newNodeName
    self.parent.__adopt(newNode)
    return newNode

  def extend(self, newNodeName):
    """
    Create new node as child of this node with name 'newNodeName',
    but pass any of this node's current children to the new node.
    """
    newNode = XmlNode(newNodeName)
    # We don't need check for name conflicts because by
    # definition 'newNode' will be this node's only child
    for subNode in self.subNodes[:]:
      newNode.__adopt(subNode)  # this will automatically remove 'subNode'
                                # from 'self's' list of sub-nodes
    self.__adopt(newNode)
    return newNode

  def isMasterNode(self):
    """
    Returns True if this node is the master-node
    (has no parent), otherwise False
    """
    if self.parent:
      return False
    return True

  def remove(self, fullRemove):
    """
    Remove this node from its parent's list of sub-nodes.
    If 'fullRemove' is false, this node's sub-nodes will
    be adopted by its parent unless any of those sub-nodes
    has the same name as one of the parent's already-extant
    children. In that circumstance, a "NodeError" is raised.
    """
    if fullRemove:
      self.parent.subNodes.remove(self)
    else:
      # check for a conflicting name
      siblingNames = [subNode.name for subNode in self.parent.subNodes if subNode != self]
      for subNode in self.subNodes:
        if subNode.name in siblingNames:
          if self.parent.isMasterNode():
            parentPath = "master-node"
          else:
            parentPath = self.parent.getPathBelowRoot()
          msg = ("Unable to pass sub-node \"%s\" " % subNode.name +
                 "to parent-node \"%s\", " % parentPath +
                 "as parent already has a child of that name.")
          raise NodeError(msg)
      # else
      self.parent.subNodes.remove(self)
      for subNode in self.subNodes:
        self.parent.__adopt(subNode)

      # If parent had no other children besides this one,
      # it can absorb this one's text.
      if len(self.parent.subNodes) == 0:
        self.parent.text = self.text[:]

  def rename(self, newName):
    """
    Rename this node unless its parent already
    has a sub-node with name 'newName'
    """
    siblingNames = [subNode.name for subNode in self.parent.subNodes if subNode != self]
    if newName in siblingNames:
      msg = ("Unable to rename \"%s\" " % self.getPathBelowRoot() +
             "to \"%s\", as parent already has a child of that name." % newName)
      raise NodeError(msg)
    # else
    self.name = newName

  def findChild(self, path):
    """
    Return a node corresponding to 'path', where
    each element of 'path' is the name of a node.
    """
    path = os.path.normpath(path)
    if path == ".":
      return self
    # else
    currentWorkingNode = self
    pathElements = path.split(os.sep)
    while len(pathElements) > 0:
      for subNode in currentWorkingNode.subNodes:
        if subNode.name == pathElements[0]:
          currentWorkingNode = subNode
          break  # break out of the for-loop and skip the "else" clause below
      else:
        # 'path' led to a non-existant node
        return None

      del pathElements[0]

    return currentWorkingNode

  def findChildren(self, soughtName):
    """
    Return a list of nodes beneath this
    node whose names match 'soughtName'
    """
    foundNodes = []
    def __find(node):
      for subNode in node.subNodes:
        if subNode.name == soughtName:
          foundNodes.append(subNode)
        __find(subNode)

    __find(self)
    return foundNodes

  def getPathBelowRoot(self):
    """
    Return a path from the master-node to this node
    where each element in the path corresponds to a
    node's 'name' property.
    """
    pathElements = []
    thisNode = self
    while thisNode.parent:
      pathElements.insert(0, thisNode.name)
      thisNode = thisNode.parent

    return os.sep.join(pathElements)

  def getXml(self):
    """
    Return as a list the lines of xml code that represent this node,
    its subNodes, their subNodes, etc., and any text contained in the
    leaf nodes.
    """
    xml = []
    def _getXml(xmlNode):
      tagIndent  = "  " * (xmlNode.depth - 1)
      lineIndent = "  " * xmlNode.depth
      xml.append(tagIndent + ("<%s>" % xmlNode.name))
      for subNode in xmlNode.subNodes:
        _getXml(subNode)
      for line in xmlNode.text:
        xml.append(lineIndent + line)
      xml.append(tagIndent + ("</%s>" % xmlNode.name))

    if not self.parent:
      # the all-encompassing 'masterNode' should not appear
      # in the xml. The first-level subNodes will have zero
      # left-hand indentation
      for subNode in self.subNodes:
        _getXml(subNode)
    else:
      _getXml(self)

    return xml
    

def parseXml(pathToXml):
  """
  Given a path to an xml file, create a master node and parse the xml
  file into sub-nodes of the master node. Return the master node.
  """
  startTagPat = "<(?!/)(.*)>"  # matches '<' only if next char is not '/'
  endTagPat   = "</(.*)>"      # matches '<' followed by '/'

  def __parse(xmlNode, lineNum):
    while lineNum < len(xmlLines):
      thisLine = xmlLines[lineNum]
      lineNum += 1
      m = re.match(startTagPat, thisLine)
      if m:
        # 'thisLine' is a start-tag
        if len(xmlNode.text) > 0:
          if xmlNode.isMasterNode():
            msg = "Master-node contains both sub-nodes and text."
          else:
            msg = "Node \"%s\" contains both sub-nodes and text." % xmlNode.getPathBelowRoot()
          raise ParseError(msg)
        # else
        newNodeName = m.group(1)
        try:
          newNode = xmlNode.add(newNodeName)
        except NodeError, e:
          if xmlNode.isMasterNode():
            msg = "Master-node has multiple children named \"%s\"." % newNodeName
          else:
            msg = "Node \"%s\" has multiple children named \"%s\"." % (xmlNode.getPathBelowRoot(), newNodeName)
          raise ParseError(msg)
        else:
          lineNum = __parse(newNode, lineNum)
      else:
        m = re.match(endTagPat, thisLine)
        if m:
          # 'thisLine' is an end-tag
          if m.group(1) == xmlNode.name:
            return lineNum
          else:
            if xmlNode.isMasterNode():
              msg = ("closing tag \"%s\" has no corresponding opening tag." % m.group(1))
            else:
              msg = ("closing tag \"%s\" does not match opening tag \"%s\"." %
                     (m.group(1), xmlNode.getPathBelowRoot()))
            raise ParseError(msg)
        else:
          # 'thisLine' is text
          if len(xmlNode.subNodes) > 0:
            if xmlNode.isMasterNode():
              msg = "Master-node contains both sub-nodes and text."
            else:
              msg = "Node \"%s\" contains both sub-nodes and text." % xmlNode.getPathBelowRoot()
            raise ParseError(msg)
          # else
          xmlNode.text.append(thisLine)
    else:
      if xmlNode.depth > 0:
        msg = "Node \"%s\" has no closing tag." % xmlNode.getPathBelowRoot()
        raise ParseError(msg)
  
  # read in the xml text file and remove empty lines and comments
  xmlLines = open(pathToXml).read().strip().split("\n")
  xmlLines = [xmlLine.strip() for xmlLine in xmlLines
              if (len(xmlLine.strip()) > 0 and
                  not xmlLine.startswith("#"))]

  # create the master node that will contain all the nodes in the text file
  masterNode = XmlNode("masterNode")

  # "__parse()" will read the info from 'xmlLines' into 'masterNode'
  __parse(masterNode, 0)
  return masterNode
