/**
 * Returns array of all elements with name 'tagName'
 * which are inside element with id 'ancestorId'.
 * This function deliberatly avoids the built-in
 * function 'getElementsByName', which is incorrectly
 * implemented in IE
 */
function getElementsInElementByName(ancestorId, tagName) {
  var returnArray = new Array();
  function find(el) {
    if (el.attributes) {
      var nameAtt = el.attributes.getNamedItem('name');
      if (nameAtt && nameAtt.value == tagName)
        returnArray[returnArray.length] = el
    }
    if (el.hasChildNodes()) {
      for (var i=0; i < el.childNodes.length; i++) {
        find(el.childNodes[i])
      }
    }
  }
  var ancestor = document.getElementById(ancestorId);
  find(ancestor)
  return returnArray;
}


/**
 * Convenience method to only retrieve a single value
 */
function getElementInElementByName(ancestorId, tagName) {
  var returnArray = getElementsInElementByName(ancestorId, tagName);
  if (returnArray.length > 0)
    return returnArray[0];
  else
    return null;
}


/**
 * return the checkbox associated with 'node'
 */
function getCheckbox(node) {
  // this string of first children will bring us
  // to the <tr> element associated with this node
  var tr = node.navObj.firstChild.firstChild.firstChild;
  // loop through each <td> of the <tr> element to find its checkbox
  for (var j=0; j < tr.childNodes.length; j++) {
    if (tr.childNodes[j].firstChild != null) {
      if (tr.childNodes[j].firstChild.nodeName == "INPUT") {
        return tr.childNodes[j].firstChild;
      }
    }
  }
}



/**
 * Returns array of values of all checked checkboxes.
 * These values will be paths under the xml-tree-root.
 *
 * This function can cause an "unresponsive script"
 * alert in Mozilla. If this happens, try entering:
 * "about:config" in the Mozilla URL bar and setting
 * "dom.max_script_run_time" to a higher value.
 */
function getCheckedNodes() {
  var checkbox
  var checkedNodes = new Array();
  function find(node) {
    checkbox = getCheckbox(node);
    if (checkbox.checked == true) {
      checkedNodes[checkedNodes.length] = node;
    }
    // If this node is open, we check its children for
    // checked checkboxes- if it's not we don't need to bother
    if (node.isOpen) {
      for (var i=0; i < node.nChildren; i++) {
        find(node.children[i]);
      }
    }
  }
  // 'indexOfEntries[0]' corresponds to the node at the root
  // of the tree (the visible tree-widget, not the xml-tree),
  // so we start checking there
  find(indexOfEntries[0]);
  return checkedNodes;
}


/**
 * Returns an array of only those checked nodes
 * which are not descendents of other checked nodes
 */
function getCulledCheckedNodes() {
  var checkedNodes = getCheckedNodes();
  var culledCheckedNodes = new Array();
  var currentWorkingNode;
  var notAChild;

  for (var i=0; i < checkedNodes.length; i++) {
    currentWorkingNode = checkedNodes[i]
    notAChild = true;
    while (currentWorkingNode.parent) {
      for (var j=0; j < checkedNodes.length; j++) {
        if (currentWorkingNode.parent == checkedNodes[j]) {
          notAChild = false;  // the old 'breaking out of 2 loops'
          break;              // problem requires a boolean var.
        }
      }
      if (notAChild == false) {
        break;
      }
      else {
        currentWorkingNode = currentWorkingNode.parent;
      }
    }
    if (notAChild == true) {
      culledCheckedNodes[culledCheckedNodes.length] = checkedNodes[i];
    }
  }
  return culledCheckedNodes;
}


/**
 * Returns array of nodes with the given name
 */
function findNodes(name) {
  var foundNodes = new Array();
  var culledCheckedNodes = getCulledCheckedNodes();

  function find(node) {
    for (var i=0; i < node.nChildren; i++) {
      if (node.children[i].desc == name) {
        foundNodes[foundNodes.length] = node.children[i];
      }
      find(node.children[i]);
    }
  }
  for (var i=0; i < culledCheckedNodes.length; i++) {
    find(culledCheckedNodes[i]);
  }
  if (foundNodes.length > 0) {
    clearCheckboxes();
    for (var i=0; i < foundNodes.length; i++) {
      foundNodes[i].forceOpeningOfAncestorFolders();
      getCheckbox(foundNodes[i]).checked = true;
    }
  }
}


/**
 * Uses the hidden input tag in leftFrame.py to determine
 * the path to the info-file we're currently using
 */
function getPathToInfo() {
  var pathToInfo = document.getElementById("path_to_info").value;
  return pathToInfo
}


/**
 * Calls 'getCheckedNodes' and returns corresponding
 * paths formatted for use as a cgi-query string
 */
function getPathsAsQueryStr() {
  var retStr = ""

  var checkedNodes = getCheckedNodes()
  for (var i=0; i<checkedNodes.length; i++) {
    retStr = retStr + "&path=" + checkedNodes[i].path;
  }
  return retStr;
}


/**
 * Called if user clicks "view files"
 */
function viewFiles() {
  var pathsAsQueryStr = getPathsAsQueryStr();
  if (pathsAsQueryStr == "") {
    alert ("You have not selected any nodes");
    return;
  }
  // else
  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=view_files" + pathsAsQueryStr
  window.open("rightFrame.py?" + queryStr, "basefrm")
}


/**
 * Called if user clicks "edit files"
 */
function editFiles() {
  var pathsAsQueryStr = getPathsAsQueryStr();
  if (pathsAsQueryStr == "") {
    alert ("You have not selected any nodes");
    return;
  }
  // else
  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=edit_files" + pathsAsQueryStr;
  window.open("rightFrame.py?" + queryStr, "basefrm")
}


/**
 * Called if user clicks "clear checkboxes"
 */
function clearCheckboxes() {
  var checkedNodes = getCheckedNodes();
  for (var i=0; i < checkedNodes.length; i++) {
    getCheckbox(checkedNodes[i]).checked = false;
  }
}


/**
 * Called if user clicks "undo last change"
 */
function undoChanges() {
  window.open("leftFrame.py?undo_changes=1", "treeframe")
}


/**
 * Called if user clicks "home"
 */
function home() {
  var queryStr = "path_to_info=" + getPathToInfo();
  window.open("rightFrame.py?" + queryStr, "basefrm")
}

/**
 * Called if user clicks "main" (big board)
 */
function bigBoard() {
  top.location = "../home.py";
}

/**
 * Called if user clicks "NODE/TEXT" link
 */
function switchControls() {
  var nodeControls = document.getElementById('nodeControls');
  var fileControls = document.getElementById('fileControls');
  if (nodeControls.style.display == '') {
    nodeControls.style.display = 'none';
    fileControls.style.display = '';
  }
  else {
    nodeControls.style.display = '';
    fileControls.style.display = 'none';
  }
}


/**
 * Start with one of the control panels open
 */
function openPanel(blockId, firstVal, secondVal) {
  hideShowBlock(blockId);
  // DEV very hacky solution because I'm in a rush!
  // With this code, a script ("viewer/viewBuilds") can only
  // send values to the open panel if the panel is 'addKeys'.
  // Obviously this functionality should be extended to all
  // panels, but right now only the Comparison test uses it.
  // Luckily, regular 'openPanel' will keep working for all
  // calls to it since js lets you call a fn with fewer than
  // the declared number of parameters. i.e., I can call this
  // fn with just 'blockId' and it will still work
  if (blockId == "addKeys") {
    if (firstVal) {
      var key = getElementInElementByName(blockId, "key");
      key.value = firstVal;
    }
    if (secondVal) {
      var val = getElementInElementByName(blockId, "val");
      val.value = secondVal;
    }
  }
}


/**
 * Make sure string only consists of alphanumeric chars,
 * the underscore, or the dash
 */
function validName(str) {
  // regex returns array of all characters which
  // are non alphanumeric and not the underscore
  nonAlphas = str.match(/\W/);
  if (nonAlphas) {
    for (var i=0; i < nonAlphas.length; i++) {
      if (nonAlphas[i] != "-")
        return false;
    }
  }
  return true;
}


/**
 * Check user input for the "node management" controls
 */
function checkNodeManagementInput(cmd, id) {
  var checkedNodes = getCheckedNodes();

  // check that user has selected something
  if (checkedNodes.length == 0) {
    alert ("You have not selected any nodes");
    return;
  }

  // not allowed to rename the master node
  if (cmd == "rename_node" && checkedNodes[0].path == "") {
    alert ("You may not rename the master node");
    return;
  }

  // these operations are illegal on node at root of display
  if (cmd == "remove_node" || cmd == "clone_branch") {
    if (checkedNodes[0] == indexOfEntries[0]) {
      alert ("This operation may not be used on the node at the root of the display");
      return;
    }
  }

  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=" + cmd;

  var firstInput = document.getElementById(id).attributes.getNamedItem("firstInput").value

  if (cmd == "remove_node") {
    // here, 'newNode' is used to say whether we'll remove all of this
    // node's children or whether this node's parent will adopt them
    var newNode = getElementInElementByName(id, firstInput).checked;
    if (newNode) {
      queryStr += "&new_node=on";
    }
  }
  else {
    var newNode = getElementInElementByName(id, firstInput).value;

    // strip leading and trailing whitespace
    newNode = newNode.replace(/(^\s*|\s*$)/g, '')

    if (newNode == "") {
      alert("Node names must be at least one character long");
      return;
    }

    // else complain if non-alphanumeric characters are in name
    if (validName(newNode) == false) {
      alert("Node names may only contain alphanumeric characters, underscores, and dashes");
      return;
    }

    // else
    if (newNode.length > 0) {
      // Note that for "find_nodes", 'newNode' will not be the name of
      // a new node, but the name of the node(s) we're looking for
      queryStr += "&new_node=" + newNode
    }
  }
  if (cmd == "find_nodes") {
    // This command merely opens and closes tree-nodes,
    // but doesn't add or subtract them, so we don't need
    // to use "rightFrame.py" to do any heavy-lifting or
    // to refresh the tree (which can take a long time).
    // However, we still call "rightFrame.py" anyway, so
    // we can have a list of what nodes were found.
    queryStr += getPathsAsQueryStr();
    findNodes(newNode);
    window.open("rightFrame.py?" + queryStr, "basefrm");
  }
  else {
    // add paths to query string
    queryStr += getPathsAsQueryStr();
    window.open("rightFrame.py?" + queryStr, "basefrm");
  }
}


/**
 * Check user input for the "test.info" controls
 */
function checkInfoFileManagementInput(cmd, id) {
  // cmd will be one of: 'add_key', 'append_val',
  // 'remove_key', 'regex_find', or 'regex_replace'
  var checkedNodes = getCheckedNodes();
  if (checkedNodes.length == 0) {
    alert ("You have not selected any nodes");
    return;
  }

  var key = getElementInElementByName(id, 'key').value;
  var val = getElementInElementByName(id, 'val').value;
  if ((cmd == "regex_find") || (cmd == "regex_replace")) {
    if (key == "") {
      alert("You must enter a regular expression");
      return;
    }
  }
  else {
    if (key == "") {
      alert("You must indicate a key");
      return;
    }
    else if (validName(key) == false) {
      alert("names of keys should consist only of alphanumeric characters, underscores, and dashes");
      return;
    }
  }

  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=" + cmd + "&key=" + key + "&val=" + val
  queryStr += getPathsAsQueryStr();
  window.open("rightFrame.py?" + queryStr, "basefrm")
}


function hideShowBlock(blockId) {
  var upArrow   = "leftFrame/images/upArrow.gif"
  var downArrow = "leftFrame/images/downArrow.gif"
  var el = document.getElementById(blockId);
  if (el.style.display == 'none') {
    // hide all other control blocks in this panel
    var partOf = el.attributes.getNamedItem("partOf").value;
    var others = getElementsInElementByName(partOf, "controlBlock");
    for (var i=0; i < others.length; i++) {
      if ((others[i].id != blockId) && (others[i].style.display == '')) {
        var thisArrow = document.getElementById(others[i].id + "Arrow");
        thisArrow.src = downArrow;
        others[i].style.display = 'none';
      }
    }
    document.getElementById(blockId + "Arrow").src = upArrow;
    el.style.display = '';
    // give focus to first input element inside this block
    var firstInputName = el.attributes.getNamedItem("firstInput").value;
    getElementInElementByName(blockId, firstInputName).focus();
  }
  else {
    document.getElementById(blockId + "Arrow").src = downArrow;
    el.style.display = 'none';
  }
}


function goIfEnter(event, cmd, id) {
  // two methods for detecting which key was pressed
  // the first for IE, the second for Mozilla-type
  if (event.keyCode == "13") {
    switch(cmd) {
      case "find_nodes":  // all these fall through to "extend_branch"
      case "add_node":
      case "remove_node":
      case "rename_node":
      case "clone_branch":
      case "extend_branch":
        checkNodeManagementInput(cmd, id);
        break;
      default:
        checkInfoFileManagementInput(cmd, id);
        break;
    }
    // this line is essential, or pressing 'enter'
    // to submit will not work in IE!
    event.returnValue = false
  }
}
