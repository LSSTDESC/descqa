function getPathToInfo() {
  // sneak peek into the left frame to get this value
  var pathToInfo = top.treeframe.document.getElementById("path_to_info").value;
  return pathToInfo;
}

function getPaths() {
  var retStr = ""
  var paths = document.getElementsByName('path')
  for (var i=0; i<paths.length; i++) {
    retStr = retStr + "&path=" + paths[i].value;
  }
  return retStr;
}

function changeOK() {
  if (document.getElementById("commit").disabled == true) {
    // no changes have been introduced
    return true;
  }
  else {
    return confirm("You have made changes to the text of this node.\n" +
                   "Do you want to continue without committing?");
  }
}

function enableCommit() {
  var commit = document.getElementById("commit");
  commit.disabled = false;
}

function loadFile(editPath) {
  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=edit_files" + "&edit_path=" + editPath;
  queryStr += getPaths();
  window.open("rightFrame.py?" + queryStr, "basefrm");
}

function pickNewPathToInfo(el) {
  // find user's selection, skipping first "option", which
  // is not a path, but describes the <select> box's function
  for (var i=1; i < el.options.length; i++) {
    if (el.options[i].selected == true) {
      location = "rightFrame.py?path_to_info=" + getPathToInfo() + "&new_path_to_info=" + el.options[i].value;
      return;
    }
  }
  // nothing matched - user switched back to the
  // first option which is not really an option
  location = "rightFrame.py?path_to_info=" + getPathToInfo();
}

function redirect(el, pathToInfo) {
  var newPathToInfo;
  if (document.getElementById("new_path_to_info")) {
    newPathToInfo = document.getElementById("new_path_to_info").value;
  }
  else {
    newPathToInfo = getPathToInfo();
  }

  for (var i=0; i < el.options.length; i++) {
    if (el.options[i].selected == true) {
      var val = el.options[i].value;
      if (val == "/") {
        top.location="home.py?path_to_info=" + newPathToInfo;
      }
      else {
        top.location="home.py?path_to_info=" + newPathToInfo + "&start_node=" + val;
      }
    }
  }
}

function refreshTree(queryStr) {
  // The incoming 'queryStr' will have "start_checked"
  // and "start_visible" information from rightFrame.py

  // check for a "start_node" value in the left frame
  var pathToStartNode = top.treeframe.document.getElementById("start_node");
  if (pathToStartNode)
    queryStr = "start_node=" + pathToStartNode.value + "&" + queryStr;

  // find the "path_to_info" value in the left frame
  queryStr = "path_to_info=" + getPathToInfo() + "&" + queryStr;

  window.open("leftFrame.py?" + queryStr, "treeframe");
}

function URLencode(s) {
  /** 
   * regular javascript "escape" does not encode the plus
   * sign, the percent sign, or the slash, so we do these
   * manually. Thanks to the poorly laid-out but useful:
   * "http://64.18.163.122/rgagnon/" for this tidbit
   * The site also says that escape() does not encode the
   * single or double quote, but I have not found this to
   * be true in my browsers.
   */
  s = s.replace(/\%/g, '%25').
        replace(/\+/g, '%2B').
        replace(/\//g, '%2F');
  return escape(s)
}

function writeFile(writePath) {
  var text = URLencode(document.getElementById('text').value)
  var queryStr = "path_to_info=" + getPathToInfo();
  queryStr += "&cmd=write_file&write_path=" + writePath + "&text=" + text;
  queryStr += getPaths();
  window.open("rightFrame.py?" + queryStr, "basefrm");
}
