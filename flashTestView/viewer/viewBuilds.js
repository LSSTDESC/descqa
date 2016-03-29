function getCheckedCheckboxes() {
  var checkedCheckboxes = new Array();
  var allCheckboxes = document.getElementsByName("build");
  if (allCheckboxes) {
    for (var i=0; i < allCheckboxes.length; i++) {
      if (allCheckboxes[i].checked == true) {
        checkedCheckboxes[checkedCheckboxes.length] = allCheckboxes[i];
      }
    }
  }
  return checkedCheckboxes;
}
  

function getCheckboxesAsQueryStr() {
  var queryStr = "";
  // gather values of all checked checkboxes
  var checkboxes = document.getElementsByName("build");
  if (checkboxes) {
    for (var i=0; i < checkboxes.length; i++) {
      if (checkboxes[i].checked == true) {
         if (queryStr.length == 0)
           queryStr += "start_checked=" + checkboxes[i].value;
         else
           queryStr += "&start_checked=" + checkboxes[i].value;
      }
    }
  }
  return queryStr;
}  

/**
 * returns array of all elements with name 'tagName'
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
 * convenience method to only retrieve a single value
 */
function getElementInElementByName(ancestorId, tagName) {
  var returnArray = getElementsInElementByName(ancestorId, tagName);
  if (returnArray.length > 0)
    return returnArray[0];
  else
    return null;
}

function selectAll(testName) {
  var checkboxes = document.getElementsByName("build");
  if (checkboxes) {
    for (var i=0; i < checkboxes.length; i++) {
      if (checkboxes[i].attributes.getNamedItem("testname").value == testName)
        checkboxes[i].checked = true;
    }
  }
}

function clearAll() {
  var checkboxes = document.getElementsByName("build");
  if (checkboxes) {
    for (var i=0; i < checkboxes.length; i++) {
      checkboxes[i].checked = false;
    }
  }
}

function signalWrongTests(testName) {
  var checkboxes = document.getElementsByName("build");
  if (checkboxes) {
    for (var i=0; i < checkboxes.length; i++) {
      if ((checkboxes[i].attributes.getNamedItem("testname").value != testName) &&
          (checkboxes[i].checked == true)) {
        alert ("Please uncheck all boxes which do not\ncorrespond to tests of type " + testName);
        return false;
      }
    }
  }
  return true;
}

function launchEditor(startNode, pathToInfo) {
  // when user clicks the "edit" button, values from
  // checkboxes are collected and the infoFile editor
  // is launched
  var checkboxesAsQueryStr = getCheckboxesAsQueryStr()
  if (checkboxesAsQueryStr.length == 0) {
     alert("You have not selected any builds.");
  }
  else {
    var queryStr;
    if (pathToInfo.length > 0) {
      queryStr = "path_to_info=" + pathToInfo + "&start_node=" + startNode + "&" + checkboxesAsQueryStr;
    }
    else {
      queryStr = "start_node=" + startNode + "&" + checkboxesAsQueryStr;
    }
    queryStr += "&edit_files=1";
    location = "../infoFileManager/home.cgi?" + queryStr
  }
}

function checkSubmitTemplate() {
  return true;
}
