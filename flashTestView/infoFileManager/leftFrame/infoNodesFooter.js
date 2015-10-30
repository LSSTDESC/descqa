

// This function, setInitialLayout, overrides one in
// 'ftiens4.js' with the same name.
//
// In this version, any nodes listed in 'startVisible' or
// 'startChecked' (defined in 'infoNodesHeader.txt' and
// populated by javascript written by 'leftFrame.py') will
// be opened, along with their ancestors, when the page is
// first viewed. Also, the checkboxes of any nodes listed
// in 'startChecked' will be checked. This was necessary
// for the functioning of the benchmarks manager and a
// nice feature for the directory management tools.

function setInitialLayout() {
  if (browserVersion > 0 && !STARTALLOPEN) {
    clickOnNodeObj(fld0);
    var i;
    // open all folders in either 'startChecked' or
    // 'startVisible' and their ancestors
    for (i=0; i<startChecked.length; i++) {
      startChecked[i].forceOpeningOfAncestorFolders();
    }
    for (i=0; i<startVisible.length; i++) {
      startVisible[i].forceOpeningOfAncestorFolders();
    }
    // check checkboxes next to folders in 'startChecked'
    for (i=0; i<startChecked.length; i++) {
      // find the checkbox associated with this node
      // and check it. See "ftiens4.js" for similar code
      var tr = startChecked[i].navObj.firstChild.firstChild.firstChild;
      // loop through each <td> of the <tr> element
      for (var j=0; j < tr.childNodes.length; j++) {
        if (tr.childNodes[j].firstChild != null) {
          if (tr.childNodes[j].firstChild.nodeName == "INPUT") {
            tr.childNodes[j].firstChild.checked = true;
          }
        }
      }
    }
    // paths in startChecked or startVisible which for some
    // reason are not in the info files tree end up here
    if (notFound.length > 0) {
      var text = "The following 'test.info' files were not found:";
      for (i=0; i<notFound.length; i++) {
        text += "\n" + notFound[i]
      }
      alert(text);
    }
  }
}
