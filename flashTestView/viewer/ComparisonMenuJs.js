function ComparisonSubmit() {
  /**
   * Make sure user has not checked builds that are associated with
   * more than one class of tester object. If there are no problems,
   * redirect the user's browser to the infofile manager
   *
   * FlashTest records the class of tester object used for a given
   * build in the file "linkAttributes", thus:
   * 
   *   testerClass: SfocuTester
   *
   * When the results are viewed with FlashTestView, viewBuilds.py
   * parses the "linkAttributes" file and incorporates the key/value
   * pairs found therein into attribute names and values assigned to
   * the checkbox elements of the associated build, where they can
   * be read by javascript.
   */
  var checkedCheckboxes = getCheckedCheckboxes();
  var testerClasses = new Array();
  var foundMatch;
  var el;
  var testerClass;

  for (var i=0; i < checkedCheckboxes.length; i++) {
    foundMatch = false;
    el = checkedCheckboxes[i].attributes.getNamedItem("testerClass");
    if (el) {
      testerClass = el.value;
      for (var j=0; j < testerClasses.length; j++) {
        if (testerClasses[j] == testerClass) {
          foundMatch = true;
          break;
        }
      }
      if (foundMatch == false) {
        testerClasses[testerClasses.length] = testerClass;
      }
    }
  }

  if (testerClasses.length > 1) {
    var msg = "";
    for (var i=0; i < testerClasses.length; i++) {
      msg += testerClasses[i] + "\n";
    }
    alert("You may only update benchmarks for one class of tester at a time.\n" +
          "You currently have selected builds with the following classes of tester:\n" +
          msg);
    return false;
  }

  // we know now that the boxes the user has checked are good for a
  // legitimate submit action, and that the class of tester object
  // used in all builds is recorded in 'testerClasses[0]'

  // Now we construct the query string to send to the infofile manager...

  // Start by telling the info file editor which info-file to use
  // and which node within that file to use as the base node.
  // The "path_to_info" and "site" tags are guaranteed to exist
  // by viewBuilds.py and viewBuildsTemplate.ezt
  var pathToInfo = document.getElementById("pathToInfo").value;
  var site = document.getElementById("site").value; 
  var queryStr = "path_to_info=" + pathToInfo + "&start_node=" + site;

  // add the checkboxes we've picked
  queryStr += "&" + getCheckboxesAsQueryStr();

  // add an instruction to view them
  queryStr += "&view_files=1"

  // determine what key and value we'll send to the infofile manager
  var newKey;
  var newValue;
  var dateSelector = document.getElementById("dateSelector");
  var selectedDate = dateSelector.options[dateSelector.selectedIndex].value;

  if (testerClasses[0] == "SfocuTester") {
    newKey = "shortPathToBenchmark";
    newValue = "<siteDir>/" + selectedDate + "/<buildDir>/<runDir>/<chkMax>"
  }
  if (testerClasses[0] == "GridDumpTester") {
    newKey = "shortPathToBenchmarkDir";
    newValue = "<siteDir>/" + selectedDate + "/<buildDir>/<runDir>"
  }
  
  queryStr += "&open_panel=addKeys";
  queryStr += "&key=" + newKey;
  queryStr += "&value=" + newValue;

  location = "../infoFileManager/home.py?" + queryStr
}
