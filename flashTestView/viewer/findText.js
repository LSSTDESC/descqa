/**
 * locate text 'soughtText' in window
 * and select it. Scrolls if necessary.
 */
function findText(soughtText) {
  var ua = window.navigator.userAgent.toLowerCase();

  // unfortunately there is no cross-browser way
  // to do this that I know of.
  if (ua.indexOf('gecko') != -1) {
    window.find(soughtText, true, false);
  }
  else if (ua.indexOf('msie') != -1) {
    var range = document.body.createTextRange();
    var found = range.findText(soughtText);
    range.select();
    range.scrollIntoView();
  }
  else {
    return;  // sorry, you're out of luck
  }
}
