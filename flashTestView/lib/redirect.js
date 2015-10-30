function redirect(el) {
  // start loop at 1 to skip first
  // option, which is always a blank
  for (var i=1; i < el.options.length; i++) {
    if (el.options[i].selected == true) {
      location="home.py?target_dir=" + el.options[i].value;
    }
  }
}
