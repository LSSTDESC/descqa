import os, re

class Template:
  def __init__(self):
    self.divHeader = ""

  def insertJs(self, pathToInvocationDir):
    return ""

  def insertHtml(self, pathToInvocationDir):
    return ""

class Comparison(Template):
  """
  use sfocu to compare a checkpoint file against
  a user-determined benchmark
  """
  def __init__(self):
    self.divHeader = "Update Comparison Test Benchmarks"

  def insertJs(self, pathToInvocationDir):
    """
    custom javascript for the "update comparison benchmarks" box
    """
    js = open("ComparisonMenuJs.js").read()
    return js

  def insertHtml(self, pathToInvocationDir):
    """
    custom html that builds the "update comparison benchmarks" box
    """
    pathToInvocationDirParent = os.path.dirname(pathToInvocationDir) 

    items    = os.listdir(pathToInvocationDirParent)
    dateDirs = []

    for item in items:
      if (os.path.isdir(os.path.join(pathToInvocationDirParent, item)) and
          re.match("^\d\d\d\d-\d\d-\d\d.*", item)):
        dateDirs.append(item)

    # sort 'dateDirs' and reverse it so most recent
    # date will be at the top of our select element
    dateDirs.sort()
    dateDirs.reverse()

    # it's hard to put together long strings that look nice in the source code
    # as well, so I'm using a list which I join together at the end with newlines
    html = []

    # the visible input element that lets the user pick a date
    html.append("<table><tr><td valign=\"top\" style=\"font-size: 12px\">")
    html.append("Select the date of the new benchmark and click \"submit\" when ready.")
    html.append("\"test.info\" files for selected builds will be adjusted")
    html.append("to use the chosen benchmark.")
    html.append("</td><td valign=\"top\">")
    html.append("<select id=\"dateSelector\">")

    for dateDir in dateDirs:
      html.append("<option value=\"%s\">%s</option>" % (dateDir, dateDir))
    html.append("</select>")
    html.append("</td></tr></table>")
    return "\n".join(html)

#DEV: Need to change this to composite for production
class Composite(Template):
  """
  This is similar to a comparison test, except that it runs a restart case
  in line with a compariosn test.  User defines the comparison and restart
  benchmarks.
  """

  def __init__(self):
    self.divHeader = "Update Composite Test Benchmarks"

  def insertJs(self, pathToInvocationDir):
    """
    custom javascript for the "update comparison benchmarks" box
    """
    js = open("CompositeMenuJs.js").read()
    return js

  def insertHtml(self, pathToInvocationDir):
    """
    custom html that builds the "update comparison benchmarks" box
    """
    pathToInvocationDirParent = os.path.dirname(pathToInvocationDir) 

    items    = os.listdir(pathToInvocationDirParent)
    dateDirs = []

    for item in items:
      if (os.path.isdir(os.path.join(pathToInvocationDirParent, item)) and
          re.match("^\d\d\d\d-\d\d-\d\d.*", item)):
        dateDirs.append(item)

    # sort 'dateDirs' and reverse it so most recent
    # date will be at the top of our select element
    dateDirs.sort()
    dateDirs.reverse()

    # it's hard to put together long strings that look nice in the source code
    # as well, so I'm using a list which I join together at the end with newlines
    html = []

    # the visible input element that lets the user pick a date
    html.append("<table><tr><td valign=\"top\" style=\"font-size: 12px\">")
    html.append("Select the date of the new benchmark and click \"submit\" when ready.")
    html.append("\"test.info\" files for selected builds will be adjusted")
    html.append("to use the chosen benchmark.")
    html.append("</td><td valign=\"top\">")
    html.append("<form name=\"updateType\">")
    html.append("<input type=\"radio\" name=\"testType\"")
    html.append(" value=\"comparison\" checked>Comparison<br/>")
    html.append("<input type=\"radio\" name=\"testType\"")
    html.append(" value=\"restart\">Restart<br/>")
    html.append("</form>")

    html.append("<select id=\"dateSelector\">")

    for dateDir in dateDirs:
      html.append("<option value=\"%s\">%s</option>" % (dateDir, dateDir))
    html.append("</select>")

    html.append("</td></tr></table>")
    return "\n".join(html)
    
