import os, re

class ParseError(Exception):
  pass

def __cleanLines(lines):

  l = 0
  while l < len(lines):
    c = 0
    while c < len(lines[l]):
      if lines[l][c] == '"':
        # jump forward to matching double-quote
        c = lines[l].find('"',c+1)
      elif lines[l][c] == "'":
        # jump forward to matching single-quote
        c = lines[l].find("'",c+1)
      elif lines[l][c] == "#":
        # trim off any comments (not btwn quotes)
        lines[l] = lines[l][:c]

      if c < 0:
        # one of the above finds came up empty
        # because of an unmatched quote mark
        break
      else:
        c += 1
    l += 1

  # remove blank lines
  lines = [line.strip() for line in lines if len(line.strip()) > 0]

  return lines


def fileToList(pathToFile):
  """
  Open 'filename' and parse its text into a list. The elements of
  the list will be the lines of the text as delimited by newlines.
  Comments and blank lines are elimnated in all cases.

  This function is called by "flashTest.py" to get lists of lines
  from "errors" and "files_to_delete" files, and also to turn user
  test-path files (such as follow the "-f" option to flashTest.py)
  into lists of arguments.
  """
  lines = open(pathToFile).read().split("\n")
  return __cleanLines(lines)


def getPathsAndOpts(args):

  def __linkEqualsSigns(args):
    """
    Join up any dangling equals signs, e.g.:

      test/path bar =baz   -->   test/path bar=baz

    such that 'bar=baz' becomes an option to test-path.
    Note, however, that the dangling equals in:

      foo bar= baz

    is not joined to 'baz', as this syntax indicates that
    'bar' is an option to test-path that takes no argument
    """
    args2 = []
    lastArg = None
    for arg in args:
      if lastArg and arg.startswith("="):
        if (lastArg + arg).count("=") > 1:
          raise ParseError, "Options to tests must be of form 'key=value', not '%s %s'" % (lastArg, arg)
        else:
          args2[-1] = args2[-1] + arg
      else:
        args2.append(arg)
      lastArg = arg
    return args2

  pathsAndOpts = []

  if args:
    args = __linkEqualsSigns(args)
    if args[0].startswith("="):
      raise ParseError, "test-option value \"%s\" has no corresponding key." % args[0]
    # else
    if args[0].count("=") > 0:
      raise ParseError, "option \"%s\" must follow a test-path" % args[0]
    # else

    i = 0
    while i < len(args):
      if args[i].startswith("-"):
        msg = ("The option \"%s\" is not allowed in this position.\n" % args[i] +
               "Options to flashTest.py must precede all test-paths.")
        raise ParseError, msg
      # else
      if args[i].count("=") >= 1:
        key, val = args[i].split("=",1)
        pathsAndOpts[-1][1][key] = val.strip("'\"")
      else:
        pathsAndOpts.append((args[i], {}))
      i+=1

  return pathsAndOpts


def parseCommandLine(args, standAloneOpts=None):
  flashTestOpts = {}

  # Options to flashTest.py must come immediately after the name of the
  # executable itself and before any paths or options to specific tests.
  i = 0
  while i < len(args):
    if not args[i].startswith("-"):
      break
    # else, this is an option
    if args[i] in standAloneOpts:
      # this option takes no argument
      flashTestOpts[args[i]] = ""
    else:
      # the option's argument is the next element in the list
      if i+1 == len(args):
        # but there are no more elements in the list!
        raise ParseError, "Option %s requires an argument" % args[i]
      else:
        flashTestOpts[args[i]] = args[i+1]
        i += 1
    i += 1

  # Now, everything left in 'args' should be paths to tests and
  # command-line arguments to those tests.
  pathsAndOpts = getPathsAndOpts(args[i:])

  return (flashTestOpts, pathsAndOpts)


def parseFile(pathToFile):
  """
  Take a path to a textfile of newline-delimited key/value pairs and
  return a corresponding dictionary.

  key/value pairs are denoted by:

    key: value

  section headers are given as:

    [header]

  The header will be prepended to any keys found below it so:

    [header]
    foo: fooval

  will yield {"header.foo": "fooval"} in 'returnDict'
  """
  lines = open(pathToFile).read().split("\n")
  return parseLines(lines)


def parseLines(lines):
  """
  Take some text of newline-delimited key/value pairs
  and return a corresponding dictionary
  """
  returnDict = {}
  header = ""

  lines = __cleanLines(lines)

  for line in lines:
    if re.search("^\S+?:", line):
      key = line.split(":",1)[0].strip()
      val = line.split(":",1)[1].strip()
      if header:
        key = "%s.%s" % (header, key)
      returnDict[key] = val
    elif re.search("^\[.*\]$", line):
      header = line.strip("[]").strip()

  return returnDict


def stringToList(s):
  """
  convenience method for converting 'parfiles' and 'transfers'
  strings into lists for use by flashTest.py
  """
  if len(s.strip()) > 0:
    return re.split("\s+",s.strip())
  else:
    return []
