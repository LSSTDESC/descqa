def parseFile(path):
  """
  Take a path to a textfile of newline-delimited
  key/value pairs and return a dictionary of strings
  mapped to strings or strings mapped to lists:

  examples:

    key: value  ->  {key: value}
    key: value1, value2 -> {key: [value1, value2]}
  """
  returnDict = {}

  lines = open(path,"r").readlines()
  # eliminate blank lines
  lines = [line.strip() for line in lines if len(line.strip()) > 0]
  # eliminate comments
  lines = [line for line in lines if not line.startswith("#")]
  for line in lines:
    if line.count(":") > 0:
      key, value = line.split(":",1)
      key = key.strip()
      value = value.strip()
      # Check if value contains commas. If so, return a list
      if value.count(",") > 0:
        value = value.split(",")
        value = [element.strip() for element in value]
      else:
        if key == "wallClockTime":
          if returnDict.has_key(key):
            value = returnDict[key] + " + " + value
        if key == "numProcs":
          if returnDict.has_key(key):
            if (value != returnDict[key]):
              value = returnDict[key] + " / " + value
      returnDict[key] = value

  return returnDict
