import re, os

class EntryPointTemplate:
  def __init__(self, testObject):
    self.owner      = testObject
    self.masterDict = testObject.masterDict

  def entryPoint1(self):
    return True

  def entryPoint2(self):
    return True

  def entryPoint3(self):
    return True


class SetupperTemplate:
  def __init__(self, testObject):
    self.owner = testObject
    self.masterDict = testObject.masterDict

  def setup(self):
    log = self.masterDict["log"]  # guaranteed to exist by flashTest.py
    log.err("No implementation provided for method \"setup\".")
    return False


class CompilerTemplate:
  def __init__(self, testObject):
    self.owner = testObject
    self.masterDict = testObject.masterDict

  def compile(self):
    log = self.masterDict["log"]  # guaranteed to exist by flashTest.py
    log.err("No implementation provided for method \"compile\".")
    return False

  def getDeletePatterns(self):
    return []

  def adjustFilesToDelete(self, filesToDelete):
    pass

  def writeFilesToDelete(self):
    """
    Gets a list of regular expressions from "getDeletePatterns()" and tries
    to match them against the files found in 'pathToBuildDir'. If any matches
    are found, the list is submitted to 'adjustFilesToDelete()' for possible
    modification. Afterwards, any elements remaining in 'filesToDelete' will
    be written into the text file "files_to_delete". The files indicated in
    this file will be deleted after the invocation is complete.

    The user may use 'adjustFilesToDelete' to remove from the list any files
    that would normally be deleted.
    """
    pathToBuildDir = self.masterDict["pathToBuildDir"]  # guaranteed to exist by flashTest.py

    deletePatterns = self.getDeletePatterns()

    filesToDelete = []
    files = [item for item in os.listdir(pathToBuildDir) if os.path.isfile(os.path.join(pathToBuildDir, item))]
    for file in files:
      for deletePattern in deletePatterns:
        if re.match(deletePattern, file):
          filesToDelete.append(file)

    if filesToDelete:
      filesToDelete.sort()
      self.adjustFilesToDelete(filesToDelete)

    if filesToDelete:
      open(os.path.join(pathToBuildDir, "files_to_delete"),"w").write("\n".join(filesToDelete))


class ExecuterTemplate:
  def __init__(self, testObject):
    self.owner = testObject
    self.masterDict = testObject.masterDict

  def execute(self):
    log = self.masterDict["log"]  # guaranteed to exist by flashTest.py
    log.err("No implementation provided for method \"execute\".")
    return False

  def getDeletePatterns(self):
    return []

  def adjustFilesToDelete(self, filesToDelete):
    pass

  def writeFilesToDelete(self):
    """
    Gets a list of regular expressions from "getDeletePatterns()" and tries
    to match them against the files found in 'pathToRunDir'. If any matches
    are found, the list is submitted to 'adjustFilesToDelete()' for possible
    modification. Afterwards, any elements remaining in 'filesToDelete' will
    be written into the text file "files_to_delete". The files indicated in
    this file will be deleted after the invocation is complete.

    The user may use 'adjustFilesToDelete' to remove from the list any files
    that would normally be deleted.
    """
    pathToRunDir = self.masterDict["pathToRunDir"]  # guaranteed to exist by flashTest.py

    deletePatterns = self.getDeletePatterns()

    filesToDelete = []
    files = [item for item in os.listdir(pathToRunDir) if os.path.isfile(os.path.join(pathToRunDir, item))]
    for file in files:
      for deletePattern in deletePatterns:
        if re.match(deletePattern, file):
          filesToDelete.append(file)

    if filesToDelete:
      filesToDelete.sort()
      self.adjustFilesToDelete(filesToDelete)

    if filesToDelete:
      open(os.path.join(pathToRunDir, "files_to_delete"),"w").write("\n".join(filesToDelete))


class TesterTemplate:
  def __init__(self, testObject):
    self.owner      = testObject
    self.masterDict = testObject.masterDict

  def openOutfile(self):
    pathToRunDir    = self.masterDict["pathToRunDir"]     # guaranteed to exist by flashTest.py
    pathToFlashTest = self.masterDict["pathToFlashTest"]  # guaranteed to exist by flashTest.py
    
    os.chdir(pathToRunDir)
    outfile = open("test_output","w")
    self.masterDict["outfile"] = outfile

    os.chdir(pathToRunDir)

  def test(self):
    log = self.masterDict["log"]  # guaranteed to exist by flashTest.py
    log.err("No implementation provided for method \"test\".")
    return False

  def closeOutfile(self):
    pathToFlashTest = self.masterDict["pathToFlashTest"]  # guaranteed to exist by flashTest.py
    outfile = self.masterDict["outfile"]                  # guaranteed to exist by openOutfile() (above)

    self.masterDict["outfile"].close()

    os.chdir(pathToFlashTest)
