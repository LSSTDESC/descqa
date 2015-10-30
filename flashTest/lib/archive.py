import os
from getProcessResults import getProcessResults

class ArchiveError(Exception):
  pass

class Archive:

  def __init__(self, log, masterDict):
    self.log = log

    # these are all guaranteed to exist in 'masterDict' by FlashTest.py
    self.pathToFlashTest     = masterDict["pathToFlashTest"]
    self.pathToSiteDir       = masterDict["pathToSiteDir"]
    self.pathToInvocationDir = masterDict["pathToInvocationDir"]
    self.pathToLocalArchive  = masterDict["pathToLocalArchive"]
    self.mainArchiveHost     = masterDict["mainArchiveHost"]
    self.pathToMainArchive   = masterDict["pathToMainArchive"]
    self.viewArchiveHost     = masterDict["viewArchiveHost"]
    self.pathToViewArchive   = masterDict["pathToViewArchive"]

    # these two might or might not exist in 'masterDict'
    self.pathToMainArchiveGnuTar = masterDict.get("pathToMainArchiveGnuTar", "tar")
    self.pathToGnuTar            = masterDict.get("pathToGnuTar", "tar")

  def confirmInLocalArchive(self, pathToLocalFileOrDir):
    """
    Check for existence of 'pathToLocalFileOrDir' in local archive. If
    not there, extract it from requisite .tar.gz file in main archive.
    Return True on success or raise ArchiveError on failure.
    """
    pathToLocalFileOrDir = os.path.normpath(pathToLocalFileOrDir)

    log                = self.log
    pathToLocalArchive = self.pathToLocalArchive

    if not os.path.isdir(pathToLocalArchive):
      raise ArchiveError("\"%s\" does not exist or is not a directory." % pathToLocalArchive)

    if os.path.isabs(pathToLocalFileOrDir):
      if not pathToLocalFileOrDir.startswith(pathToLocalArchive):
        raise ArchiveError("Path \"%s\" should be an absolute path to a file beneath" % pathToLocalFileOrDir +
                           "local archive \"%s\"" % pathToLocalArchive +
                           "or a relative path which will be interpreted as a path" +
                           "to a file beneath \"%s\"" % pathToLocalArchive)
    else:
      pathToLocalFileOrDir = os.path.join(pathToLocalArchive, pathToLocalFileOrDir)

    # ideally, 'pathToLocalFileOrDir' already exists, i.e. the
    # requested file or directory is already in the local archive
    if os.path.exists(pathToLocalFileOrDir):
      return True

    # reconstruct the relative path from the local archive to the file or directory
    shortPathToLocalFileOrDir = pathToLocalFileOrDir[len(pathToLocalArchive):].strip(os.sep)

    # try to find the necessary file or directory in the main archive
    log.info("\"%s\" not found in local archive.\n" % shortPathToLocalFileOrDir +
             "Searching the main archive...")

    pathToMainArchive = self.pathToMainArchive
    if not pathToMainArchive:
      raise ArchiveError("No path to main archive provided.")

    if not os.path.isabs(pathToMainArchive):
      raise ArchiveError("Path to main archive \"%s\" must be an absolute path." % pathToMainArchive)

    # 'shortPathToLocalFileOrDir' becomes even shorter as
    # the site and the invocation are chopped off the front
    site, shortPathToLocalFileOrDir  = shortPathToLocalFileOrDir.split(os.sep,1)
    invoc, shortPathToLocalFileOrDir = shortPathToLocalFileOrDir.split(os.sep,1)
    pathToArchivedTarFile            = os.path.join(pathToMainArchive, site, invoc) + ".tar.gz"
    mainArchiveHost                  = self.mainArchiveHost

    # copy the archived tarfile into the local archive if possible
    if mainArchiveHost:  # archive is on a remote computer

      # test for existence of archived tar file
      cmd = "ssh %s test -e %s" % (mainArchiveHost, pathToArchivedTarFile)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to connect to %s\n%s" % (mainArchiveHost,
                                                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("File \"%s\" not found on \"%s\"" % (pathToArchivedTarFile,
                                                                mainArchiveHost))

      # else
      pathToMainArchiveGnuTar     = self.pathToMainArchiveGnuTar
      pathToArchivedTarFileParent = os.path.dirname(pathToArchivedTarFile)

      cmd = "ssh %s %s xzf %s -C %s %s" % (mainArchiveHost,
                                           pathToMainArchiveGnuTar,
                                           pathToArchivedTarFile,
                                           pathToArchivedTarFileParent,
                                           os.path.join(invoc, shortPathToLocalFileOrDir))
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to extract \"%s\" from \"%s:%s\"\n%s" %
                           (os.path.join(invoc, shortPathToLocalFileOrDir),
                            mainArchiveHost,
                            pathToArchivedTarFile,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error extracting \"%s\" from \"%s:%s\"" %
                           (exitStatus,
                            os.path.join(invoc, shortPathToLocalFileOrDir),
                            mainArchiveHost, pathToArchivedTarFile))

      # else prepare to bring remote copy over to local archive.
      # Make necessary directories if they don't already exist
      pathToLocalFileOrDirParent = os.path.dirname(pathToLocalFileOrDir)

      if not os.path.isdir(pathToLocalFileOrDirParent):
        try:
          os.makedirs(pathToLocalFileOrDirParent)
        except Exception, e:
          raise ArchiveError("Unable to create directory\n%s" % e)

      # else
      cmd = "scp -r %s:%s %s" % (mainArchiveHost,
                                 os.path.join(pathToArchivedTarFileParent, invoc, shortPathToLocalFileOrDir),
                                 pathToLocalFileOrDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to copy \"%s:%s\" to local archive.\n%s" %
                           (mainArchiveHost,
                            pathToArchivedTarFile,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error copying \"%s:%s\" to local archive." %
                           (exitStatus,
                            mainArchiveHost, pathToArchivedTarFile))

      # else delete extracted copy of file on remote host
      cmd = "ssh %s rm -rf %s" % (mainArchiveHost,
                                  os.path.join(pathToArchivedTarFileParent, invoc))
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        log.warn("Unable to delete remote copy of extracted archive \"%s:%s\"" %
                 (mainArchiveHost,
                  os.path.join(pathToArchivedTarFileParent, invoc),
                  err))
      # else
      if exitStatus != 0:
        log.warn("Exit status %s indicates error deleting remote copy of extracted archive \"%s:%s\"" %
                 (exitStatus,
                  mainArchiveHost,
                  os.path.join(pathToArchivedTarFileParent, invoc)))
        
    else:  #  archive is on local computer

      # test for existence of archived tar file
      if not os.path.isfile(pathToArchivedTarFile):
        raise ArchiveError("Unable to find file \"%s\"" % pathToArchivedTarFile)

      # else
      pathToGnuTar                = self.pathToGnuTar
      pathToArchivedTarFileParent = os.path.dirname(pathToArchivedTarFile)

      cmd = "%s xvzf %s -C %s %s" % (pathToGnuTar,
                                     pathToArchivedTarFile,
                                     pathToArchivedTarFileParent,
                                     os.path.join(invoc, shortPathToLocalFileOrDir))
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to extract \"%s\" from \"%s\"\n%s" %
                           (os.path.join(invoc, shortPathToLocalFileOrDir),
                            pathToArchivedTarFile,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error extracting \"%s\" from \"%s\"" %
                           (exitStatus,
                            os.path.join(invoc, shortPathToLocalFileOrDir),
                            pathToArchivedTarFile))

      # else prepare to bring remote copy over to local archive.
      # Make necessary directories if they don't already exist
      pathToLocalFileOrDirParent = os.path.dirname(pathToLocalFileOrDir)
      if not os.path.isdir(pathToLocalFileOrDirParent):
        try:
          os.makedirs(pathToLocalFileOrDirParent)
        except Exception, e:
          raise ArchiveError("Unable to create directory\n%s" % e)


      # else
      cmd = "cp -r %s %s" % (os.path.join(pathToArchivedTarFileParent, invoc, shortPathToLocalFileOrDir),
                             pathToLocalFileOrDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to copy \"%s\" to local archive.\n%s" %
                           (pathToArchivedTarFile,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error copying \"%s\" to local archive." %
                           (exitStatus,
                            pathToArchivedTarFile))

      # else delete extracted copy of file
      cmd = "rm -rf %s" % (os.path.join(pathToArchivedTarFileParent, invoc))
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        log.warn("Unable to delete remote copy of extracted archive \"%s\"" %
                 (os.path.join(pathToArchivedTarFileParent, invoc),
                  err))
      # else
      if exitStatus != 0:
        log.warn("Exit status %s indicates error deleting remote copy of extracted archive \"%s\"" %
                 (exitStatus,
                  os.path.join(pathToArchivedTarFileParent, invoc)))

    # confirm that file now exists in local archive
    if os.path.exists(pathToLocalFileOrDir):
      log.stp("Successfully transfered \"%s\" to \"%s\"" % (shortPathToLocalFileOrDir, pathToLocalArchive))
      if os.stat(pathToLocalFileOrDir)[6] == 0:
        log.warn("\"%s\" is zero bytes" % pathToLocalFileOrDir)
      return True

    #else
    raise ArchiveError("Was unable to transfer \"%s\" to \"%s\"" %
                       (shortPathToLocalFileOrDir, pathToLocalArchive))


  def makeTarFile(self):
    """
    Create a .tar.gz of this invocation's data
    """
    pathToGnuTar        = self.pathToGnuTar
    pathToInvocationDir = self.pathToInvocationDir
    pathToTarFile       = pathToInvocationDir + ".tar.gz"

    pathToInvocationDirParent, invocationDir = os.path.split(pathToInvocationDir)

    # -C means tar will first cd into 'pathToInvocationDirParent' before creating
    # the archive, so that files will not be archived with absolute path names
    cmd = "%s cvzf %s -C %s %s" % (pathToGnuTar,
                                   pathToTarFile,
                                   pathToInvocationDirParent,
                                   invocationDir)
    out, err, duration, exitStatus = getProcessResults(cmd)

    if err:
      if os.path.exists(pathToTarFile):
        os.remove(pathToTarFile)
      raise ArchiveError("Error creating archive \"%s\"\n%s" %
                         (pathToTarFile,
                          err))
    # else
    if exitStatus != 0:
      if os.path.exists(pathToTarFile):
        os.remove(pathToTarFile)
      raise ArchiveError("Exit status %s indicates error creating archive \"%s\"" %
                         (exitStatus, pathToTarFile))

    # else
    if not os.path.isfile(pathToTarFile):
      raise ArchiveError("Error creating archive \"%s\"" %
                         pathToTarFile)

    # else
    return pathToTarFile


  def sendTarFileToMainArchive(self, pathToTarFile):
    """
    copy tarfile indicated by 'pathToTarFile' to main
    archive, which might reside on a remote host
    """
    mainArchiveHost   = self.mainArchiveHost
    pathToMainArchive = self.pathToMainArchive

    if not pathToMainArchive:
      raise ArchiveError("No path to main archive provided.\n" +
                         "Unable to archive \"%s\"" %
                         pathToTarFile)
    # else
    pathToMainArchiveSiteDir = os.path.join(pathToMainArchive, os.path.basename(self.pathToSiteDir))

    if mainArchiveHost:  # main archive is on a remote computer
      # test if a directory with this site name exists on the main archive host
      cmd = "ssh %s test -d %s" % (mainArchiveHost, pathToMainArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to connect to %s\n%s" % (mainArchiveHost,
                                                            err))
      # else
      if exitStatus == 0:
        # The directory does exist. Does a tarfile with this name already exist too?
        tarFile = os.path.basename(pathToTarFile)
        cmd = "ssh %s test -e %s" % (mainArchiveHost, os.path.join(pathToMainArchiveSiteDir, tarFile))
        out, err, duration, exitStatus = getProcessResults(cmd)

        if err:
          raise ArchiveError("Unable to connect to %s\n%s" % (mainArchiveHost,
                                                              err))
        # else
        if exitStatus == 0:
          # a tarfile with this name already exists. We will not overwrite
          raise ArchiveError("A file \"%s\" already exists on \"%s\"\n" %
                             (os.path.join(pathToMainArchiveSiteDir, tarFile), mainArchiveHost) +
                             "Aborting copy to main archive")
      else:
        # directory doesn't exist, so try to create it
        cmd = "ssh %s mkdir %s" % (mainArchiveHost, pathToMainArchiveSiteDir)
        out, err, duration, exitStatus = getProcessResults(cmd)

        if err:
          raise ArchiveError("Error creating directory \"%s\" on host \"%s\"\n%s" %
                             (pathToMainArchiveSiteDir,
                             mainArchiveHost,
                             err))
        # else
        if exitStatus != 0:
          raise ArchiveError("Exit status %s indicates error creating directory \"%s\" on host \"%s\"" %
                             (exitStatus, pathToMainArchiveSiteDir, mainArchiveHost))

        # else we assume the directory was created

      # copy the tarfile into the remote archive
      cmd = "scp %s %s:%s" % (pathToTarFile, mainArchiveHost, pathToMainArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Error transfering \"%s\" to \"%s:%s\"\n%s" %
                           (pathToTarFile,
                            mainArchiveHost,
                            pathToMainArchiveSiteDir,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error transfering \"%s\" to \"%s:%s\"" %
                           (exitStatus, pathToTarFile, mainArchiveHost, pathToMainArchiveSiteDir))

      # else we assume archive was successfully copied

    else:  # main archive is on local computer
      # test if a directory with this site name exists
      if not os.path.isdir(pathToMainArchiveSiteDir):
        try:
          os.mkdir(pathToMainArchiveSiteDir)
        except Exception, e:
          raise ArchiveError(e)

      cmd = "cp %s %s" % (pathToTarFile,
                          pathToMainArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Error copying \"%s\" to \"%s\"\n%s" %
                           (pathToTarFile,
                            pathToMainArchiveSiteDir,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error copying \"%s\" to \"%s\"" %
                           (exitStatus, pathToTarFile, pathToMainArchiveSiteDir))

      # else we assume archive was successfully copied
    return True


  def sendToViewArchive(self):
    """
    Recursively scp this invocation's data and to view archive,
    which might reside on a remote host. This is very similar to
    "sendToMainArchive()", but doesn't create a tarball
    """
    viewArchiveHost     = self.viewArchiveHost
    pathToViewArchive   = self.pathToViewArchive
    pathToInvocationDir = self.pathToInvocationDir
    pathToTarFile       = pathToInvocationDir + ".tar.gz"

    if not pathToViewArchive:
      raise ArchiveError("No path to view archive provided.\n" +
                         "Unable to archive data from \"%s\"" % self.pathToInvocationDir)
    # else
    pathToViewArchiveSiteDir = os.path.join(pathToViewArchive, os.path.basename(self.pathToSiteDir))

    if viewArchiveHost:  # view archive is on a remote computer
      # test if a directory with this site name exists on the view archive host
      cmd = "ssh %s test -d %s" % (viewArchiveHost, pathToViewArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Unable to connect to %s\n%s" % (viewArchiveHost,
                                                            err))
      # else
      if exitStatus == 0:
        # The directory does exist. Does a directory with this name already exist too?
        invocationDir = os.path.basename(pathToInvocationDir)
        cmd = "ssh %s test -d %s" % (viewArchiveHost, os.path.join(pathToViewArchiveSiteDir, invocationDir))
        out, err, duration, exitStatus = getProcessResults(cmd)

        if err:
          raise ArchiveError("Unable to connect to %s\n%s" % (viewArchiveHost,
                                                              err))
        # else
        if exitStatus == 0:
          # a directory with this name already exists. We will not overwrite
          raise ArchiveError("A directory \"%s\" already exists on \"%s\"\n" %
                             (os.path.join(pathToViewArchiveSiteDir, invocationDir), viewArchiveHost) +
                             "Aborting copy to view archive.")
      else:
        # directory doesn't exist, so try to create it
        cmd = "ssh %s mkdir %s" % (viewArchiveHost, pathToViewArchiveSiteDir)
        out, err, duration, exitStatus = getProcessResults(cmd)

        if err:
          raise ArchiveError("Error creating directory \"%s\" on host \"%s\"\n%s" %
                             (pathToViewArchiveSiteDir,
                              viewArchiveHost,
                              err))
        # else
        if exitStatus != 0:
          raise ArchiveError("Exit status %s indicates error creating directory \"%s\" on host \"%s\"" %
                             (exitStatus, pathToViewArchiveSiteDir, viewArchiveHost))

        # else we assume the directory was created


      # recursively copy the data into the remote archive 
      cmd = "scp -rC %s %s:%s" % (pathToInvocationDir, viewArchiveHost, pathToViewArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Error transfering \"%s\" to \"%s:%s\"\n%s" %
                           (pathToTarFile,
                            viewArchiveHost,
                            pathToViewArchiveSiteDir,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error transfering \"%s\" to \"%s:%s\"" %
                           (exitStatus, pathToTarFile, viewArchiveHost, pathToViewArchiveSiteDir))

      # else we assume data was successfully copied - update last access time
      # on view archive so FlashTestView will refresh page on next browser hit
      cmd = "ssh touch %s" % pathToViewArchive
      getProcessResults(cmd)  # never mind if it worked or not - we tried our best

    else:  # view archive is on local computer
      # test if a directory with this site name exists
      if not os.path.isdir(pathToViewArchiveSiteDir):
        try:
          os.mkdir(pathToViewArchiveSiteDir)
        except Exception, e:
          raise ArchiveError(e)

      cmd = "cp -r %s %s" % (pathToInvocationDir, pathToViewArchiveSiteDir)
      out, err, duration, exitStatus = getProcessResults(cmd)

      if err:
        raise ArchiveError("Error copying \"%s\" to \"%s\"\n%s" %
                           (pathToTarFile,
                            pathToViewArchiveSiteDir,
                            err))
      # else
      if exitStatus != 0:
        raise ArchiveError("Exit status %s indicates error copying \"%s\" to \"%s\"" %
                           (exitStatus, pathToTarFile, pathToViewArchiveSiteDir))

      # else we assume archive was successfully copied - update last access time
      # on view archive so FlashTestView will refresh page on next browser hit
      try:
        os.utime(pathToViewArchive, None)
      except Exception:
        pass # oh well, we did our best

    return True
