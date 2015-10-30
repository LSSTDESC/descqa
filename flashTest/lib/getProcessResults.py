import sys, os, time

def getProcessResults(cmd, timeout=None):
  """
  executes 'cmd' as a child process and returns a tuple containing
  the child's stdout and stderr together, the stderr separately,
  the duration of execution, and the process exit status. Uses
  pseudo-terminals to capture output of child process immediately
  without buffering. Parent will abort the child if child fails to
  produce output for 'timeout' consecutive seconds.
  Many thanks go to Jess Balint of the Chicago python users' group
  for invaluable help in developing this code
  """
  import re, select

  if timeout == None:
    timeout = 60 * 60  # kill process if it produces
                      # no output for 60 minutes.

  # use a pipe for the stderr, because it's not file-buffered,
  # and a pseudo-terminal for the stdout
  (stdErrR, stdErrW) = os.pipe()
  (stdOutR, stdOutW) = os.openpty()

  # os.fork() will return the child's pid to the parent and 'None' to the child
  childPid = os.fork()

  if childPid:  # this is the parent
    out = ""
    err = ""
    stdOutDone = False
    stdErrDone = False

    # parent doesn't need this end (child does)
    os.close(stdErrW)
    os.close(stdOutW)

    startTime = time.time()
    wasAborted = False

    while 1:
      # wait for some results (with a possible timeout)
      (rl, wl, xl) = select.select([stdOutR, stdErrR], [], [stdOutR, stdErrR], timeout)

      if len(rl) == 0:
        # we timed out, so kill the child (vigorously)
        os.kill(childPid, 9)
        err += "\nProcess aborted by FlashTest after producing no output for %s seconds.\n" % timeout
        wasAborted = True
        break

      # else
      if stdOutR in rl:
        try:
          # capture the output of the child. Pseudo-terminals sometimes replace
          # regular newlines '\n' with '\r\n', so replace all instances of '\r'
          # with the empty string. This behavior of pseudo-terminals makes them
          # unsafe for use when the child's output is binary data.
          o = os.read(stdOutR, 100).replace("\r","")
        except OSError, e:
          # we might not be able to read the pty, so we trap the error and set
          # 'end' to True. We don't break because there might be something yet
          # to read from 'stdErrR'.
          stdOutDone = True
        else:
          if len(o): out += o
          else: stdOutDone = True

      if stdErrR in rl:
        # stdErr is a pipe, not a pseudo-terminal like stdOut, so we can read
        # from the stream without a try-except block, knowing that a pipe will
        # return the empty string on end-of-file instead of raising an exception
        e = os.read(stdErrR, 100)
        if e:
          # put error both in regular output stream and in its own
          out += e
          err += e
        else:
          # when the child process terminates gracefully on platforms
          # other than Irix, 'stdErrR' will appear in 'rl' and a call
          # to os.read() will return the empty string (sometimes)
          stdErrDone = True

      # when the child process terminates on Irix, it shows up
      # in the 'exceptional conditions' list returned by "select"
      if len(xl) > 0:
        break

      if stdOutDone and stdErrDone:
        # break only when both stdOut and stdErr streams are empty
        break

    endTime = time.time()
    duration = endTime - startTime
    if wasAborted:
      exitStatus = 9
    else:
      try:
        exitStatus = os.waitpid(childPid,0)[1]
      except Exception:  # if 'childPid' no longer exists for some reason
        exitStatus = "unknown"

    os.close(stdErrR)
    os.close(stdOutR)
    return (out, err, duration, exitStatus)

  else:  # this is the child

    # child doesn't need this end (parent does)
    os.close(stdErrR)
    os.close(stdOutR)

    # make sure we're clean
    sys.stdout.flush()
    sys.stderr.flush()

    # correct the new stdout and stderr
    os.dup2(stdOutW, sys.stdout.fileno())
    os.dup2(stdErrW, sys.stderr.fileno())

    # Go
    cmd = re.split("\s*", cmd.strip())
    try:
      os.execvp(cmd[0], cmd)
    except Exception, e:
      print e  # will go to 'stdOutW'

    # end of program. Nothing should happen after
    # 'exec' if the child has executed correctly.
