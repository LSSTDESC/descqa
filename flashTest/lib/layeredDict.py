import re

class LayeredDict:
  """
  Object that contains a list of dictionaries, 'dicts', and implements
  many of the methods of a python dictionary.

  The get, set, and del operations:

    foo = d["foo"]

    d["foo"] = foo

    del d["foo"]

  operate on the dictionary referenced by the value of 'activeLayer'.

  The 'setActiveLayer' method sets this value and deletes the dictionary
  at the specified value as well as all dictionaries at higher-indexed
  positions. See the docstring for setActiveLayer() for more information.

  Additionally, LayeredDict allows keys to have values that are pointers
  to other keys. See the docstring for the "__getitem__" method for more
  information.
  """
  pat1 = re.compile("(<(.*?)>)")  # regex used to look for pointers in values

  def __init__(self, d={}):
    self.activeLayer = 0
    self.dicts = []
    if d.__class__.__name__ == "dict":
      self.dicts.append(d)
    elif d.__class__.__name__ == "LayeredDict":
      for innerDict in d.dicts:
        self.dicts.append(innerDict.copy())
    else:
      dType = type(d).__name__
      raise TypeError("Objects of type '%s' may not be passed to the contructor of class LayeredDict." % dType)

  def __contains__(self, key):
    if self.has_key(key):
      return True
    return False

  def __delitem__(self, key):
    return self.dicts[self.activeLayer].__delitem__(key)

  def __getitem__(self, key):
    """
    Examine the value of the given key and resolve any pointers,
    which are strings enclosed in angle brackets, to real values
    from the dictionary, if available. Return the new value.
    
    e.g., if LayeredDict 'd' has key/value pairs:

      foo: 'Mike'
      bar: '<foo>'
    
    Querying d["bar"] yields 'Mike'

    The presence of multiple pointer elements and/or non-pointer
    does not cause a problem.

    e.g., if LayeredDict 'd' has key/value pairs:

      foo: 'Mike'
      bar: '"My name is <foo>", said <foo>'

    Querying d["bar"] yields '"My name is Mike", said Mike'

    A pointer whose value is keyed to other pointers will still
    resolve to a real value as long as one is available, regardless
    of the length of the list of references in between.

    e.g., if LayeredDict 'd' has key/value pairs:

      foo: 'Mike'
      bar: '<foo>'
      baz: '<bar>'

    Querying d["baz"] still yields "Mike"

    A string enclosed in angle-brackets that does not correspond to
    a key in the dictionary will be returned normally.

    e.g., if LayeredDict 'd' has key/value pairs:

      foo: 'Mike'
      bar: '<foo>'
      baz: '<bupkus>'

    Querying d["baz"] yields the string "<bupkus>"

    Pointers contained inside lists, tuples, and dictionaries will be
    resolved wherever possible. This extends to recursively nested
    lists, tuples, dictionaries, etc.

    Note, however, that a pointer may only be resolved to a key whose
    corresponding value is a string or another pointer or sequence of
    pointers that ultimately terminate in a string.

    e.g., if LayeredDict 'd' has key/value pairs:

      foo: 'Mike'
      bar: ['hello', '<foo>']
      baz: '<bar>'

    Querying d["bar"] yields the list ['hello', 'Mike']
    but querying d["baz"] causes a TypeError.

    At first it might seem that d["baz"] should return the same as
    d["bar"], but consider the case where "baz" is keyed to a string
    that contains more elements than just the pointer...

      baz: 'Here is a list: <bar>'

    In this case it is not clear exactly how the referent of "<bar>",
    a list, should be joined to the string elements of "baz". This
    use case therefore raises the TypeError.
    """
    n = self.activeLayer
    while n >= 0:
      if self.dicts[n].has_key(key):
        val = self.dicts[n][key]
        return self.__dereferencePointers(val)
      else:
        n -= 1
    raise KeyError

  def __repr__(self):
    return self.__str__()

  def __setitem__(self, key, value):
    return self.dicts[self.activeLayer].__setitem__(key, value)

  def __str__(self):
    s = ""
    for i in range(len(self.dicts)):
      s += "%s: %s\n" % (i, self.dicts[i])
    return s.strip()

  def __dereferencePointers(self, val):
    """
    Recursively search through lists, tuples, and dictionaries
    for pointers (strings), dereferencing wherever possible.
    """
    if isinstance(val, str):
      soughtOpts = self.pat1.findall(val)
      for soughtOpt, soughtOptStripped in soughtOpts:
        if self.has_key(soughtOptStripped):
          val = val.replace(soughtOpt, self[soughtOptStripped])
    elif isinstance(val, list):
      val = [self.__dereferencePointers(item) for item in val]
    elif isinstance(val, tuple):
      val = tuple([self.__dereferencePointers(item) for item in val])
    elif isinstance(val, dict):
      val = dict([(key, self.__dereferencePointers(val[key])) for key in val])
    return val

  def copy(self):
    return LayeredDict(self)

  def despecify(self, keyword=""):
    for key in self.keys():
      m = re.match("(%s\.)(.*)" % keyword, key)
      if m: self[m.group(2)] = self[key]

  def get(self, key, alt=None):
    if self.has_key(key):
      return self[key]
    return alt

  def has_key(self, key):
    for d in self.dicts:
      if d.has_key(key):
        return True
    return False

  def keys(self):
    keysDict = {}
    for d in self.dicts:
      for key in d.keys():
        keysDict[key] = None
    return keysDict.keys()

  def setActiveLayer(self, n):
    if n < len(self.dicts):
      self.activeLayer = n
      while n < len(self.dicts):
        del self.dicts[n]
    else:
      self.activeLayer = len(self.dicts)
    self.dicts.append({})

  def update(self, d):
    return self.dicts[self.activeLayer].update(d)
