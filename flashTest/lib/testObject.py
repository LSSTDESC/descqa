import sys
import strategyTemplates

class TestObject:
  """
  Represents a single object whose public methods define behavior for all
  phases of a FlashTest run. TestObject is never sub-classed. Instead, each
  instance is assembled out of components which are themselves instances of
  user-defined strategy classes specified by the user in the "config" file
  and in "test.info" files.

  For example, the user may specify in the "config" file that all calls to
  "setup()" should be calls to the setup method of the user-defined class
  "MySetupper", that calls to "test()" for problems of type "Comparison"
  should be calls to the test method of the user-defined class "MyTester",
  and that calls to "test()" for problems of type "UnitTestUG" should be
  calls to the test method of the user-defined class "MyTester2".

  Here is the relevant snippet of our example "config" file:

    setupper: MySetupper

    [Comparison]
    tester: MyTester

    [UnitTestUG]
    tester: MyTester2

  However, the user may override the values in "config" with values from a
  "test.info" file. e.g., if a subset of tests of type "UnitTestUG" require
  the test method of the user-defined class MyTester3, the file:

  flashTest/infoFiles/<mySite>/UnitTestUG/<someSubset>/test.info

  can define:

    tester: MyTester3


  Components not specified by the user in "config" or "test.info" files are
  instantiated from templates defined in "strategyTemplates.py".

  Components which are specified by the user should subclass these same
  templates.

  TestObject instances define a component "entryPoint" on initialization. If
  the user wishes to specify a custom entry-point class, he or she must do
  so in the "config" file. This single component can not be overridden in a
  "test.info" file.

  After initialization, flashTest.py will call the method installComponent()
  for the setupper, compiler, executer, and tester components.
  """
  def __init__(self, masterDict):
    self.entryPoint = None
    self.setupper   = None
    self.compiler   = None
    self.executer   = None
    self.tester     = None
    self.masterDict = masterDict

    # give this object an entryPoint object
    self.installComponent("entryPoint")


  def installComponent(self, component):
    """
    called first by TestObject's __init__() method and later by flashTest.py to
    instantiate a user-defined strategy class or, if this is not specified, of
    a default strategy template.
    """
    if self.masterDict.has_key(component):
      className = self.masterDict[component]
      # 'className' is the name of the strategy class that the user
      # wants to instantiate as the "setupper", "compiler", etc, but
      # we still need to know what module it's in...
        
      # Does 'className' specify its module by the form "myModule.MySetupper"?
      if className.find(".") > 0:
        modName, className = className.split(".",1)
      # No? Then has the user specified a "useModule" key?
      elif self.masterDict.has_key("useModule"):
        modName = self.masterDict["useModule"]
      # No? Then we don't know what module to look in.
      else:
        errMsg = ("You must indicate a module where class \"%s\" can be found.\n" % className +
                  "Add a \"useModule\" key to \"config\" or indicate the module\n" +
                  "with \"modName.%s\"\n" % className +
                  "Modules should be located in the FlashTest \"lib\" directory.")
        raise Exception(errMsg)
      # At this point, 'modName' has been defined, so we know where to look for 'className'
      if modName not in sys.modules:
        __import__(modName)

    else:
      modName = "strategyTemplates"
      # Name of default setupper class is "SetupperTemplate",
      # default compiler is "CompilerTemplate", etc.
      className = "%sTemplate" % (component[0].upper() + component[1:])


    # Check to see if the requested component from
    # the requested module is already in place
    if (not hasattr(self, component) or
        getattr(self, component).__class__.__name__ != className or
        getattr(self, component).__module__ != modName):

      # intantiate 'className' and make 'instance' an attribute of 'self'
      if hasattr(sys.modules[modName], className):
        instance = getattr(sys.modules[modName], className)(self)
        setattr(self, component, instance)
      else:
        errMsg = ("Module \"%s\" defines no class called \"%s\"\n" % (modName, className))
        raise Exception(errMsg)


  # These methods are called by flashTest.py and invoke a method
  # or methods on the instance of the strategy class specified in
  # "config" and instantiated in TestObject.__init__ (above)
  def entryPoint1(self):
    return self.entryPoint.entryPoint1()

  def entryPoint2(self):
    return self.entryPoint.entryPoint2()

  def entryPoint3(self):
    return self.entryPoint.entryPoint3()

  def setup(self):
    return self.setupper.setup()

  def compile(self):
    result = self.compiler.compile()
    self.compiler.writeFilesToDelete()
    return result

  def execute(self):
    result = self.executer.execute()
    self.executer.writeFilesToDelete()
    return result

  def test(self):
    self.tester.openOutfile()
    result = self.tester.test()
    self.tester.closeOutfile()
    return result
