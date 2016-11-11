__all__ = ['Invocation', 'BigBoard']
import os
import re
import cgi
import bisect
import itertools
import cPickle as pickle

class Invocation:
    """
    encapsulates data visible for a single FlashTest invocation, (a
    date, possibly with a suffix) at the top level of FlashTestView
    """
    def __init__(self, name, dir_path):
        self.name = name
        m = re.match(r'(20\d{2}-[01]\d-[0123]\d)(?:_(\d+))?', self.name)
        assert m is not None
        self.path = os.path.join(dir_path, name)
        assert os.path.isdir(self.path)
        m = m.groups()
        self.date = m[0]
        self.sameday_index = int(m[1] or 0)
        self.keep = True
        self.html = None

    def __cmp__(self, other):
        return cmp(other.date, self.date) or cmp(other.sameday_index, self.sameday_index)

    def gen_invocation_html(self):
        tests = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and not d.startswith('_')]
        tests.sort()
        
        if os.path.isdir(os.path.join(self.path, '_group_by_catalog')):
            catalogs = [d for d in os.listdir(os.path.join(self.path, '_group_by_catalog'))]
            catalogs.sort()
        else:
            catalogs = []

        try:
            with open(os.path.join(self.path, 'errors')) as f:
                status = f.read()
        except (OSError, IOError):
            status = ''

        if status:
            light = 'yellow' if '_ERROR' in status else 'green'
        else:
            light = 'red'
            status = 'status file "errors" not found or no content!'

        status = '<br>'.join((cgi.escape(l, True) for l in status.splitlines()))

        try:
            with open(os.path.join(self.path, 'user')) as f:
                user = f.readline()
        except (OSError, IOError):
            user = ''
        else:
            user = user.strip()
            user = '({})'.format(user) if user else ''
            
        output = []
        output.append('<td>&nbsp;<a href="viewer/viewBuilds.cgi?target_dir={}" onMouseOver="appear(\'\', \'{}\');" onMouseOut="disappear();">{}</a>&nbsp;{}&nbsp;&nbsp;</td>'.format(\
                self.path, status, self.name, user))
        output.append('<td><img src="images/{}.gif">&nbsp;&nbsp;</td>'.format(light))
        test_links = '&nbsp;|&nbsp;'.join(('<a href="viewer/viewBuild.cgi?target_dir={0}/{1}">{1}</a>'.format(self.path, t) for t in tests))
        catalog_links = '&nbsp;|&nbsp;'.join(('<a href="viewer/viewBuild.cgi?target_dir={0}/_group_by_catalog/{1}">{1}</a>'.format(self.path, c) for c in catalogs))
        output.append('<td>{}<br>{}&nbsp;</td>'.format(test_links, catalog_links))

        self.html = '\n'.join(output)


class BigBoard:
    """
    Encapsulates all data related to a top-level visualization of a
    FlashTest output directory, that is, the big board with the red
    and green lights.    The path to the directory whose contents are
    thus represented is contained in the 'dir_path' member.
    """

    def __init__(self, dir_path):
        assert os.path.isdir(dir_path)
        self.dir_path = dir_path
        self.invocationList = []


    def generate(self, reset=False):
        newInvocationList = []

        for item in os.listdir(self.dir_path):
            try:
                invocation = Invocation(item, self.dir_path)
            except AssertionError:
                continue
            
            if not reset:
                i = bisect.bisect_left(self.invocationList, invocation)
                if i < len(self.invocationList) and self.invocationList[i] == invocation:
                    invocation.html = self.invocationList[i].html
            if not invocation.html:
                invocation.gen_invocation_html()

            bisect.insort(newInvocationList, invocation)
        
        self.invocationList = newInvocationList       


    def dump(self, path):
        with open(path, 'w') as f:
            pickle.dump(self.invocationList, f, pickle.HIGHEST_PROTOCOL)


    def load(self, path):
        with open(path, 'r') as f:
            self.invocationList = pickle.load(f)


    def get_html(self, skiprows=0, nrows=50):
        """
        Generate html corresponding to this big board instance
        """
        output = []
        output.append('<table class="bigboard" border="0" width="100%" cellspacing="0">')

        colored = True
        for invocation in itertools.islice(self.invocationList, skiprows, skiprows+nrows):
            output.append('<tr {} valign="top">{}</tr>'.format('style="background-color:#bbf"' if colored else '', invocation.html))
            colored = not colored

        output.append('</table>')
        return '\n'.join(output)


    def get_count(self):
        return len(self.invocationList)
