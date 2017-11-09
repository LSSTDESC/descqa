from __future__ import unicode_literals
import os
import bisect
import itertools
import pickle

from .invocation import Invocation

__all__ = ['BigBoard']

class BigBoard:
    """
    Encapsulates all data related to a top-level visualization of a
    FlashTest output directory, that is, the big board with the red
    and green lights.    The path to the directory whose contents are
    thus represented is contained in the 'dir_path' member.
    """

    def __init__(self, dir_path, cache_file=None):
        assert os.path.isdir(dir_path)
        self.dir_path = dir_path
        self.invocationList = []
        if cache_file:
            loaded = self.load(cache_file)
            if not loaded:
                self.invocationList = []


    def generate(self, days_to_show=None, cache_file=None, new_style=False):
        newInvocationList = []

        for item in os.listdir(self.dir_path):
            try:
                invocation = Invocation(item, self.dir_path, days_to_show)
            except AssertionError:
                continue

            # assume self.invocationList is sorted
            i = bisect.bisect_left(self.invocationList, invocation)
            if i < len(self.invocationList) and self.invocationList[i] == invocation:
                invocation.html = self.invocationList[i].html

            if not invocation.html:
                invocation.gen_invocation_html(new_style)

            newInvocationList.append(invocation)

        self.invocationList = sorted(newInvocationList)

        if cache_file:
            return self.dump(cache_file)


    def dump(self, path):
        try:
            with open(path, 'w') as f:
                pickle.dump(self.invocationList, f, pickle.HIGHEST_PROTOCOL)
        except (IOError, OSError):
            return False
        else:
            return True


    def load(self, path):
        try:
            with open(path, 'r') as f:
                self.invocationList = pickle.load(f)
        except (IOError, OSError):
            return False
        else:
            return True


    def get_html(self, skiprows=0, nrows=50):
        """
        Generate html corresponding to this big board instance
        """
        output = []
        output.append('<table class="bigboard" border="0" width="100%" cellspacing="0">')

        for invocation in itertools.islice(self.invocationList, skiprows, skiprows+nrows):
            output.append('<tr>{}</tr>'.format(invocation.html))

        output.append('</table>')
        return '\n'.join(output)


    def get_count(self):
        return len(self.invocationList)
