from __future__ import unicode_literals
import os
import re
import json
import datetime
import base64

__all__ = ['encode_png', 'get_all_runs', 'DescqaRun']

ALLOWED_EXT = {'txt', 'dat', 'csv', 'log', 'json', 'yaml', 'pdf', 'png'}
STATUS_COLORS = {'PASSED': 'green', 'SKIPPED': 'gold', 'FAILED': 'orangered', 'ERROR': 'darkred'}


def encode_png(png_content):
    return base64.b64encode(png_content).decode('ascii').replace('\n', '')


class File(object):
    '''
    encapsulates the data needed to locate any files
    '''
    def __init__(self, filename, dir_path=None, rel_dir_path=None):
        if dir_path is None:
            self.path = filename
            self.filename = os.path.basename(filename)
        else:
            self.path = os.path.join(dir_path, filename)
            self.filename = filename
        if rel_dir_path is not None:
            self.relpath = os.path.join(rel_dir_path, filename)
        self._data = None
        self.is_png = self.filename.lower().endswith('.png')

    @property
    def data(self):
        if self.is_png and self._data is None:
            self._data = encode_png(open(self.path, 'rb').read())
        return self._data


class DescqaItem(object):
    def __init__(self, test, catalog, run, base_dir):
        if catalog is None:
            self.relpath = os.path.join(run, test)
            self.is_test_summary = True
        else:
            self.relpath = os.path.join(run, test, catalog)
            self.is_test_summary = False
        self.path = os.path.join(base_dir, self.relpath)
        self.test = test
        self.catalog = catalog
        self.run = run
        self.name = ''
        self._status = None
        self._summary = None
        self._score = None
        self._status_color = None
        self._files = None

    def _parse_status(self):
        if self.is_test_summary:
            return

        try:
            with open(os.path.join(self.path, 'STATUS')) as f:
                lines = f.readlines()
        except (OSError, IOError):
            lines = []

        while len(lines) < 3:
            lines.append('')

        self._status = lines[0].strip().upper() or 'NO_STATUS_FILE_ERROR'
        self._summary = lines[1].strip()
        self._score = lines[2].strip()

        for status, color in STATUS_COLORS.items():
            if self._status.endswith(status):
                self._status_color = color
                break
        else:
            self._status_color = 'darkred'

    @property
    def status(self):
        if self._status is None:
            self._parse_status()
        return self._status

    @property
    def summary(self):
        if self._summary is None:
            self._parse_status()
        return self._summary

    @property
    def score(self):
        if self._score is None:
            self._parse_status()
        return self._score

    @property
    def status_color(self):
        if self._status_color is None:
            self._parse_status()
        return self._status_color

    def _get_files(self):
        files = []
        for item in sorted((f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))):
            if item.rpartition('.')[-1].lower() in ALLOWED_EXT:
                files.append(File(item, self.path, self.relpath))
        return tuple(files)

    @property
    def files(self):
        if self._files is None:
            self._files = self._get_files()
        return self._files


class DescqaRun(object):
    def __init__(self, run_name, base_dir):
        self.path = os.path.join(base_dir, run_name)
        self.base_dir = base_dir
        assert os.path.isdir(self.path)
        assert not os.path.exists(os.path.join(self.path, '.lock'))

        self.name = run_name
        m = re.match(r'(20\d{2}-[01]\d-[0123]\d)(?:_(\d+))?', self.name)
        assert m is not None
        m = m.groups()
        self.sort_key = datetime.datetime(*(int(i) for i in m[0].split('-')), microsecond=int(m[1] or 0))

        self._tests = None
        self._catalogs = None
        self._test_prefixes = None
        self._catalog_prefixes = None
        self._status = None
        self._data = dict()

    @staticmethod
    def _find_subdirs(path):
        return tuple(sorted((d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('_'))))

    def _find_tests(self):
        return self._find_subdirs(self.path)

    def _find_catalogs(self):
        return self._find_subdirs(os.path.join(self.path, self.tests[0])) if len(self.tests) else tuple()

    @staticmethod
    def _find_prefixes(items):
        prefixes = set()
        for item in items:
            prefixes.add(item.partition('_')[0])
        return tuple(sorted(prefixes))

    @property
    def tests(self):
        if self._tests is None:
            self._tests = self._find_tests()
        return self._tests

    @property
    def catalogs(self):
        if self._catalogs is None:
            self._catalogs = self._find_catalogs()
        return self._catalogs

    @property
    def test_prefixes(self):
        if self._test_prefixes is None:
            self._test_prefixes = self._find_prefixes(self.tests)
        return self._test_prefixes

    @property
    def catalog_prefixes(self):
        if self._catalog_prefixes is None:
            self._catalog_prefixes = self._find_prefixes(self.catalogs)
        return self._catalog_prefixes

    @staticmethod
    def _get_things(things, prefix=None, return_iter=False):
        it = (t for t in things if prefix is None or t.startswith(prefix))
        return it if return_iter else tuple(it)

    def get_tests(self, prefix=None, return_iter=False):
        return self._get_things(self.tests, prefix, return_iter)

    def get_catalogs(self, prefix=None, return_iter=False):
        return self._get_things(self.catalogs, prefix, return_iter)

    def __getitem__(self, key):
        if key not in self._data:
            try:
                test, catalog = key
            except ValueError:
                test = key
                catalog = None
            if test in self.tests and (catalog in self.catalogs or catalog is None):
                self._data[key] = DescqaItem(test, catalog, self.name, self.base_dir)
            else:
                raise KeyError('(test, catalog) = {} does not exist'.format(key))

        return self._data[key]

    @property
    def status(self):
        if self._status is None:
            try:
                with open(os.path.join(self.path, 'STATUS.json')) as f:
                    self._status = json.load(f)
            except (IOError, OSError):
                self._status = dict()
        return self._status


def get_all_runs(base_dir, run_filter=None):
    all_runs = list()
    for run_name in os.listdir(base_dir):
        try:
            run = DescqaRun(run_name, base_dir)
        except AssertionError:
            continue
        if run_filter is None or run_filter(run):
            all_runs.append(run)
    all_runs.sort(key=lambda r: r.sort_key, reverse=True)
    return all_runs
