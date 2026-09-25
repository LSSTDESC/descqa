from __future__ import unicode_literals

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

from descqaweb.twopanels import print_file


class PrintFileTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_print_file_allows_files_under_root(self):
        root_dir = self._mkdir('results')
        self._write('results/ok.txt', 'ok')

        output = self._print_file('ok.txt', root_dir)

        self.assertIn('Content-Type: text/plain; charset=utf-8', output)
        self.assertIn('ok', output)

    def test_print_file_rejects_sibling_directory_escape(self):
        root_dir = self._mkdir('results')
        self._mkdir('results-old')
        self._write('results-old/secret.txt', 'secret')

        output = self._print_file('../results-old/secret.txt', root_dir)

        self.assertIn('[Error] Cannot open/read file', output)
        self.assertNotIn('\nsecret\n', output)

    def test_print_file_rejects_symlink_escape(self):
        root_dir = self._mkdir('results')
        outside_dir = self._mkdir('outside')
        self._write('outside/secret.txt', 'secret')
        self._symlink(outside_dir, 'results/link')

        output = self._print_file('link/secret.txt', root_dir)

        self.assertIn('[Error] Cannot open/read file', output)
        self.assertNotIn('\nsecret\n', output)

    def _mkdir(self, relative_path):
        path = self._path(relative_path)
        os.makedirs(path)
        return path

    def _path(self, relative_path):
        return os.path.join(self.tmpdir, relative_path)

    def _print_file(self, target_file, root_dir):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            print_file(target_file, root_dir=root_dir)
        return stdout.getvalue()

    def _symlink(self, target, relative_path):
        os.symlink(target, self._path(relative_path))

    def _write(self, relative_path, content):
        with open(self._path(relative_path), 'w') as f:
            f.write(content)


if __name__ == '__main__':
    unittest.main()
