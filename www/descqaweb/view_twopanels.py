from __future__ import unicode_literals, print_function
import os
from .config import *

__all__ = ['render', 'render_left', 'print_file']


class FlashRun(object):
    '''
    encapsulates one run of the Flash code against
    a single parfile. A list of FlashRun objects
    will form part of the data dictionary passed to
    the ezt template
    '''
    def __init__(self, name):
        self.name = name


class File(object):
    '''
    encapsulates the data needed to locate any files
    '''
    def __init__(self, path_or_dir, basename=None):
        if basename is None:
            self.path = path_or_dir
            self.filename = os.path.basename(path_or_dir)
        else:
            self.path = os.path.join(path_or_dir, basename)
            self.filename = basename
        if self.filename.lower().endswith('.png'):
            self.data = open(self.path, 'rb').read().encode('base64').replace('\n', '')


_ALLOWED_EXT = ('.txt', '.dat', '.csv', '.log', '.yaml', '.pdf', '.png')


def gather_runs(targetDir):

    # we assume any directories in 'targetDir' to be the output
    # of a single *run* of Flash (i.e., the output resulting from
    # the Flash executable's being run against a single parfile)
    # Information in this directory will be stored in a FlashRun
    # object (see class definition at top of file)
    runs = [FlashRun(item) for item in sorted(os.listdir(targetDir)) \
            if os.path.isdir(os.path.join(targetDir, item))]

    for run in runs:
        run.fullPath = os.path.join(targetDir, run.name)
        run.outfiles = []
        items = sorted(os.listdir(run.fullPath))
        for item in items:
            item_lower = item.lower()
            if any(item_lower.endswith(ext) for ext in _ALLOWED_EXT):
                run.outfiles.append(File(run.fullPath, item))

        try:
            with open(os.path.join(run.fullPath, 'STATUS')) as f:
                run.status = f.readline().strip().upper()
                run.summary = f.readline().strip()
        except (OSError, IOError):
            run.status = 'NO_STATUS_FILE_ERROR'
            run.summary = ''

        for status, color in (('PASSED', 'green'), ('SKIPPED', 'gold'), ('FAILED', 'orangered'), ('ERROR', 'darkred')):
            if run.status.endswith(status):
                run.statusColor = color
                break

    return runs


def render_left(template, selected_run, catalog, test, catalog_prefix, test_prefix):

    targetDir = os.path.join(pathToOutputDir, selected_run)
    if not os.path.isabs(targetDir):
        raise ValueError('`target_dir` is not correctly set')

    if test:
        targetDir = os.path.join(targetDir, test)
    elif catalog:
        targetDir = os.path.join(targetDir, '_group_by_catalog', catalog)

    # the data dictionary we will pass to the ezt template
    templateData = {}

    # fill in data that has to do with this build
    # (i.e. setup and compilation data)
    templateData['fullBuildPath']       = targetDir
    templateData['pathToInvocationDir'] = os.path.dirname(targetDir)
    templateData['buildDir']            = os.path.basename(targetDir)
    templateData['invocationDir']       = os.path.basename(os.path.dirname(targetDir))
    templateData['test_prefix']         = test_prefix
    templateData['catalog_prefix']      = catalog_prefix


    # YYM: hack to get _group_by_catalog work
    templateData['isGroupByCatalog'] = None
    GROUP_BY_CATALOG_DIRNAME = '_group_by_catalog'
    if templateData['invocationDir'] == GROUP_BY_CATALOG_DIRNAME:
        templateData['isGroupByCatalog'] = True
        templateData['invocationDir'] = os.path.basename(os.path.dirname(templateData['pathToInvocationDir']))
        templateData['pathToInvocationDir'] = os.path.dirname(templateData['pathToInvocationDir'])

    # make url look nicer
    templateData['pathToInvocationDir'] = templateData['invocationDir']

    # search for summary plot:
    templateData['summaryData'] = []
    items = [item for item in sorted(os.listdir(targetDir)) if os.path.isfile(os.path.join(targetDir, item))]
    for item in items:
        item_lower = item.lower()
        if any(item_lower.endswith(ext) for ext in _ALLOWED_EXT):
            templateData['summaryData'].append(File(targetDir, item))


    templateData['runs'] = gather_runs(targetDir) or None

    return template.render(runs=templateData['runs'],
        templateData=templateData,
        selected_run=selected_run,
        test_prefix=test_prefix,
        catalog_prefix=catalog_prefix,
        test=test,
        catalog=catalog)


def render(template, run, catalog=None, test=None, test_prefix=None, catalog_prefix=None):
    return template.render(catalog=catalog, test=test, targetDir=run, run=run, test_prefix=test_prefix, catalog_prefix=catalog_prefix)


def print_file(targetFile):
    try:
        with open(targetFile, 'rb') as f:
            file_content = f.read()

    except (OSError, IOError):
        print('Content-type: text/plain')
        print()
        print('[Error] Cannot open/read file {}'.format(targetFile))

    else:
        if targetFile.lower().endswith('.png'):
            print('Content-type: text/html')
            print()
            print('<!DOCTYPE html>')
            print('<html><body>')
            print('<img src="data:image/png;base64,{}" width="100%">'.format(file_content.encode('base64').replace('\n', '')))
            print('</body></html>')
        elif targetFile.lower().endswith('.pdf'):
            print('Content-type: application/pdf')
            print('Content-Length: {}'.format(len(file_content)))
            print('Content-Disposition: inline; filename="{}"'.format(os.path.basename(targetFile)))
            print()
            print(file_content)
        else:
            print('Content-type: text/plain')
            print('Content-Length: {}'.format(len(file_content)))
            print()
            print(file_content)
