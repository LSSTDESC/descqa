#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import re
import json
import time
import os

def left_frame(selected_run,catalog,test):

    try:
        configDict = littleParser.parseFile('config')
    except:
        configDict = {}
    siteTitle = configDict.get('siteTitle', '')
    pathToOutputDir = configDict.get('pathToOutputDir', '')
    targetDir=os.path.join(pathToOutputDir,selected_run) #FIXME
    if test:
        targetDir=os.path.join(targetDir,test)
    elif catalog:
        targetDir=os.path.join(targetDir,'_group_by_catalog',catalog)

    class FlashRun:
        '''
        encapsulates one run of the Flash code against
        a single parfile. A list of FlashRun objects
        will form part of the data dictionary passed to
        the ezt template
        '''
        def __init__(self, name):
            self.name = name

    class File:
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


    # check target_dir
    form = cgi.FieldStorage()
    # targetDir = form.getfirst('run')
    if not os.path.isabs(targetDir):
        raise ValueError('`target_dir` is not correctly set')
    test_prefix=form.getfirst('test_prefix','')
    catalog_prefix=form.getfirst('catalog_prefix','')

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
    filepath = os.path.join(targetDir, 'summary_plot.png')
    templateData['summaryPlot'] = File(filepath) if os.path.isfile(filepath) else None
    filepath = os.path.join(targetDir, 'summary_plot.log')
    templateData['summaryPlotLog'] = File(filepath) if os.path.isfile(filepath) else None


    # we assume any directories in 'targetDir' to be the output
    # of a single *run* of Flash (i.e., the output resulting from
    # the Flash executable's being run against a single parfile)
    # Information in this directory will be stored in a FlashRun
    # object (see class definition at top of file)
    runs = [FlashRun(item) for item in sorted(os.listdir(targetDir)) \
            if os.path.isdir(os.path.join(targetDir, item))]

    for run in runs:
        run.fullPath = os.path.join(targetDir, run.name)
        run.datfiles = []            
        run.logfiles = []             
        run.imgfiles = []
        items = sorted(os.listdir(run.fullPath))
        for item in items:
            item_lower = item.lower()
            if item_lower.endswith('.log'):
                run.logfiles.append(File(run.fullPath, item))
            elif any(item_lower.endswith(ext) for ext in ('.txt', '.dat', '.csv')):
                run.datfiles.append(File(run.fullPath, item))
            elif item_lower.endswith('.png'):
                run.imgfiles.append(File(run.fullPath, item))

        try:
            with open(os.path.join(run.fullPath, 'STATUS')) as f:
                run.status = f.readline().strip().upper()
                run.summary = f.readline().strip()
        except (OSError, IOError):
            import traceback
            print '<pre>'
            traceback.print_exc()
            print '</pre>'
            run.status = 'NO_STATUS_FILE_ERROR'
            run.summary = ''

        for status, color in (('PASSED', 'green'), ('SKIPPED', 'gold'), ('FAILED', 'orangered'), ('ERROR', 'darkred')):
            if run.status.endswith(status):
                run.statusColor = color
                break

    templateData['runs'] = runs or None

    template = env.get_template('leftFrame.html')
    print template.render(runs=runs,templateData=templateData,selected_run=selected_run,test_prefix=test_prefix,
        catalog_prefix=catalog_prefix,test=test,catalog=catalog)

def view_run(run,catalog=None,test=None):

    targetDir=run
    targetItem=catalog

    # load config
    try:
        configDict = littleParser.parseFile('config')
    except:
        configDict = {}

    form = cgi.FieldStorage()     
    test_prefix = form.getfirst('test_prefix', '')
    catalog_prefix = form.getfirst('catalog_prefix', '')

    siteTitle = configDict.get('siteTitle', '')
    pathToOutputDir = configDict.get('pathToOutputDir', '')
    targetDir=os.path.join(pathToOutputDir,run)
    template = env.get_template('runplots.html')
    print template.render(catalog=catalog,test=test,targetDir=targetDir,run=run,test_prefix=test_prefix,catalog_prefix=catalog_prefix)

def run_summary(run):
    color_dict = {'PASSED': 'green', 'SKIPPED': 'gold', 'FAILED': 'orangered', 'ERROR': 'darkred'}

    class TestDir(object):
        def __init__(self, name, parent_dir,test_prefix,catalog_prefix):
            assert not name.startswith('_')
            self.name = name
            self.path = os.path.join(parent_dir, name)
            self.test_prefix = test_prefix
            self.catalog_prefix = catalog_prefix
            assert os.path.isdir(self.path)
            self.prefix = name.partition('_')[0]

        def __cmp__(self, other):
            return cmp(self.name, other.name)


    class TestMember(TestDir):
        html = None
        def __init__(self, name, parent_dir, test_prefix, catalog_prefix, groupname):
            super(TestMember,self).__init__(name,parent_dir, test_prefix, catalog_prefix)
            self.groupname = groupname


        def get_html(self):
            if self.html is None:
                try:
                    with open(os.path.join(self.path, 'STATUS')) as f:
                        lines = f.readlines()
                    status = lines[0].strip()
                except (OSError, IOError, IndexError):
                    status = 'NO_STATUS_FILE_ERROR'
                    lines = []
                
                try:
                    score = '<br>' + cgi.escape(lines[2].strip())
                except IndexError:
                    score = ''
                
                status = status.rpartition('_')[-1]
                color = color_dict.get(status, 'darkred')
                
                self.html = '<td class="{}"><a class="celllink" href="index.cgi?run={}&test={}&catalog={}">{}{}</a></td>'.format(\
                        color, os.path.basename(os.path.dirname(os.path.dirname(self.path))), self.groupname, self.name, cgi.escape(status), score)
            
            return self.html


    class TestGroup(TestDir):
        members = None

        def get_members(self):
            if self.members is None:
                self.members = {}

                for name in os.listdir(self.path):
                    try:
                        member = TestMember(name, self.path, self.test_prefix, self.catalog_prefix, self.name )
                    except AssertionError:
                        continue
                    self.members[name] = member
            return self.members

        def get_html(self, sorted_member_names, target_dir_base=None):
            # target_dir = self.path if target_dir_base is None else os.path.join(target_dir_base, self.name)
            target_dir = self.path if target_dir_base is None else target_dir_base
            members = self.get_members()
            # html = ['<td><a href="viewer/viewBuild.cgi?target_dir={}">{}</a></td>'.format(target_dir, self.name)]
            html = ['<td><a href="index.cgi?run={0}&test={1}&test_prefix={2}&catalog_prefix={3}">{1}</a></td>'.format(target_dir_base, self.name, test_prefix,catalog_prefix)]
            for member_name in sorted_member_names:
                member = members.get(member_name)
                html.append(member.get_html() if member else '<td>&nbsp;</td>')
            return '<tr>{}</tr>'.format(''.join(html))


    def get_filter_link(targetDir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
        text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
        if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
            return '<span style="color:gray">{}</span>'.format(text)
        # return '<a href="viewer/viewBuilds.cgi?target_dir={}&test_prefix={}&catalog_prefix={}">{}</a>'.format(targetDir, new_test_prefix, new_catalog_prefix, text)
        return '<a href="index.cgi?run={}&test_prefix={}&catalog_prefix={}">{}</a>'.format(targetDir, new_test_prefix, new_catalog_prefix, text)

    # load config
    try:
        configDict = littleParser.parseFile('config')
    except:
        configDict = {}

    siteTitle = configDict.get('siteTitle', '')
    pathToOutputDir = configDict.get('pathToOutputDir', '')
    if not os.path.isabs(pathToOutputDir):
        raise ValueError('`pathToOutputDir` in `config` should be an absolute path')

    # check target_dir
    form = cgi.FieldStorage()     
    if run:
        targetDir_base = run
        targetDir = os.path.abspath(os.path.join(pathToOutputDir, run))

    test_prefix = form.getfirst('test_prefix', '')
    catalog_prefix = form.getfirst('catalog_prefix', '')

    all_groups = []
    for name in os.listdir(targetDir):
        try:
            group = TestGroup(name, targetDir, test_prefix, catalog_prefix)
        except AssertionError:
            continue
        all_groups.append(group)
    all_groups.sort()

    catalog_list= set()
    test_prefix_union = set()
    catalog_prefix_union = set()

    for group in all_groups:
        test_prefix_union.add(group.prefix)
        for member in group.get_members().itervalues():
            catalog_prefix_union.add(member.prefix)
            if not catalog_prefix or member.prefix == catalog_prefix:
                catalog_list.add(member.name)

    catalog_list = sorted(catalog_list)
    test_prefix_union = sorted(test_prefix_union)
    catalog_prefix_union = sorted(catalog_prefix_union)

    try:
        with open(os.path.join(targetDir, 'STATUS.json')) as f:
            master_status = json.load(f)
    except:
        master_status = {}


    try:
        configDict = littleParser.parseFile('config')
        siteTitle = configDict.get('siteTitle', '')
    except:
        siteTitle = ''

    comment=None
    user=None
    start_time=None
    time_used=None

    if master_status:
        comment =  master_status.get('comment', '')
        user = master_status.get('user', 'UNKNOWN')
        if 'start_time' in master_status:
            start_time = time.strftime('at %Y/%m/%d %H:%M:%S PT', time.localtime(master_status.get('start_time')))
        time_used = (master_status.get('end_time', -1.0) - master_status.get('start_time', 0.0))/60.0


    test_prefix_union.insert(0, '')
    links = '&nbsp;|&nbsp;'.join((get_filter_link(run, True, p, catalog_prefix, test_prefix, catalog_prefix) for p in test_prefix_union))
    test_links_str = '[&nbsp;Test prefix: {}&nbsp;]<br>'.format(links)

    catalog_prefix_union.insert(0, '')
    links = '&nbsp;|&nbsp;'.join((get_filter_link(run, False, test_prefix, p, test_prefix, catalog_prefix) for p in catalog_prefix_union))
    catalog_links_str = '[&nbsp;Catalog prefix: {}&nbsp;]'.format(links)


    table_width = (len(catalog_list) + 1)*130
    if table_width > 1280:
        table_width = "100%"
    else:
        table_width = "{}px".format(table_width)

    # header_row = ['<td><a href="viewer/viewBuild.cgi?target_dir={1}/{0}">{0}</a></td>'.format(name, os.path.join(targetDir_base, '_group_by_catalog')) for name in catalog_list]
    header_row = ['<td><a href="index.cgi?run={1}&catalog={0}&test_prefix={2}&catalog_prefix={3}">{0}</a></td>'.format(name, targetDir_base, test_prefix, catalog_prefix) for name in catalog_list]
    header_row_str = '<tr><td>&nbsp;</td>{}</tr>'.format('\n'.join(header_row))

    template = env.get_template('run.html')
    print template.render(comment=comment,user=user,start_time=start_time,time_used=time_used,
        test_links=test_links_str,catalog_links=catalog_links_str,table_width=table_width,
        header_row=header_row_str,all_groups=all_groups, catalog_list=catalog_list,
        targetDir=targetDir_base,test_prefix=test_prefix,catalog_prefix=catalog_prefix)


def all_runs(page):
    configDict = littleParser.parseFile('config')
    pathToOutputDir = configDict['pathToOutputDir'] # must have
    invocationsPerPage = int(configDict.get('invocationsPerPage', 25))
    siteTitle = configDict.get('siteTitle', '')
    days_to_show = int(configDict.get('days_to_show', 15))

    bigboard_cache = configDict.get('bigboard_cache')
    try:
        bigboard_cache += '.new'
    except TypeError:
        pass

    bigboard = BigBoard(pathToOutputDir, bigboard_cache)
    cache_dumped = bigboard.generate(days_to_show, bigboard_cache, new_style=True)

    if cache_dumped:
        try:
            os.chmod(bigboard_cache, stat.S_IWOTH+stat.S_IROTH+stat.S_IWGRP+stat.S_IRGRP+stat.S_IRUSR+stat.S_IWUSR)
        except OSError:
            pass

    count = bigboard.get_count()
    if not count:
        print '<h1>nothing to show!</h1>'
        print '</body></html>'
        sys.exit(0)

    npages = ((count - 1) // invocationsPerPage) + 1
    if page > npages:
        page = npages

    template = env.get_template('runtable.html')
    html=bigboard.get_html(invocationsPerPage*(page-1), invocationsPerPage)
    print template.render(html=html,page=page,npages=npages)

def details():
    import config_validation as vc
    import config_catalog as cc
    vcs=[(k,getattr(vc,k)) for k in vc.__dict__.keys() if isinstance(getattr(vc,k),vc._ValidationConfig)]
    ccs=[(k,getattr(cc,k)) for k in cc.__dict__.keys() if isinstance(getattr(cc,k),cc._CatalogConfig)]
    template = env.get_template('details.html')
    print template.render(validation_configs=vcs,catalog_configs=ccs)

def find_last_run():
    try:
        configDict = littleParser.parseFile('config')
    except:
        configDict = {}

    siteTitle = configDict.get('siteTitle', '')
    pathToOutputDir = configDict.get('pathToOutputDir', '')
    if not os.path.isabs(pathToOutputDir):
        raise ValueEror('`pathToOutputDir` in `config` should be an absolute path')

    names = os.listdir(pathToOutputDir)
    names.sort(reverse=True)
    targetDir_base = names[0]
    targetDir = os.path.abspath(os.path.join(pathToOutputDir, names[0]))
    for name in names:
        statusfile=os.path.join(pathToOutputDir,name,'STATUS.json')
        if os.path.exists(statusfile):
            status = json.load(file(statusfile))
            if status.get('comment') == 'full run':
                targetDir_base = name
                targetDir = os.path.abspath(os.path.join(pathToOutputDir, name))
                break

    return targetDir_base

try:
    import os
    import sys
    import stat
    from utils.invocations_simple import Invocation, BigBoard
    from utils import littleParser

    sys.path.insert(0, '../../lib/python-cgi')
    from jinja2 import Template
    from jinja2 import Environment, PackageLoader#, select_autoescape
    env = Environment(
        loader=PackageLoader('descqa', 'templates'),
            #autoescape=select_autoescape(['html', 'xml'])
            )


    form = cgi.FieldStorage()
    if 'details' in form.keys():
        details()
    elif 'header' in form.keys():
        template = env.get_template('header.html')
        print template.render()
    else:
      run = form.getfirst('run')
      if run:
        catalog = form.getfirst('catalog')
        test = form.getfirst('test')
        if 'leftframe' in form:
            left_frame(run, catalog,test)
        elif run == 'all':
            try:
                page = int(form.getfirst('page', 1))
            except:
                page = 1
            all_runs(page)
        else:
            if catalog or test:
                view_run(run,catalog,test)
            else:
                run_summary(run)
      else:
        # no run given; show last full run
        run = find_last_run()
        run_summary(run)

except Exception,e:
    import traceback
    print '<pre>'
    print 'exception',e
    traceback.print_exc(file=sys.stdout)
    print '</pre>'
