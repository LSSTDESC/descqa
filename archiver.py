import os
import re
import time
import subprocess
import argparse

def archive(src, dst):
    if os.path.exists(dst):
        raise ValueError('{} already exists'.format(dst))
    subprocess.check_call(['tar', '-czf', dst, src])
    subprocess.check_call(['rm', '-rf', src])    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('dest_dir')
    parser.add_argument('archive_to_date', help='in the format of YYYY-MM-DD')
    args = parser.parse_args()
    
    if re.match(r'20\d{2}-[01]\d-[0123]\d', args.archive_to_date) is None:
        parser.error('`archive_to_date` must be in the format of YYYY-MM-DD')

    items = os.listdir(args.source_dir)
    for d in items:
        src = os.path.join(args.source_dir, d)
        dst = os.path.join(args.dest_dir, d + '.tar.gz')
        
        if not os.path.isdir(src):
            continue
        m = re.match(r'(20\d{2}-[01]\d-[0123]\d)(?:_\d+)?', d)
        if m is None:
            continue
        if m.groups()[0] > args.archive_to_date:
            continue

        try:
            archive(src, dst)
        except:
            print('Failed to archive {}'.format(d))
        else:
            print('{} archived'.format(d))


if __name__ == '__main__':
    main()

