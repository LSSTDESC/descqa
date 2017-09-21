#!/bin/bash -l
. /opt/modules/default/etc/modules.sh
cd /global/projecta/projectdirs/lsst/www/descqa/v2
./run_master.sh -m "full run" 1>>/global/projecta/projectdirs/lsst/groups/CS/descqa/run/v2/cron.log 2>&1

