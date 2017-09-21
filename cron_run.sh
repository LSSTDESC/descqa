#!/bin/bash -l
. /opt/modules/default/etc/modules.sh
DESCQA=/global/projecta/projectdirs/lsst/groups/CS/descqa
cd ${DESCQA}/src/v2
./run_master.sh -m "full run" 1>>${DESCQA}/run/cron.log 2>&1
