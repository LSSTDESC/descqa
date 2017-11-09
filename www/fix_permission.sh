#!/bin/bash
set -e

python -c "import descqaweb"

mkdir -p cache
chmod o+r    ../index.html .htaccess
chmod o+rx   *.cgi
chmod o+rwx  cache
chmod o+rx   descqaweb descqaweb/templates descqaweb/view_bigboard style . ..
chmod -R o+r descqaweb style
