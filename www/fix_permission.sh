#!/bin/bash
set -e

python -c "import descqaweb"

chmod o+rx descqaweb descqaweb/templates style index.cgi . ..
chmod -R o+r .

