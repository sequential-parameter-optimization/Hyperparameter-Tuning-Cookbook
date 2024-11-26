#!/bin/sh
rm -f dist/parksim*; python -m build; python -m pip install dist/parksim*.tar.gz
python -m mkdocs build
