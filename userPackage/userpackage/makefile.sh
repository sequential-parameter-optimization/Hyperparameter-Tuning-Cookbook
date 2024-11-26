#!/bin/sh
rm -f dist/userpackage*; python -m build; python -m pip install dist/userpackage*.tar.gz
python -m mkdocs build
