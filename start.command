#!/bin/zsh
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Set up the Python environment using README.md, then run this file again.'
  read '?Press Return to close.'
  exit 1
fi
exec .venv/bin/python app.py
