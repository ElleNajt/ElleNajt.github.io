#!/usr/bin/env sh

# set -x
set -euo pipefail
emacs --batch -l ./publish.el
