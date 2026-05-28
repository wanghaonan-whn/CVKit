#!/usr/bin/env bash
set -e

python -m build --wheel
rm -rf build *.egg-info