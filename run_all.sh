#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/etl.py
python src/detect.py > /dev/null
python src/graph.py > /dev/null
python src/graph_feature.py > /dev/null
(cd src && python evaluate.py > /dev/null)
echo "pipeline complete; see results/"
