#!/usr/bin/env bash
# Fetch the real Loghub datasets used by the pipeline and verify checksums.
set -euo pipefail
cd "$(dirname "$0")/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/raw"; cd "$(dirname "$0")/raw"; }
base=https://raw.githubusercontent.com
curl -sSfLO $base/logpai/loglizer/master/data/HDFS/HDFS_100k.log_structured.csv
curl -sSfLO $base/logpai/loglizer/master/data/HDFS/anomaly_label.csv
curl -sSfLO $base/logpai/loghub/master/BGL/BGL_2k.log_structured.csv
curl -sSfLO $base/logpai/loghub/master/BGL/BGL_2k.log_templates.csv
curl -sSfLO $base/logpai/loghub/master/HDFS/HDFS_2k.log_templates.csv
sha256sum -c SHA256SUMS
