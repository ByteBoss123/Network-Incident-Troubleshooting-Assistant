#!/usr/bin/env bash
# Load the graph into Neo4j and verify it matches the in-process results.
# Option A (local, needs Docker):  ./scripts/neo4j_run.sh local
# Option B (AuraDB Free):          export NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io NEO4J_PASSWORD=...; ./scripts/neo4j_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "local" ]]; then
  # throwaway container: a random password per run, bound to localhost only, never stored in the repo
  export NEO4J_USER=neo4j NEO4J_PASSWORD="$(openssl rand -hex 16)" NEO4J_URI=bolt://localhost:7687
  docker run -d --name netincident-neo4j -p 127.0.0.1:7687:7687 -p 127.0.0.1:7474:7474 \
    -e NEO4J_AUTH="$NEO4J_USER/$NEO4J_PASSWORD" neo4j:5-community
  echo "waiting for Neo4j..."
  for _ in $(seq 1 60); do
    python -c "import os; from neo4j import GraphDatabase as G; G.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])).verify_connectivity()" 2>/dev/null && break
    sleep 2
  done
fi
: "${NEO4J_URI:?set NEO4J_URI}" "${NEO4J_PASSWORD:?set NEO4J_PASSWORD}"
python src/neo4j_load.py
python src/neo4j_check.py
