#!/usr/bin/env bash
# Load the graph into Neo4j and verify it matches the in-process results.
# Option A (local, needs Docker):  ./scripts/neo4j_run.sh local
# Option B (AuraDB Free):          export NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io NEO4J_PASSWORD=...; ./scripts/neo4j_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "local" ]]; then
  docker run -d --name netincident-neo4j -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/netincident-local neo4j:5-community
  export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=netincident-local
  echo "waiting for Neo4j..."; for _ in $(seq 1 60); do python -c "from neo4j import GraphDatabase as G; G.driver('$NEO4J_URI',auth=('neo4j','netincident-local')).verify_connectivity()" 2>/dev/null && break; sleep 2; done
fi
: "${NEO4J_URI:?set NEO4J_URI}" "${NEO4J_PASSWORD:?set NEO4J_PASSWORD}"
python src/neo4j_load.py
python src/neo4j_check.py
