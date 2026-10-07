# Network Incident Troubleshooting Assistant

Log anomaly detection, a telemetry-derived topology graph, and a LangChain retrieval assistant for incident triage, built on real distributed-system logs.

## Data (all real, public)

| Source | What | Size |
|---|---|---|
| Loghub HDFS v1 (logpai/loglizer mirror) | Hadoop DataNode/NameNode logs, 26-minute excerpt (2008-11-09 20:35–21:01), parsed into 19 event templates | 104,815 lines, 7,940 blocks, 203 hosts |
| HDFS v1 `anomaly_label.csv` | Expert block-level labels from the full 11M-line run | 313 anomalous blocks in the excerpt (3.94%) |
| Loghub BGL (Blue Gene/L supercomputer) | RAS log sample with alert labels; node ids parsed into rack / midplane / node hierarchy | 2,000 lines, 143 alert lines, 64 racks, 1,777 nodes |

SHA-256 checksums of the downloaded files are in `data/raw/SHA256SUMS`.

## Pipeline

1. **ETL** (`src/etl.py`): Python + DuckDB SQL. Builds tables for events, labelled block sessions, block-by-template counts, replica placement and the BGL hierarchy. Checks: 0 duplicate block ids, 0 event lines without a labelled block.
2. **Anomaly detection** (`src/detect.py`): block-level event-count vectors with a **chronological** 70/30 split (5,558 train / 2,382 test blocks, 121 test anomalies).
3. **Topology graph** (`src/graph.py`, `src/neo4j_load.py`):
   - Host–host co-replica edges (13,985), re-replication transfers (27), and subnets.
   - Block→host `STORED_ON` / `RECEIVED_BY` edges, and the BGL Rack→Midplane→Node hierarchy.
   - Exported as CSV with a Neo4j loader plus Cypher queries (hotspot hosts, blast radius, rack alerts, shortest path). The same analysis is reproduced in DuckDB/NetworkX.
4. **Graph feature** (`src/graph_feature.py`): a leakage-free replica-host risk score. Host anomaly rates come from train-split blocks only, and the threshold is chosen on train.
5. **Assistant** (`src/rag.py`, `src/api.py`):
   - Built as a LangChain LCEL chain: question routing, then structured context (block log sequence, detector scores, host risk, rack alerts), then BM25 retrieval over a train-only knowledge base (template cards, past labelled incidents, host and alert cards), then the prompt, then the LLM.
   - FastAPI endpoints: `/ask`, `/block/{id}`, `/health`.
6. **Evaluation** (`src/evaluate.py`): a 97-item golden set built from held-out blocks and analytic ground truth.

Run everything with `./run_all.sh`, then `pytest tests` (9 tests).

## Results (from `results/*.json`)

**Detection on held-out (later) blocks, 121 anomalies of 2,382:**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Rule: any exception template | 1.000 | 0.298 | 0.459 | 0.649 | 0.333 |
| Isolation Forest | 0.979 | 0.380 | 0.548 | 0.770 | 0.487 |
| Logistic regression (supervised) | 0.979 | 0.388 | 0.556 | 0.612 | 0.416 |
| **PCA residual (unsupervised)** | **0.962** | **0.628** | **0.760** | **0.816** | **0.608** |
| **PCA + replica-host risk (graph)** | **0.957** | **0.727** | **0.826** | – | – |

- **Recall ceiling.** 44 of the 121 test anomalies (36.4%) have exactly the same event-count vector as the most common healthy block, which accounts for 81.25% of normal blocks. Any detector using log-count features alone therefore tops out at 63.6% recall without flagging that healthy majority. PCA reaches 62.8%.
- **Topology helps.** Adding the topology signal rescues 12 of those anomalies at the cost of 1 extra false alarm, lifting recall to 72.7%. The gain comes from a single host, `10.251.106.10`, with 55 of its 62 blocks anomalous (binomial p = 2.2e-69, Bonferroni-significant). None of its own log lines show an exception, so the root cause is not visible in this excerpt.
- **Supervised underperforms.** The supervised model did worse than unsupervised PCA. That is reported as-is, not tuned away.

**Hotspots:** 3 of 203 hosts have Bonferroni-significant anomaly concentration. On BGL, rack R30 accounts for 61 of the 143 alert lines.

**Assistant golden set:**
- 97 items in total; 87.6% overall task accuracy.
- Block verdicts: 85.0% accuracy (80 items), with precision 0.938 and recall 0.750.
- Host, rack, template and missing-block questions all answered correctly. These sets are small (2–6 items each).
- 0 ungrounded citations out of 732.
- Latency: p50 25 ms, p95 35 ms in-process.

## Limitations, disclosed

- **LLM backend not executed.** The Claude backend (`langchain-anthropic`, enabled by `ANTHROPIC_API_KEY`) is wired in but was not run, because no API key was available in the build sandbox. The reported eval uses the deterministic grounded generator, so its 0-ungrounded-citations result holds by construction. The grounding check is there to score the LLM backend once it runs.
- **Neo4j not executed.** The loader and Cypher queries were not run, because Neo4j could not be installed in the sandbox (Docker Hub and Neo4j downloads were blocked). `results/graph_metrics.json` holds the expected output for a parity check.
- **Changes after the first eval run.** The first eval scored 82.5% overall (`results/rag_eval_metrics_v1.json`). Two items were then fixed:
  - Templates first seen in the test window were missing from the knowledge base, so it now holds the full parsed template catalog, with outcome rates still drawn from train only.
  - A scorer bug counted blast-radius peers as hotspot hosts.

  A tokenizer with lowercase and suffix stemming was also added at the same time.
- **Threshold choice.** The PCA and Isolation Forest thresholds are the train 96th percentile, a value picked from the roughly 4% anomaly base rate.
- **Excerpt only.** Results cover a 26-minute excerpt, not the full HDFS run. Block labels come from the full run, so some "anomalous" blocks may fail outside this window.
- **Dataset quirk.** In this release every `Receiving block ... src: dest:` line has src equal to dest. Host-to-host edges therefore come from co-replica placement and re-replication lines rather than those transfer lines.
