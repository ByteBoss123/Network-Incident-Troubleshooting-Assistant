# Network Incident Troubleshooting Assistant

Log anomaly detection, a telemetry-derived topology graph, IDS security telemetry, and a LangChain retrieval assistant for incident triage, evaluated with real LLMs on Amazon Bedrock. Everything runs on real, public data.

## Data

| Source | What | Size |
|---|---|---|
| Loghub HDFS v1 (logpai/loglizer mirror) | Hadoop DataNode/NameNode logs, 26-minute excerpt, 19 parsed event templates | 104,815 lines, 7,940 blocks, 203 hosts |
| HDFS v1 `anomaly_label.csv` | Expert block-level labels from the full run | 313 anomalous blocks (3.94%) |
| Loghub BGL (Blue Gene/L) | RAS log sample with alert labels; node ids parsed into rack → midplane → node | 2,000 lines, 143 alerts, 64 racks |
| Stratosphere Lab Suricata capture (StratosphereLinuxIPS repo) | Real IDS sensor output (EVE JSON), 24 h, 2021-06-06/07: NetFlow-style flow records, signature alerts, DNS | 5,000 events: 3,512 flows, 745 alerts, 122 signatures, 1,057 external IPs |

`data/download.sh` fetches everything and verifies SHA-256 checksums.

## Pipeline (`./run_all.sh` or the Airflow DAG, then `pytest tests` — 13 tests)

1. **ETL** (`src/etl.py`, `src/security.py`): Python + DuckDB SQL into event, block-session, block×template, replica, BGL hierarchy, and `sec_flows` / `sec_alerts` / `sec_dns` tables. Checks: 0 duplicate block ids, 0 unlabeled event lines, 0 duplicate flow ids.
2. **Log anomaly detection** (`src/detect.py`): block event-count vectors, chronological 70/30 split.
3. **Topology graph** (`src/graph.py`, `src/neo4j_load.py`): host–host co-replica edges (13,985), re-replication transfers, subnets, Block→Host edges, BGL Rack→Midplane→Node, and the security graph (IP –FLOW{port}→ IP, IP –TRIGGERED→ Signature). Exported as CSV with a Neo4j loader, Cypher queries (hotspot hosts, blast radius, top scanners, top signatures, rack alerts, shortest path), and a parity checker (`src/neo4j_check.py`) against the DuckDB/NetworkX results.
4. **Graph feature** (`src/graph_feature.py`): leakage-free replica-host risk (host failure rates from train blocks only; threshold chosen on train).
5. **Security analytics** (`src/security.py`): hourly alert series with a robust z-score burst detector, scanner fan-out, and an alert-triage model that predicts which inbound flows trip a Suricata signature from flow metadata only (ports, protocol, packets, bytes, duration, state; no IPs or reputation lists), chronological split.
6. **Assistant** (`src/rag.py`, `src/api.py`): LangChain LCEL chain. It routes the question, builds structured context (block log sequence with the detector's verdict, host risk, rack alerts, IDS view of a source IP, top scanners), BM25-retrieves from a train-only knowledge base (template cards, past labelled incidents, host cards, IDS signature cards, BGL alert cards), then generates with an LLM or a deterministic grounded generator. FastAPI: `/ask`, `/block/{id}`, `/health`.
7. **Evaluation** (`src/evaluate.py`, `src/llm_eval.py`): 114-item golden set from held-out data and analytic ground truth (80 block verdicts, host, rack, template, signature, source-IP, scanner and missing-id questions). LLM answers are checked for correctness and for hallucinated ids: any block id, IP, event id, rack or signature id not present in the exact prompt sent.
8. **Dashboard** (`dashboard/`): React console built from the results (`python dashboard/build_data.py && python dashboard/build.py`), with an in-page block / source-IP triage lookup.

## Results (from `results/*.json`)

**Log anomaly detection, 2,382 later blocks, 121 anomalies:**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Exception-template rule | 1.000 | 0.298 | 0.459 | 0.649 | 0.333 |
| Isolation Forest | 0.979 | 0.380 | 0.548 | 0.770 | 0.487 |
| Logistic regression (supervised) | 0.979 | 0.388 | 0.556 | 0.612 | 0.416 |
| PCA residual (unsupervised) | 0.962 | 0.628 | 0.760 | 0.816 | 0.608 |
| **PCA + replica-host risk (graph)** | **0.957** | **0.727** | **0.826** | – | – |

- 44 of 121 test anomalies (36.4%) have exactly the most common healthy event-count vector, so count features alone cap recall at 63.6%. The topology signal rescued 12 of them for 1 extra false alarm. The gain comes from one host, `10.251.106.10` (55/62 blocks failed, binomial p = 2.2e-69).
- 3 of 203 hosts are Bonferroni-significant hotspots. BGL rack R30 has 61 of 143 alert lines.

**Security telemetry (Suricata, 24 h):**
- Widest scanner `193.46.255.92` probed 482 distinct ports in 536 flows. The top signature, MSSQL port 1433 inbound scan, fired 131 times from 56 sources.
- Alert triage from flow metadata (698 later inbound flows, 16.8% alerted): gradient boosting PR-AUC 0.430 (2.6x the base rate), ROC-AUC 0.777, recall 0.752 at precision 0.294. Logistic regression reached PR-AUC 0.233; a sensitive-port rule scored below chance (ROC-AUC 0.391).
- 88.0% of alerted test flows carry a reputation-list signature, which flow features cannot see directly; the model still recalled 75.7% of them versus 71.4% of behavioural alerts.
- No hourly burst crossed the robust z > 3.5 threshold (median 30 alerts/hour).

**Assistant evaluation:**

| Generator | Set | Items | Accuracy | Block P / R | Hallucinated ids | Content-filtered | p50 latency |
|---|---|---|---|---|---|---|---|
| Deterministic grounded (no LLM) | full | 114 | 89.5% | 0.94 / 0.75 | 0 of 769 cited | – | 11 ms |
| Llama 3.3 70B (Bedrock) | 50-item stratified subset | 50 | 96.0% | 1.00 / 0.88 | 0 of 200 | 0 | 3.3 s |
| Amazon Nova Pro (Bedrock) | 50-item stratified subset | 50 | 82.0% | 1.00 / 0.88 | 0 of 134 | 6 | 3.6 s |
| Qwen2.5-1.5B-Instruct, local CPU (8 vCPU, no API) | 50-item stratified subset | 50 | 62.0% | 0.62 / 1.00 | 0 of 106 | 0 | 17.0 s |

- Both hosted LLMs followed the detector's verdict and cited only ids present in their prompt.
- **Local inference** (`local_llm/run_local.py`, Hugging Face Transformers on CPU, greedy decoding, same frozen prompts):
  Qwen2.5-1.5B also cited 0 hallucinated ids, yet answered 62% (64% before the scorer fix below): it called 6 of 12 flagged/unflagged sources wrong,
  read "detector_verdict: normal" as "not healthy" on 5 blocks, and invented a failure cause for both
  nonexistent block ids. Id-grounding alone does not catch fabricated reasoning; the verdict checks do.
  About 4.6 output tokens/s on 8 vCPU (median 17 s per answer), 11 s model load.
- The first two local runs produced gibberish for every prompt of roughly 410-510 tokens (and only those);
  switching from PyTorch's fused SDPA attention to the eager implementation fixed it (runner documents this).
- Nova Pro's content filter refused 6 security questions ("Is <IP> malicious?"), counted as misses; its other 3 misses list only the top hotspot host when 3 qualify (2) or follow a detector miss (1).
- Llama's 2 misses: a block the detector itself missed (the model correctly reported the detector's "normal"), and a template question it said the context did not cover.

**Sequence model (TensorFlow, `src/deeplog_tf.py`).** A DeepLog-style 2-layer LSTM (TensorFlow/Keras) learns
the next log event of normal blocks (58,216 training windows, top-g chosen on validation only).
On the same 2,382 test blocks it reaches F1 0.569 (P 0.737, R 0.463), below PCA's 0.760; every anomaly it
catches PCA also catches (56 of 56), with 20 false alarms vs PCA's 3. The reason is in the data: all 45 test
anomalies PCA misses have event sequences identical, in order, to normal training blocks, so no log-only
model (counts or order) can separate them. Only the topology signal recovered any of them (12).

**Orchestration (Airflow, `dags/netincident_pipeline.py`).** `etl -> quality_gate -> detect / security ->
graph -> graph_feature -> deeplog_lstm -> evaluate`, retries 1. The quality gate fails the run before any
modeling if row counts, block-id uniqueness or event labeling break. `airflow dags test` (Airflow 2.10.5):
8 of 8 tasks succeeded in about 63 s; the run reproduced the detection, graph-feature and security metrics
byte for byte. The rerun exposed nondeterministic tie ordering in the hotspot peer list, now fixed.

**Local inference optimization (`local_llm/run_gguf.py`, `results/local_inference_optimization.json`).** Same 50
frozen prompts, Qwen2.5-1.5B-Instruct, greedy decoding, all three GGUF runs in one job on a 48-vCPU machine:

| llama.cpp precision | Accuracy | Model file | Median latency | Output tokens/s |
|---|---|---|---|---|
| FP16 (baseline) | 60% | 3,560 MB | 2.93 s | 25.1 |
| **Q8_0** | **60%** (same verdict on all 50 items) | **1,895 MB** | **2.16 s** | **37.6** |
| Q4_K_M | 54% | 1,117 MB | 1.86 s | 35.3 |

8-bit cut the model 47%, median latency 26% and raised throughput 50% with identical per-item results. 4-bit lost
3 questions, one of them dangerous for triage (an anomalous block called normal), so Q8_0 is the shipping choice.
For reference, the Hugging Face fp32 run scores 62%. PyTorch dynamic int8 (activations quantized at run time) broke
the model in every variant tried (all layers; per-channel weights with an fp32 output head): 36-38%, 46-48 of 50
answers ran to the 300-token cap, 32-125 hallucinated ids.

**Scorer fix found during this work.** llama.cpp first appeared to score 48% vs Hugging Face's 64%. Reading the
answers showed the rule-based judge misread negations: "No, there is no evidence ... malicious" counted as calling
the source malicious, and "No anomaly detected" counted as anomalous. The judge now honors a leading yes/no and
negated mentions. Every run was rescored (`src/rescore_all.py`, `results/rescore_scorer_v1.json` vs `_v2.json`):
all Bedrock scores are unchanged; Hugging Face fp32 64% -> 62% (one lucky pass removed); llama.cpp FP16/Q8 48% -> 60%.

**Prompt engineering A/B (`src/prompt_ab.py`, `results/prompt_ab_metrics.json`).** Three system prompts on the same
50 frozen contexts. A rule-heavy prompt with a one-shot example, revised on half the questions only, did not beat
the original on the held-out half (Nova Lite 20/25 vs 21/25) and cut Llama 3.1 8B, never seen while writing it,
from 94% to 76%, copying the example id into 4 answers. The original grounded prompt stays. Llama 3.1 8B with the
original prompt scored 94% (0 hallucinated ids of 158), 2 points below Llama 3.3 70B.

**Network configuration audit (`src/netconfig_audit.py`, `results/netconfig_audit.json`).** A read-only snapshot
of a real AWS account's network configuration (17 regions, 17 VPCs, 21 security groups, 7 interfaces, load-balancer
listeners; `data/netconfig/`) is checked against the ports external sources probed and alerted on in the Suricata
data. One rule is open to the internet (TCP 80 on an internet-facing load balancer with an HTTP-only listener); port
80 is the third most-alerted port in the IDS data (18 alerted flows from 47 sources), so it is reported as medium.
None of the other top-10 alerted ports (1433, 22, 2375, 3389, 8080, 6379, 4573, 8545, 445) is exposed. Also flagged:
2 task interfaces with public IPs their security groups never admit internet traffic to, and 2 unattached security
groups. Runs as the `netconfig_audit` task in the Airflow DAG (9 of 9 tasks succeeded). The IDS data comes from a
different network, so it is a prior on what attackers probe, not traffic seen by this account.

**Design reviews:** `docs/design_reviews.md` (detector, graph store, generator and prompt, orchestration).

## Limitations, disclosed

- **LLM subset.** The Bedrock run used a seeded, stratified 50-item subset (all 34 non-block items + 8 anomalous + 8 normal blocks) because prompts had to be embedded in the Bedrock call script. The exact prompts sent are frozen in `results/llm_prompts_bedrock.jsonl`; prompt lengths were checked against the local copies (0 mismatches).
- **Claude on Bedrock not run.** Anthropic models on the account return "use case details have not been submitted", so the eval used Llama 3.3 70B and Nova Pro. The `langchain-anthropic` backend is wired in but unexecuted.
- **Scorer changes after audit.** After reading the LLM answers, two scorer fixes were made: host-list answers are scored on all IPs cited, not just the first line, and the id-grounding check counts the system prompt's own "E7" example as allowed. The first deterministic eval (82.5%, `results/rag_eval_metrics_v1.json`) was also followed by a template-catalog fix and a tokenizer change.
- **Neo4j run in AuraDB Free.** The graph (11,303 nodes, 58,099 relationships) was loaded into a Neo4j AuraDB Free instance through the Aura Query workspace, and all 6 Cypher parity checks matched the DuckDB/NetworkX results (`results/neo4j_query_results.json`, `results/neo4j_parity.json`). `./scripts/neo4j_run.sh` reproduces the load with the Python driver.
- **Data scope.** HDFS results cover a 26-minute excerpt with labels from the full run; the Suricata capture is one sensor over 24 hours; BGL is a 2,000-line sample. The deterministic generator's grounding result holds by construction.
- **Dataset quirk.** HDFS `Receiving block … src: dest:` lines always have src = dest, so host edges come from co-replica placement and re-replication lines.
