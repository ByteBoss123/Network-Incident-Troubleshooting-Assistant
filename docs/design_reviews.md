# Design reviews

Each review records a decision this project actually made: the options considered, the evidence used to choose,
and what was checked afterwards. Numbers come from `results/`; every review names the file.

---

## DR-1 Anomaly detector: PCA residual, plus a topology feature, over IF / LR / LSTM

**Question.** Which detector flags anomalous HDFS block sessions on a time-ordered split?

| Option | Test F1 | Note |
|---|---|---|
| Exception-template rule | 0.459 | precision 1.0, recall 0.30 |
| Isolation Forest | 0.548 | |
| Logistic regression (supervised) | 0.556 | needs labels |
| TensorFlow LSTM (DeepLog) | 0.569 | 20 false alarms vs PCA's 3 |
| **PCA residual (unsupervised)** | **0.760** | chosen |
| PCA + replica-host risk | 0.826 | +12 anomalies for +1 false alarm |

**Evidence.** `detection_metrics.json`, `deeplog_tf_metrics.json`, `graph_feature_metrics.json`.
All 45 test anomalies PCA misses have event sequences identical, in order, to normal training blocks, so no
log-only model (counts or order) can separate them; that is why the second signal is topology, not a bigger model.

**Decision.** PCA residual as the primary detector; replica-host risk (train-only rates, threshold chosen on train)
as the second signal. The LSTM stays as a documented baseline, not in the serving path.

**Risks / follow-ups.** The host-risk gain comes from one host (55/62 blocks failed); on a fleet without a bad
host it adds nothing. Re-check when new data arrives.

---

## DR-2 Graph store: Neo4j for topology queries, DuckDB/NetworkX as the reference

**Question.** Where do host/IP/rack relationships live, and how do we trust the graph answers?

**Options.** (a) NetworkX only, in process; (b) Neo4j only; (c) Neo4j for interactive Cypher queries with a
DuckDB/NetworkX reference implementation and a parity check.

**Decision.** (c). Neo4j AuraDB holds 11,303 nodes and 58,099 relationships; `src/neo4j_check.py` compares six
query results against the reference.

**Evidence.** `neo4j_parity.json`: 6 of 6 checks match (top hosts by anomaly rate, host count, co-replica edges,
top rack, top scanner, top signature).

**Consequence found later.** An Airflow rerun showed the hotspot peer list changed order between runs (ties).
Fixed with a deterministic tie-break; set-level exports were already identical.

---

## DR-3 Answer generator and system prompt

**Question.** Which LLM and prompt should generate incident answers?

| Model (same 50 frozen prompts) | Accuracy | Hallucinated ids | p50 latency |
|---|---|---|---|
| Llama 3.3 70B, Bedrock | 96% | 0 of 200 | 3.3 s |
| Llama 3.1 8B, Bedrock | 94% | 0 of 158 | 2.9 s |
| Amazon Nova Pro, Bedrock | 82% | 0 of 134 | 3.6 s (6 content-filter refusals) |
| Amazon Nova Lite, Bedrock | 80% | 0 of 101 | 3.8 s (6 refusals) |
| Qwen2.5-1.5B, local CPU | 64% | 0 of 106 | 17.0 s |

**Prompt A/B** (`prompt_ab_metrics.json`, protocol in `src/prompt_ab.py`): a rule-heavy prompt with a one-shot
example (v3) was revised on half A only. Nova Lite, held-out half B: 21/25 (original) vs 20/25 (v3).
Llama 3.1 8B, never seen while writing the prompt: 94% (original) vs 76% (v3); v3 also made Llama copy the
example id `blk_123` into 4 answers, which the standard id check does not catch because the id is in the prompt.
v2 (rules + a "Line 1" template) cut Nova Lite to 70%: the model printed "Line 1:" literally.

**Decision.** Keep the original grounded prompt. Default to Llama 3.1 8B where cost matters (2 points below 70B);
70B where accuracy matters. Do not use Nova for security questions (content-filter refusals).
Add an example-id check to the grounding test before trying few-shot prompts again.

---

## DR-4 Orchestration: Airflow DAG with a data-quality gate

**Question.** How should the pipeline run repeatedly and fail safely?

**Options.** (a) `run_all.sh` on a schedule; (b) Airflow DAG with retries and a quality gate before modeling.

**Decision.** (b), `dags/netincident_pipeline.py`. The gate asserts row counts, block-id uniqueness and that every
event line maps to a labeled block, so a broken extract stops before any model is retrained.

**Evidence.** `airflow dags test` (Airflow 2.10.5): 8 of 8 tasks succeeded in about 63 s; detection, graph-feature
and security metrics reproduced byte for byte; the RAG eval reproduced accuracy and citations (only timings differ).

---

## DR-5 Local inference: 4-bit GGUF in llama.cpp, not PyTorch dynamic int8

**Question.** How should the assistant run on a CPU-only host with no hosted API?

**Evidence** (`results/local_inference_optimization.json`, same 50 prompts, same 8-vCPU machine for the GGUF runs):
llama.cpp Q4_K_M 50% accuracy, 1,117 MB, 6.4 s median vs FP16 48%, 3,560 MB, 9.1 s. PyTorch dynamic int8
(all layers; per-channel with an fp32 head) produced degenerate output: 36-38% accuracy, 46-48 of 50 answers at the
300-token cap, 32-125 hallucinated ids.

**Decision.** Ship Q4_K_M through llama.cpp for local deployments. Treat any new quantization as a model change:
rerun the 50-question set and the hallucinated-id check before using it.
