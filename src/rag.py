"""LangChain incident-troubleshooting assistant over real HDFS/BGL telemetry.

Pipeline (LCEL):  question
   -> route: pull block ids / host ips / rack ids out of the question (regex)
   -> structured context: block log sequence + PCA anomaly score (DuckDB), host anomaly
      concentration + co-replica peers (graph results), rack alert counts (BGL)
   -> BM25 retrieval over a knowledge base built from the TRAIN split only:
      event-template cards, past labelled block incidents, host cards, BGL alert-type cards
   -> prompt -> LLM (Claude via langchain-anthropic when ANTHROPIC_API_KEY is set,
      otherwise a deterministic grounded generator so the pipeline and eval run offline)
   -> answer with cited evidence ids

No test-split block appears in the retrieval corpus, so evaluating on test blocks is leakage-free.
"""
import json
import os
import re
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "netincident.duckdb"
RES = ROOT / "results"

BLK = re.compile(r"blk_-?\d+")
IPRE = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")
RACK = re.compile(r"\bR\d{2}\b")
EXCEPTION_EVENTS = {"E7", "E10", "E14", "E27", "E8", "E13"}

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     ("You are a network/storage operations assistant. Answer ONLY from the context. "
     "Give a one-line verdict first, then evidence as bullet points citing ids in [brackets] "
     "exactly as they appear in the context (block ids, host ips, event ids like E7, rack ids). "
     "If the context does not contain the answer, say so.")),
    ("human", "Question: {question}\n\nStructured context:\n{structured}\n\nRetrieved knowledge:\n{retrieved}"),
])


# ---------------------------------------------------------------- data access
@lru_cache(maxsize=1)
def _con():
    return duckdb.connect(str(DB), read_only=True)


@lru_cache(maxsize=1)
def scores():
    return pd.read_csv(RES / "test_scores.csv").set_index("block_id")


@lru_cache(maxsize=1)
def host_conc():
    return pd.read_csv(RES / "host_anomaly_concentration.csv").set_index("ip")


def block_events(block_id):
    return _con().execute(
        "SELECT event_id, COUNT(*) n, any_value(event_template) t FROM hdfs_events WHERE block_id = ? "
        "GROUP BY 1 ORDER BY 1", [block_id]).fetchall()


def block_hosts(block_id):
    return [r[0] for r in _con().execute(
        "SELECT DISTINCT host_ip FROM hdfs_replicas WHERE block_id = ? ORDER BY 1", [block_id]).fetchall()]


def seq_text(events):
    return " ".join(f"{e}x{n}" for e, n, _ in events)


# ---------------------------------------------------------------- knowledge base
@lru_cache(maxsize=1)
def knowledge_base():
    con = _con()
    sc = scores()
    train_ids = set(sc.index[sc["split"] == "train"])
    docs = []
    # Template catalog = every parsed template (parser output, no labels); outcome stats from TRAIN only.
    tmpl = con.execute("""
        SELECT e.event_id, any_value(e.event_template) t, any_value(e.level) lvl,
               COUNT(DISTINCT CASE WHEN b.is_anomaly AND e.block_id IN (SELECT unnest(?)) THEN e.block_id END) a,
               COUNT(DISTINCT CASE WHEN NOT b.is_anomaly AND e.block_id IN (SELECT unnest(?)) THEN e.block_id END) n
        FROM hdfs_events e JOIN hdfs_blocks b USING(block_id) GROUP BY 1 ORDER BY 1""",
                       [list(train_ids), list(train_ids)]).fetchall()
    for eid, t, lvl, a, n in tmpl:
        if a + n:
            stat = (f"seen in {a} anomalous and {n} normal training blocks; "
                    f"{a / (a + n):.1%} of blocks containing {eid} were anomalous.")
        else:
            stat = "not seen in the training window, so no historical outcome rate is available."
        docs.append(Document(page_content=f"template {eid} {t} level {lvl}. {stat}",
                             metadata={"kind": "template", "id": eid}))
    # past incidents: every training block, as an event-sequence document
    seqs = con.execute("""
        SELECT be.block_id, string_agg(be.event_id || 'x' || be.cnt, ' ' ORDER BY be.event_id), b.is_anomaly
        FROM hdfs_block_event be JOIN hdfs_blocks b USING(block_id)
        WHERE be.block_id IN (SELECT unnest(?)) GROUP BY be.block_id, b.is_anomaly
        ORDER BY be.block_id""", [list(train_ids)]).fetchall()
    for bid, seq, y in seqs:
        docs.append(Document(page_content=f"past block {bid} events {seq} outcome {'ANOMALY' if y else 'normal'}",
                             metadata={"kind": "incident", "id": bid, "label": int(y), "seq": seq}))
    for ip, r in host_conc().iterrows():
        docs.append(Document(
            page_content=f"host {ip} datanode touched {int(r.blocks)} blocks, {int(r.anomalous)} anomalous "
                         f"({r.rate:.1%}); binomial p={r.p_value:.2g}; "
                         f"{'statistically elevated anomaly concentration' if r.significant else 'not elevated'}",
            metadata={"kind": "host", "id": ip, "significant": bool(r.significant)}))
    for at, n, racks in con.execute("""
            SELECT alert_type, COUNT(*), string_agg(DISTINCT rack, ' ') FROM bgl_events
            WHERE is_alert GROUP BY 1 ORDER BY 1""").fetchall():
        docs.append(Document(page_content=f"BGL alert type {at} occurred {n} times on racks {racks}",
                             metadata={"kind": "bgl_alert", "id": at}))
    for sid, sig, cat, sev, n, srcs in con.execute("""
            SELECT signature_id, any_value(signature), any_value(category), any_value(severity),
                   COUNT(*), COUNT(DISTINCT src_ip) FROM sec_alerts GROUP BY 1 ORDER BY 1""").fetchall():
        docs.append(Document(page_content=f"IDS signature {sid} {sig}; category {cat}; severity {sev}; "
                                          f"fired {n} times from {srcs} source IPs in the Suricata sensor data",
                             metadata={"kind": "signature", "id": str(sid)}))
    return docs


@lru_cache(maxsize=1)
def seq_history():
    hist = {}
    for d in knowledge_base():
        if d.metadata["kind"] == "incident":
            hist.setdefault(d.metadata["seq"], Counter())[d.metadata["label"]] += 1
    return hist


TOKEN = re.compile(r"[a-z0-9_.]+")


def tokenize(text):
    """Lowercase, split on punctuation, crude suffix stemming (deleted/deleting -> delet)."""
    out = []
    for t in TOKEN.findall(text.lower()):
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suf) and not t[-len(suf) - 1].isdigit():
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


@lru_cache(maxsize=1)
def retrievers():
    kb = knowledge_base()
    incidents = [d for d in kb if d.metadata["kind"] == "incident"]
    general = [d for d in kb if d.metadata["kind"] != "incident"]
    return (BM25Retriever.from_documents(general, k=4, preprocess_func=tokenize),
            BM25Retriever.from_documents(incidents, k=10))


# ---------------------------------------------------------------- context building
def structured_context(q):
    ctx = {"blocks": [], "hosts": [], "racks": [], "sources": [], "scanners": []}
    for b in dict.fromkeys(BLK.findall(q)):
        ev = block_events(b)
        if not ev:
            ctx["blocks"].append({"id": b, "found": False})
            continue
        s = scores().loc[b] if b in scores().index else None
        ctx["blocks"].append({
            "id": b, "found": True, "events": [[e, n] for e, n, _ in ev],
            "exception_events": [e for e, _, _ in ev if e in EXCEPTION_EVENTS],
            "replica_hosts": block_hosts(b),
            "pca_residual": None if s is None else float(s.pca_residual),
            "pca_flag": None if s is None else bool(s.pca_flag),
            "replica_host_risk": None if s is None else round(float(s.host_risk), 4),
        })
        b = ctx["blocks"][-1]
        b["host_risk_threshold"] = risk_threshold()
        # detector decision handed to the generator, so the LLM explains rather than re-detects
        b["detector_verdict"] = "ANOMALOUS" if block_verdict(b) else "normal"
    hc = host_conc()
    want_hosts = list(dict.fromkeys(IPRE.findall(q)))
    if not want_hosts and re.search(r"\bhosts?\b|datanode", q, re.IGNORECASE):
        want_hosts = list(hc.index[hc.significant])
    graph = json.loads((RES / "graph_metrics.json").read_text())
    for ip in want_hosts:
        if ip in hc.index:
            r = hc.loc[ip]
            h = {"ip": ip, "blocks": int(r.blocks), "anomalous": int(r.anomalous), "rate": round(float(r.rate), 4),
                 "p_value": float(f"{r.p_value:.3g}"), "significant": bool(r.significant)}
            if ip == graph["anomaly_hotspot_neighbors"]["host"]:
                h["top_co_replica_peers"] = graph["anomaly_hotspot_neighbors"]["top_peers"]
            ctx["hosts"].append(h)
    racks = RACK.findall(q)
    if racks or re.search(r"\brack|BGL", q, re.IGNORECASE):
        rows = _con().execute("""SELECT rack, SUM(is_alert::INT) a FROM bgl_events WHERE rack IS NOT NULL
                                 GROUP BY 1 ORDER BY a DESC, rack""").fetchall()
        ctx["racks"] = [{"rack": r, "alerts": int(a)} for r, a in rows if not racks or r in racks][:5]
    for ip in want_hosts if want_hosts else IPRE.findall(q):
        src = security_source(ip)
        if src:
            ctx["sources"].append(src)
    if re.search(r"scanning|scanners?\b|which sources?|attack sources|top attackers|probing", q, re.IGNORECASE) and not ctx["sources"]:
        sec = json.loads((RES / "security_metrics.json").read_text())
        ctx["scanners"] = sec["graph"]["top_fanout_sources"]
    return ctx


def security_source(ip):
    """IDS view of one source IP from the Suricata telemetry, or None if the IP never appears there."""
    con = _con()
    f = con.execute("""SELECT COUNT(*), COUNT(DISTINCT dest_port), SUM(alerted::INT), MIN(start), MAX(start)
                       FROM sec_flows WHERE src_ip = ?""", [ip]).fetchone()
    a = con.execute("""SELECT signature, category, COUNT(*) n FROM sec_alerts WHERE src_ip = ?
                       GROUP BY 1,2 ORDER BY n DESC LIMIT 3""", [ip]).fetchall()
    if not f[0] and not a:
        return None
    return {"ip": ip, "flows": int(f[0]), "distinct_dest_ports": int(f[1] or 0), "alerted_flows": int(f[2] or 0),
            "first_seen": str(f[3]), "last_seen": str(f[4]), "alerts": sum(n for _, _, n in a) if a else 0,
            "top_signatures": [[sig, cat, int(n)] for sig, cat, n in a]}


def retrieve(q, ctx):
    general, incidents = retrievers()
    docs = general.invoke(q)
    neighbors = []
    for b in ctx["blocks"]:
        if b.get("found"):
            query = " ".join(f"{e}x{n}" for e, n in b["events"])
            # BM25 ranks by shared event-count tokens; keep only exact-sequence neighbours if any exist
            # exact-sequence history is a full lookup over all training blocks (BM25 top-k would
            # arbitrarily truncate the thousands of identical healthy sequences); fall back to BM25.
            hist = seq_history().get(query)
            hits = incidents.invoke(query)
            if hist:
                b["similar_past"], b["similar_past_exact"] = hist, True
            else:
                b["similar_past"] = Counter(h.metadata["label"] for h in hits[:5])
                b["similar_past_exact"] = False
            neighbors += hits[:3]
    return docs + neighbors


def fmt_ctx(ctx):
    return json.dumps(ctx, default=lambda o: dict(o) if isinstance(o, Counter) else str(o), separators=(",", ":"))


def fmt_docs(docs):
    return "\n".join(f"- ({d.metadata['kind']}:{d.metadata['id']}) {d.page_content}" for d in docs)


# ---------------------------------------------------------------- generation
@lru_cache(maxsize=1)
def risk_threshold():
    return json.loads((RES / "graph_feature_metrics.json").read_text())["risk_threshold_chosen_on_train"]


def block_verdict(b):
    """Detector decision: exception template fired OR PCA flag OR replica host risk >= train-chosen threshold."""
    return bool(b["exception_events"]) or bool(b["pca_flag"]) or (b["replica_host_risk"] or 0) >= risk_threshold()


def grounded_answer(x):
    """Deterministic generator: composes the answer strictly from structured context + retrieved docs."""
    ctx, docs = x["ctx"], x["docs"]
    lines = []
    for b in ctx["blocks"]:
        if not b.get("found"):
            lines.append(f"Block [{b['id']}] does not appear in the indexed logs.")
            continue
        anom = block_verdict(b)
        lines.append(f"Verdict: block [{b['id']}] is {'LIKELY ANOMALOUS' if anom else 'likely normal'}.")
        if b["exception_events"]:
            lines.append("- Exception/warning templates fired: " + ", ".join(f"[{e}]" for e in b["exception_events"]))
        lines.append(f"- PCA residual {b['pca_residual']:.2f} ({'above' if b['pca_flag'] else 'below'} the "
                     f"training 96th-percentile threshold)")
        seq = " ".join(f"[{e}]x{n}" for e, n in b["events"])
        lines.append(f"- Event sequence: {seq}")
        if b["replica_hosts"]:
            lines.append("- Replicas stored on: " + ", ".join(f"[{h}]" for h in b["replica_hosts"]))
            hr = b["replica_host_risk"]
            lines.append(f"- Replica-host risk {hr:.1%} (highest earlier anomaly rate among its hosts; "
                         f"{'at/above' if hr >= risk_threshold() else 'below'} the {risk_threshold():.0%} threshold)")
        sp = b.get("similar_past", {})
        if sp:
            lines.append(f"- Similar past blocks ({'identical sequence' if b['similar_past_exact'] else 'nearest by BM25'}):"
                         f" {sp.get(1, 0)} anomalous / {sp.get(0, 0)} normal")
    if ctx["hosts"]:
        sig = [h for h in ctx["hosts"] if h["significant"]]
        lines.append(f"Verdict: {len(sig)} of {len(ctx['hosts'])} host(s) shown have statistically elevated "
                     f"anomalous-block concentration.")
        for h in ctx["hosts"]:
            lines.append(f"- [{h['ip']}]: {h['anomalous']}/{h['blocks']} blocks anomalous ({h['rate']:.1%}), "
                         f"p={h['p_value']}{' (significant)' if h['significant'] else ''}")
            if h.get("top_co_replica_peers"):
                peers = ", ".join(f"[{p}] ({a}/{n} shared anomalous)" for p, n, a in h["top_co_replica_peers"][:3])
                lines.append(f"  blast radius, top co-replica peers: {peers}")
    if ctx["racks"]:
        top = ctx["racks"][0]
        lines.append(f"Verdict: rack [{top['rack']}] has the most BGL alerts ({top['alerts']}).")
        lines += [f"- [{r['rack']}]: {r['alerts']} alerts" for r in ctx["racks"][1:]]
    for src in ctx["sources"]:
        bad = src["alerts"] > 0 or src["alerted_flows"] > 0
        lines.append(f"Verdict: source [{src['ip']}] was {'FLAGGED by the IDS' if bad else 'not flagged by the IDS'} "
                     f"({src['alerts']} alerts).")
        lines.append(f"- {src['flows']} flows to {src['distinct_dest_ports']} distinct ports, "
                     f"{src['alerted_flows']} alerted; seen {src['first_seen']} to {src['last_seen']}")
        for sig, cat, n in src["top_signatures"]:
            lines.append(f"- [{sig}] ({cat}) x{n}")
    if ctx["scanners"]:
        top = ctx["scanners"][0]
        lines.append(f"Verdict: [{top['src_ip']}] is the widest scanner, probing {top['ports']} distinct ports.")
        lines += [f"- [{x['src_ip']}]: {x['ports']} ports, {x['flows']} flows" for x in ctx["scanners"][1:]]
    if not lines:
        tm = [d for d in docs if d.metadata["kind"] in ("template", "bgl_alert", "host", "signature")][:3]
        if not tm:
            return "The indexed telemetry does not contain an answer to this question."
        lines.append("Closest knowledge-base entries:")
        lines += [f"- [{d.metadata['id']}] {d.page_content}" for d in tm]
    return "\n".join(lines)


def llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.environ.get("NETINCIDENT_MODEL", "claude-sonnet-4-5"), temperature=0,
                             max_tokens=600)
    return None


def build_chain():
    prep = RunnablePassthrough.assign(ctx=lambda x: structured_context(x["question"]))
    prep = prep | RunnablePassthrough.assign(docs=lambda x: retrieve(x["question"], x["ctx"]))
    model = llm()
    if model is None:
        gen = RunnableLambda(grounded_answer)
        backend = "deterministic-grounded"
    else:
        gen = ({"question": lambda x: x["question"], "structured": lambda x: fmt_ctx(x["ctx"]),
                "retrieved": lambda x: fmt_docs(x["docs"])} | PROMPT | model | StrOutputParser())
        backend = f"anthropic:{model.model}"
    chain = prep | RunnablePassthrough.assign(answer=gen)
    return chain, backend


def ask(question, chain=None):
    chain = chain or build_chain()[0]
    out = chain.invoke({"question": question})
    return {"question": question, "answer": out["answer"], "context": json.loads(fmt_ctx(out["ctx"])),
            "retrieved": [f"{d.metadata['kind']}:{d.metadata['id']}" for d in out["docs"]]}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Which hosts have elevated anomaly rates?"
    print(ask(q)["answer"])
