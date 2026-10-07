"""ETL: real Loghub HDFS (100k lines, block labels) and BGL (2k lines, alert labels) -> DuckDB.

Tables
  hdfs_events      one row per log line (block id, host ips, event template)
  hdfs_blocks      one row per block session with its label
  hdfs_block_event block x event-template count matrix (long form)
  hdfs_transfers   host -> host block transfers parsed from "Receiving block ... src: dest:"
  hdfs_replicas    block -> host replica placement parsed from "addStoredBlock"
  bgl_events       BGL alerts with rack/midplane/node hierarchy parsed from node id
"""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "netincident.duckdb"

BLK = re.compile(r"(blk_-?\d+)")
IP = re.compile(r"(\d+\.\d+\.\d+\.\d+)(?::\d+)?")
TRANSFER = re.compile(r"src: /(\d+\.\d+\.\d+\.\d+):\d+ dest: /(\d+\.\d+\.\d+\.\d+):\d+")
STORED = re.compile(r"addStoredBlock: blockMap updated: (\d+\.\d+\.\d+\.\d+):\d+ is added to (blk_-?\d+)")
BGL_NODE = re.compile(r"^(R\d+)-(M\d+)-(N[0-9A-F]+)")


def build(db_path: Path = DB) -> dict:
    con = duckdb.connect(str(db_path))
    ev = pd.read_csv(RAW / "HDFS_100k.log_structured.csv", dtype=str)
    labels = pd.read_csv(RAW / "anomaly_label.csv", dtype=str)

    ev["block_id"] = ev["Content"].str.extract(BLK, expand=False)
    ev["ts"] = pd.to_datetime("20" + ev["Date"] + ev["Time"].str.zfill(6), format="%Y%m%d%H%M%S")
    n_raw = len(ev)
    ev = ev.dropna(subset=["block_id"])
    con.register("ev_df", ev[["LineId", "ts", "Level", "Component", "Content", "EventId", "EventTemplate", "block_id"]])
    con.register("lab_df", labels)
    con.execute("""
        CREATE OR REPLACE TABLE hdfs_events AS
        SELECT CAST(LineId AS INTEGER) AS line_id, ts, Level AS level, Component AS component,
               Content AS content, EventId AS event_id, EventTemplate AS event_template, block_id
        FROM ev_df""")
    # Only blocks whose whole session we can label; label file covers the full 11M-line HDFS v1 run.
    con.execute("""
        CREATE OR REPLACE TABLE hdfs_blocks AS
        SELECT e.block_id, l.Label = 'Anomaly' AS is_anomaly,
               MIN(e.ts) AS first_ts, MAX(e.ts) AS last_ts, COUNT(*) AS n_lines,
               COUNT(DISTINCT e.event_id) AS n_distinct_events
        FROM hdfs_events e JOIN lab_df l ON l.BlockId = e.block_id
        GROUP BY e.block_id, l.Label""")
    con.execute("""
        CREATE OR REPLACE TABLE hdfs_block_event AS
        SELECT block_id, event_id, COUNT(*) AS cnt FROM hdfs_events GROUP BY 1, 2""")

    tr = ev["Content"].str.extract(TRANSFER).dropna()
    tr.columns = ["src_ip", "dst_ip"]
    tr["block_id"] = ev.loc[tr.index, "block_id"]
    st = ev["Content"].str.extract(STORED).dropna()
    st.columns = ["host_ip", "block_id"]
    con.register("tr_df", tr)
    con.register("st_df", st)
    con.execute("CREATE OR REPLACE TABLE hdfs_transfers AS SELECT * FROM tr_df")
    con.execute("CREATE OR REPLACE TABLE hdfs_replicas AS SELECT DISTINCT * FROM st_df")

    bgl = pd.read_csv(RAW / "BGL_2k.log_structured.csv", dtype=str)
    hier = bgl["Node"].str.extract(BGL_NODE)
    bgl["rack"], bgl["midplane"], bgl["node_card"] = hier[0], hier[1], hier[2]
    bgl["is_alert"] = bgl["Label"] != "-"
    con.register("bgl_df", bgl)
    con.execute("""
        CREATE OR REPLACE TABLE bgl_events AS
        SELECT CAST(LineId AS INTEGER) AS line_id, Label AS alert_type, is_alert,
               to_timestamp(CAST(Timestamp AS BIGINT)) AS ts, Node AS node, rack, midplane, node_card,
               Component AS component, Level AS level, Content AS content, EventId AS event_id, EventTemplate AS event_template
        FROM bgl_df""")

    stats = {
        "hdfs_raw_lines": n_raw,
        "hdfs_lines_with_block": con.execute("SELECT COUNT(*) FROM hdfs_events").fetchone()[0],
        "hdfs_blocks": con.execute("SELECT COUNT(*) FROM hdfs_blocks").fetchone()[0],
        "hdfs_anomalous_blocks": con.execute("SELECT SUM(is_anomaly::INT) FROM hdfs_blocks").fetchone()[0],
        "hdfs_event_templates": con.execute("SELECT COUNT(DISTINCT event_id) FROM hdfs_events").fetchone()[0],
        "hdfs_transfers": con.execute("SELECT COUNT(*) FROM hdfs_transfers").fetchone()[0],
        "hdfs_hosts": con.execute(
            "SELECT COUNT(*) FROM (SELECT src_ip AS ip FROM hdfs_transfers UNION SELECT dst_ip FROM hdfs_transfers "
            "UNION SELECT host_ip FROM hdfs_replicas)").fetchone()[0],
        "hdfs_replica_edges": con.execute("SELECT COUNT(*) FROM hdfs_replicas").fetchone()[0],
        "bgl_lines": con.execute("SELECT COUNT(*) FROM bgl_events").fetchone()[0],
        "bgl_alert_lines": con.execute("SELECT SUM(is_alert::INT) FROM bgl_events").fetchone()[0],
        "bgl_nodes": con.execute("SELECT COUNT(DISTINCT node) FROM bgl_events").fetchone()[0],
        "bgl_racks": con.execute("SELECT COUNT(DISTINCT rack) FROM bgl_events").fetchone()[0],
    }
    # orphan / duplicate checks
    stats["dup_block_rows"] = con.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT block_id) FROM hdfs_blocks").fetchone()[0]
    stats["unlabeled_block_lines"] = con.execute(
        "SELECT COUNT(*) FROM hdfs_events e LEFT JOIN hdfs_blocks b USING(block_id) WHERE b.block_id IS NULL"
    ).fetchone()[0]
    con.close()
    return stats


if __name__ == "__main__":
    s = build()
    for k, v in s.items():
        print(f"{k:28s} {v}")
    sys.exit(0)
