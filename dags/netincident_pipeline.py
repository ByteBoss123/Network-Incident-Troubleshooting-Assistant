"""Airflow DAG: the Network Incident pipeline end to end, with data-quality gates.

    etl -> quality_gate -> detect -> graph -> graph_feature -> deeplog_lstm -> evaluate
                        \\-> security --------------------------------------/
                                    \\-> netconfig_audit

Each task runs one project script with the project's own interpreter (PROJECT_PYTHON), so the
Airflow environment only needs Airflow. `quality_gate` fails the run before any modeling if the
ETL output breaks its invariants (row counts, unique ids, no unlabeled event lines).

Run once without a scheduler:  airflow dags test netincident_pipeline 2026-10-09
"""
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT = os.environ.get("NETINCIDENT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("PROJECT_PYTHON", "python3")


def quality_gate():
    import subprocess
    code = (
        "import duckdb; c=duckdb.connect('data/netincident.duckdb', read_only=True);"
        "q=lambda s: c.execute(s).fetchone()[0];"
        "assert q('SELECT COUNT(*) FROM hdfs_events') == 104815, 'event rows';"
        "assert q('SELECT COUNT(*) - COUNT(DISTINCT block_id) FROM hdfs_blocks') == 0, 'dup block ids';"
        "assert q('SELECT COUNT(*) FROM hdfs_events e LEFT JOIN hdfs_blocks b USING(block_id) "
        "WHERE b.block_id IS NULL') == 0, 'unlabeled event lines';"
        "assert q('SELECT COUNT(*) FROM hdfs_blocks') == 7940, 'block rows';"
        "print('quality gate passed')"
    )
    out = subprocess.run([PY, "-c", code], cwd=PROJECT, capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(f"data quality gate failed: {out.stderr.strip()[-500:]}")
    print(out.stdout.strip())


def step(task_id, cmd):
    return BashOperator(task_id=task_id, bash_command=f"cd {PROJECT} && {cmd}", append_env=True)


with DAG(
    dag_id="netincident_pipeline",
    description="Log anomaly detection, topology graph, IDS triage and RAG eval on Loghub HDFS/BGL + Suricata",
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["netincident"],
) as dag:
    etl = step("etl", f"{PY} src/etl.py")
    gate = PythonOperator(task_id="quality_gate", python_callable=quality_gate)
    detect = step("detect", f"{PY} src/detect.py > /dev/null")
    graph = step("graph", f"{PY} src/graph.py > /dev/null")
    graph_feature = step("graph_feature", f"{PY} src/graph_feature.py > /dev/null")
    security = step("security", f"{PY} src/security.py > /dev/null")
    lstm = step("deeplog_lstm", f"{PY} src/deeplog_tf.py > /dev/null 2>&1")
    evaluate = step("evaluate", f"cd src && {PY} evaluate.py > /dev/null")
    # network configuration check: cloud security-group exposure vs ports attackers probe in the IDS data
    netconfig = step("netconfig_audit", f"{PY} src/netconfig_audit.py > /dev/null")

    etl >> gate >> [detect, security]
    detect >> graph >> graph_feature >> lstm >> evaluate
    security >> [evaluate, netconfig]
