"""Network audit on the account's own traffic: VPC Flow Logs from the public interfaces (load balancer + tasks).

Input: flow-log records (custom format `start end interface-id srcaddr dstaddr srcport dstport protocol packets bytes
action tcp-flags`) exported to data/netconfig/flowlogs_*.jsonl as {"message": "<record>"} lines.
Output: which destination ports internet sources probed on this account, whether the security groups accepted or
rejected them, and which internet-exposed ports actually received inbound traffic.
"""
import ipaddress
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")]
PROTO = {"6": "tcp", "17": "udp", "1": "icmp"}


def parse(line):
    f = line.split()
    if len(f) < 12 or f[3] == "-":
        return None
    return {"start": int(f[0]), "end": int(f[1]), "eni": f[2], "src": f[3], "dst": f[4], "sport": f[5],
            "dport": int(f[6]) if f[6].isdigit() else None, "proto": PROTO.get(f[7], f[7]),
            "packets": int(f[8]), "bytes": int(f[9]), "action": f[10]}


def external(ip):
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not any(a in n for n in PRIVATE)


EPHEMERAL = 32768  # Linux ephemeral range start; security groups are stateful


def is_reply(r):
    """ACCEPTed inbound packet to an ephemeral port = reply to a connection this host opened (stateful SG).
    Unsolicited packets to ports without an allow rule are REJECTed, so they never look like this."""
    return r["action"] == "ACCEPT" and r["dport"] >= EPHEMERAL


def audit(records, exposed_ports=(80,), eni_roles=None):
    inbound_all = [r for r in records if r and external(r["src"]) and not external(r["dst"]) and r["dport"] is not None]
    replies = [r for r in inbound_all if is_reply(r)]
    inbound = [r for r in inbound_all if not is_reply(r)]
    by_port = defaultdict(lambda: {"flows": 0, "sources": set(), "accepted": 0, "rejected": 0})
    for r in inbound:
        p = by_port[(r["proto"], r["dport"])]
        p["flows"] += 1
        p["sources"].add(r["src"])
        p["accepted" if r["action"] == "ACCEPT" else "rejected"] += 1
    ports = sorted(({"proto": k[0], "port": k[1], "flows": v["flows"], "sources": len(v["sources"]),
                     "accepted": v["accepted"], "rejected": v["rejected"]} for k, v in by_port.items()),
                   key=lambda d: (-d["sources"], -d["flows"]))
    t0 = min((r["start"] for r in records if r), default=None)
    t1 = max((r["end"] for r in records if r), default=None)
    accepted_ports = sorted({d["port"] for d in ports if d["accepted"]})
    return {
        "window_minutes": round((t1 - t0) / 60, 1) if t0 else 0,
        "records": sum(1 for r in records if r), "inbound_external_flows": len(inbound),
        "reply_flows_excluded": len(replies),
        "external_sources": len({r["src"] for r in inbound}),
        "distinct_ports_probed": len(ports),
        "ports_with_accepted_inbound": accepted_ports,
        "unexpected_accepted_ports": [p for p in accepted_ports if p not in exposed_ports],
        "rejected_flows": sum(d["rejected"] for d in ports),
        "rejected_share": round(sum(d["rejected"] for d in ports) / max(1, len(inbound)), 4),
        "top_ports_by_sources": ports[:15],
        "by_interface": _by_interface(inbound, eni_roles or {}),
    }


def _by_interface(inbound, roles):
    out = defaultdict(lambda: {"flows": 0, "sources": set(), "accepted": 0})
    for r in inbound:
        k = roles.get(r["eni"], r["eni"])
        out[k]["flows"] += 1
        out[k]["sources"].add(r["src"])
        out[k]["accepted"] += r["action"] == "ACCEPT"
    return {k: {"flows": v["flows"], "sources": len(v["sources"]), "accepted": v["accepted"]} for k, v in out.items()}


def run(path):
    recs = [parse(json.loads(line)["message"]) for line in Path(path).open() if line.strip()]
    roles = {"eni-08ff0d987911648e4": "load_balancer", "eni-058aa7142e18defd6": "load_balancer",
             "eni-06d360bb285c52fc9": "ecs_task_public_ip", "eni-0417aefd9d4098a5d": "ecs_task_public_ip"}
    out = audit(recs, eni_roles=roles)
    (RES / "flowlog_audit.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2))
