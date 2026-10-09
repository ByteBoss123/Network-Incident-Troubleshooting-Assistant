"""Network configuration audit: cloud network config checked against what attackers actually probe.

Input 1: a read-only snapshot of AWS network configuration (VPCs, security groups, network interfaces,
load-balancer listeners) across all enabled regions, collected with `collect()` (Describe* calls only).
Input 2: the Suricata IDS flows already in DuckDB: which destination ports external sources probed and alerted on.

Checks:
  internet_exposed_rule   inbound rule open to 0.0.0.0/0 or ::/0 (port range, attached interfaces)
  probed_port_exposed     an internet-exposed port that external sources probed in the IDS data
  plaintext_listener      internet-facing load balancer with HTTP and no HTTPS listener
  unneeded_public_ip      interface with a public IP whose security groups admit no internet traffic
  unattached_sg           non-default security group attached to no interface (stale config)
"""
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "netconfig" / "aws_snapshot_2026-10-09.json"
DB = ROOT / "data" / "netincident.duckdb"
RES = ROOT / "results"
OPEN = {"0.0.0.0/0", "::/0"}


def collect(session, regions=None):  # pragma: no cover - needs AWS credentials
    """Read-only collector (boto3 session). Produces the snapshot shape used by audit()."""
    ec2 = session.client("ec2", region_name="us-east-1")
    regions = regions or [r["RegionName"] for r in ec2.describe_regions()["Regions"]]
    out = {"regions": []}
    for rn in regions:
        c = session.client("ec2", region_name=rn)
        sgs = c.describe_security_groups()["SecurityGroups"]
        enis = c.describe_network_interfaces()["NetworkInterfaces"]
        out["regions"].append({
            "region": rn,
            "vpcs": [{"default": v.get("IsDefault")} for v in c.describe_vpcs()["Vpcs"]],
            "security_groups": [{"id": g["GroupId"], "name": g["GroupName"], "rules": [{
                "proto": p.get("IpProtocol"), "from": p.get("FromPort"), "to": p.get("ToPort"),
                "cidrs": [r["CidrIp"] for r in p.get("IpRanges", [])]
                + [r["CidrIpv6"] for r in p.get("Ipv6Ranges", [])],
                "sg_refs": [r["GroupId"] for r in p.get("UserIdGroupPairs", [])],
                "prefix_lists": [x["PrefixListId"] for x in p.get("PrefixListIds", [])]} for p in g["IpPermissions"]]}
                for g in sgs],
            "enis": [{"id": n["NetworkInterfaceId"], "type": n.get("InterfaceType"),
                      "public_ip": bool(n.get("Association", {}).get("PublicIp")),
                      "sgs": [g["GroupId"] for g in n.get("Groups", [])]} for n in enis],
        })
    return out


def _ports(rule):
    if rule["proto"] == "-1" or rule["from"] is None:
        return (0, 65535)
    return (max(rule["from"], 0), max(rule["to"], 0))


def probed_ports(db=DB):
    con = duckdb.connect(str(db), read_only=True)
    df = con.execute("""SELECT dest_port, COUNT(*) AS flows, SUM(alerted::INT) AS alerted,
                               COUNT(DISTINCT src_ip) AS sources
                        FROM sec_flows WHERE external_src GROUP BY 1""").df()
    con.close()
    return {int(r.dest_port): {"flows": int(r.flows), "alerted": int(r.alerted), "sources": int(r.sources)}
            for r in df.itertuples()}


def audit(snapshot, probed):
    findings = []
    sg_index, attached = {}, set()
    for reg in snapshot["regions"]:
        for g in reg["security_groups"]:
            sg_index[g["id"]] = (reg["region"], g)
        for n in reg["enis"]:
            attached.update(n["sgs"])
    exposed_sgs = set()
    for sid, (region, g) in sg_index.items():
        for rule in g["rules"]:
            if OPEN & set(rule["cidrs"]):
                lo, hi = _ports(rule)
                exposed_sgs.add(sid)
                findings.append({"check": "internet_exposed_rule", "severity": "info", "region": region,
                                 "sg": f"{g['name']} ({sid})", "ports": f"{rule['proto']} {lo}-{hi}",
                                 "attached": sid in attached})
                hits = {p: v for p, v in probed.items() if lo <= p <= hi and v["alerted"] > 0}
                for p, v in sorted(hits.items(), key=lambda kv: -kv[1]["alerted"]):
                    findings.append({"check": "probed_port_exposed", "severity": "medium", "region": region,
                                     "sg": f"{g['name']} ({sid})", "port": p, **v})
    for reg in snapshot["regions"]:
        for lb in reg.get("load_balancers", []):
            protos = {x["protocol"] for x in lb["listeners"]}
            if lb["scheme"] == "internet-facing" and "HTTP" in protos and "HTTPS" not in protos:
                findings.append({"check": "plaintext_listener", "severity": "medium", "region": reg["region"],
                                 "load_balancer": lb["name"], "listeners": sorted(protos)})
        for n in reg["enis"]:
            if n["public_ip"] and not (set(n["sgs"]) & exposed_sgs):
                findings.append({"check": "unneeded_public_ip", "severity": "low", "region": reg["region"],
                                 "eni": n["id"], "type": n["type"]})
    for sid, (region, g) in sg_index.items():
        if g["name"] != "default" and sid not in attached:
            findings.append({"check": "unattached_sg", "severity": "low", "region": region,
                             "sg": f"{g['name']} ({sid})"})
    top_probed = sorted(probed.items(), key=lambda kv: -kv[1]["alerted"])[:10]
    exposed_ports = {f["port"] for f in findings if f["check"] == "probed_port_exposed"}
    summary = {
        "regions": len(snapshot["regions"]),
        "vpcs": sum(len(r["vpcs"]) for r in snapshot["regions"]),
        "security_groups": len(sg_index), "interfaces": sum(len(r["enis"]) for r in snapshot["regions"]),
        "internet_exposed_rules": sum(f["check"] == "internet_exposed_rule" for f in findings),
        "top10_alerted_ports": [p for p, _ in top_probed],
        "top10_alerted_ports_exposed": sorted(p for p, _ in top_probed if p in exposed_ports),
        "findings_by_severity": {s: sum(f["severity"] == s for f in findings) for s in ("medium", "low", "info")},
    }
    return {"summary": summary, "findings": findings}


def run():
    out = audit(json.loads(SNAP.read_text()), probed_ports())
    (RES / "netconfig_audit.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
