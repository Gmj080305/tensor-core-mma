#!/usr/bin/env python3
"""Collect timing / area / power from finished ORFS runs into a Markdown table.

Looks in <work>/{logs,reports}/<pdk>/<design>/<variant>/, where <work> is
ORFS's WORK_HOME (or $ORFS/flow when WORK_HOME is not set). It first reads
the metrics JSON files ORFS writes (keys such as finish__timing__setup__ws,
finish__design__instance__area, finish__power__total); if none are found it
falls back to parsing 6_finish.rpt. ORFS report formats change between
releases: if a column comes out empty, open the report and adjust the regex.
"""
import argparse
import glob
import json
import os
import re


def from_json(variant_dirs):
    found = {}
    for d in variant_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                data = json.load(open(path))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for key, val in data.items():
                k = key.lower()
                if "finish" not in k:
                    continue
                if "timing__setup__ws" in k:
                    found.setdefault("wns", val)
                elif "design__instance__area" == k.split("finish__")[-1]:
                    found.setdefault("area", val)
                elif "power__total" in k:
                    found.setdefault("power", val)
    return found


def from_report(report_dir):
    found = {}
    rpt = os.path.join(report_dir, "6_finish.rpt")
    if not os.path.exists(rpt):
        return found
    txt = open(rpt).read()
    m = re.search(r"worst slack(?: max)?\s+(-?[\d.]+)", txt)
    if m:
        found["wns"] = float(m.group(1))
    m = re.search(r"Design area\s+([\d.]+)\s+u\^2", txt)
    if m:
        found["area"] = float(m.group(1))
    m = re.search(r"^Total\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)", txt, re.M)
    if m:
        found["power"] = float(m.group(1))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="ORFS WORK_HOME (or $ORFS/flow)")
    ap.add_argument("--pdk", required=True)
    ap.add_argument("--design", default="tc_fda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    unit = "ps" if a.pdk == "asap7" else "ns"
    base = a.work
    rows = []
    for rdir in sorted(glob.glob(os.path.join(base, "reports", a.pdk, a.design, "*"))):
        variant = os.path.basename(rdir)
        ldir = os.path.join(base, "logs", a.pdk, a.design, variant)
        m = from_json([ldir, rdir])
        for k, v in from_report(rdir).items():
            m.setdefault(k, v)
        rows.append((variant, m))
    with open(a.out, "w") as o:
        o.write("# ORFS results: one FDA lane, %s\n\n" % a.pdk)
        o.write("| variant | worst setup slack (%s) | std-cell area (um^2) | total power (W) |\n" % unit)
        o.write("|---|---|---|---|\n")
        for variant, m in rows:
            o.write("| %s | %s | %s | %s |\n" % (variant, m.get("wns", ""), m.get("area", ""), m.get("power", "")))
        o.write("\nAchievable clock period = constraint period - worst slack.\n")
    print("wrote %s (%d variants)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
