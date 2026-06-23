#!/usr/bin/env python3
"""Structural coverage analysis for the FD-include training set.

The FD-include antipattern requires an inclusion use case (the target of an
``<<include>>`` relationship) to satisfy ALL of the following. Violating any one
of them disqualifies it, so a use case that violates a criterion is a "hard
negative": a structure that looks include-related but must NOT be flagged.

  (i)   included by exactly one base use case
  (ii)  has no direct actor association
  (iii) neither includes nor extends any other use case
  (iv)  is not extended by any use case
  (v)   is not involved in any generalization

This script scans every ``*.puml`` sample and reports how many samples and
domains contain each disqualifying structure. Sparse coverage of a structure
predicts false positives on real diagrams that contain it (see the MAPSTEDI
case study), and points the data-generation pipeline at structures to oversample.

Usage:
    python scripts/analyze_hard_negatives.py [RUN_DIR]
    # default RUN_DIR: output/run_20260325_012118
"""
import os
import re
import sys
import glob
from collections import defaultdict

INCLUDE = re.compile(r'^\s*([A-Za-z0-9_"]+)\s*\.?\.>\s*([A-Za-z0-9_"]+)\s*:\s*<<include>>')
EXTEND  = re.compile(r'^\s*([A-Za-z0-9_"]+)\s*\.?\.>\s*([A-Za-z0-9_"]+)\s*:\s*<<extend>>')
ACTOR   = re.compile(r'^\s*actor\s+"[^"]*"\s+as\s+(\w+)')
USECASE = re.compile(r'^\s*usecase\s+"([^"]+)"\s+as\s+(\w+)')
GENERAL = re.compile(r'^\s*([A-Za-z0-9_"]+)\s*(?:<\|--|--\|>)\s*([A-Za-z0-9_"]+)')
ASSOC   = re.compile(r'^\s*([A-Za-z0-9_"]+)\s*--\s*([A-Za-z0-9_"]+)\s*$')


def parse(path):
    actors, usecases, names = set(), set(), {}
    includes, extends, assocs, generals = [], [], [], []
    for line in open(path, encoding="utf-8"):
        m = ACTOR.match(line)
        if m:
            actors.add(m.group(1)); continue
        m = USECASE.match(line)
        if m:
            usecases.add(m.group(2)); names[m.group(2)] = m.group(1); continue
        m = INCLUDE.match(line)
        if m:
            includes.append((m.group(1), m.group(2))); continue
        m = EXTEND.match(line)
        if m:
            extends.append((m.group(1), m.group(2))); continue
        m = GENERAL.match(line)
        if m:
            generals.append((m.group(1), m.group(2))); continue
        m = ASSOC.match(line)
        if m:
            assocs.append((m.group(1), m.group(2)))
    return dict(actors=actors, usecases=usecases, names=names,
                includes=includes, extends=extends, assocs=assocs, generals=generals)


def disqualifiers(d):
    """Return {criterion: set(inclusion use cases violating it)} for one diagram."""
    inc_targets = {t for _, t in d["includes"]}        # inclusion use cases
    inc_sources = {s for s, _ in d["includes"]}
    ext_sources = {s for s, _ in d["extends"]}
    ext_targets = {t for _, t in d["extends"]}
    gen_nodes   = {x for pair in d["generals"] for x in pair}

    base_count = defaultdict(set)
    for s, t in d["includes"]:
        base_count[t].add(s)

    actor_assoc = set()
    for a, b in d["assocs"]:
        if a in d["actors"] and b in d["usecases"]:
            actor_assoc.add(b)
        if b in d["actors"] and a in d["usecases"]:
            actor_assoc.add(a)

    out = {
        "shared_inclusion (>=2 bases)":     {t for t in inc_targets if len(base_count[t]) >= 2},
        "actor_associated_inclusion":       {t for t in inc_targets if t in actor_assoc},
        "inclusion_that_includes_another":  {t for t in inc_targets if t in inc_sources},
        "inclusion_that_extends_another":   {t for t in inc_targets if t in ext_sources},
        "inclusion_extended_by_another":    {t for t in inc_targets if t in ext_targets},
        "inclusion_in_generalization":      {t for t in inc_targets if t in gen_nodes},
    }
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "output/run_20260325_012118"
    files = sorted(glob.glob(os.path.join(run, "domains", "*", "*.puml")))
    if not files:
        sys.exit(f"no .puml files under {run}/domains/*/")

    keys = ["shared_inclusion (>=2 bases)", "actor_associated_inclusion",
            "inclusion_that_includes_another", "inclusion_that_extends_another",
            "inclusion_extended_by_another", "inclusion_in_generalization"]
    sample_hits = defaultdict(int)
    domain_hits = defaultdict(set)
    examples    = defaultdict(list)

    for f in files:
        d = parse(f)
        dq = disqualifiers(d)
        dom = os.path.basename(os.path.dirname(f))
        for k in keys:
            if dq[k]:
                sample_hits[k] += 1
                domain_hits[k].add(dom)
                if len(examples[k]) < 5:
                    label = ", ".join(d["names"].get(x, x) for x in sorted(dq[k]))
                    examples[k].append(f"{dom}/{os.path.basename(f)}: {label}")

    n = len(files)
    ndom = len({os.path.basename(os.path.dirname(f)) for f in files})
    print(f"Scanned {n} samples across {ndom} domains in {run}\n")
    print(f"{'disqualifying structure':38} {'samples':>8} {'domains':>8}")
    print("-" * 58)
    for k in keys:
        print(f"{k:38} {sample_hits[k]:>4}/{n:<3} {len(domain_hits[k]):>4}/{ndom:<3}")
    print("\nExamples (up to 5 per structure):")
    for k in keys:
        print(f"\n  {k}:")
        for e in examples[k] or ["    (none)"]:
            print(f"    {e}")


if __name__ == "__main__":
    main()
