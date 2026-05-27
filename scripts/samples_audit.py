#!/usr/bin/env python3
"""Verify all samples in a run's samples.jsonl against source files,
and verify .puml / .jinja / .yaml files against domain audit logs.

Checks each sample's three messages:
  1. system  — must match _TRAINING_SYSTEM_PROMPT from generate_samples.py
  2. user    — must match task instruction + PlantUML source from the .puml file
  3. assistant — must match the JSON answer extracted from the .jinja file

Usage:
    python samples_audit.py [run_dir]

    run_dir defaults to output/run_20260325_012118 relative to this script.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

# ── Constants (duplicated from generate_samples.py) ───────────────────────────

_TRAINING_SYSTEM_PROMPT = (
    "You are an expert in UML use case diagram analysis. You detect antipatterns in "
    "PlantUML models. Analyze the provided model and respond with a JSON object using "
    "this exact structure:\n"
    "{\n"
    '  "detected": boolean,\n'
    '  "total_antipattern_types": number,\n'
    '  "total_instances": number,\n'
    '  "antipatterns": [\n'
    '    {\n'
    '      "antipattern_name": string,\n'
    '      "instance_count": number,\n'
    '      "instances": [\n'
    '        {\n'
    '          "elements": [string],\n'
    '          "explanation": string\n'
    '        }\n'
    '      ]\n'
    '    }\n'
    '  ]\n'
    "}"
)

_TASK_INSTRUCTION = {
    "detect": (
        "Analyze the following PlantUML use case model and detect if it contains "
        "any antipatterns.\n"
        "Base your analysis solely on the model provided. "
        "Do not use any outside knowledge.\n"
        "Respond with a JSON object."
    ),
    "detect-and-refactor": (
        "Analyze the following PlantUML use case model, detect if it contains "
        "any antipatterns,\n"
        "and if an antipattern is found, provide the full refactored PlantUML model.\n"
        "Base your analysis solely on the model provided. "
        "Do not use any outside knowledge.\n"
        "Respond with a JSON object, followed by the refactored PlantUML model if applicable."
    ),
}

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_answer_from_jinja(jinja_content: str) -> str | None:
    """Return the JSON answer string from a .jinja file.

    Handles both plain JSON and ```json ... ``` code-fenced variants.
    Returns None if the Answer section cannot be found.
    """
    marker = "Answer:\n"
    idx = jinja_content.find(marker)
    if idx == -1:
        return None
    raw = jinja_content[idx + len(marker):]

    if raw.startswith("```json\n"):
        raw = raw[len("```json\n"):]
        fence_end = raw.find("\n```")
        if fence_end != -1:
            raw = raw[:fence_end]
    elif raw.startswith("```\n"):
        raw = raw[len("```\n"):]
        fence_end = raw.find("\n```")
        if fence_end != -1:
            raw = raw[:fence_end]

    return raw.strip()


def extract_puml_from_jinja(jinja_content: str) -> str | None:
    """Return the PlantUML source from a .jinja file (between 'PlantUML Model:' and 'Answer:')."""
    prefix = "PlantUML Model:\n"
    idx = jinja_content.find(prefix)
    if idx == -1:
        return None
    rest = jinja_content[idx + len(prefix):]
    answer_idx = rest.find("\nAnswer:\n")
    if answer_idx == -1:
        return None
    return rest[:answer_idx].strip()


# ── Audit parsing ─────────────────────────────────────────────────────────────


def _extract_json_from_block(text: str) -> str | None:
    """Extract a JSON object from text that is either fenced (```json) or inline ({...)."""
    s = text.strip()
    if s.startswith("```json\n"):
        inner = s[len("```json\n"):]
        end = inner.find("\n```")
        return inner[:end].strip() if end != -1 else inner.strip()
    if s.startswith("```\n"):
        inner = s[len("```\n"):]
        end = inner.find("\n```")
        return inner[:end].strip() if end != -1 else inner.strip()
    if s.startswith("{"):
        return s
    return None


def parse_audit_file(audit_path: Path) -> dict:
    """Parse an audit file, returning the latest expected values for each artifact.

    Iterates all prompt–response pairs in the audit (including REPROCESS sections)
    in order, so later pairs overwrite earlier ones.  PlantUML blocks are only
    updated by full-generation pairs; JSON is updated by every pair that contains one.

    Returns a dict with keys:
        v1_puml        – last VERSION 1 PlantUML (@startuml...@enduml, stripped)
        v2_puml        – last VERSION 2 PlantUML
        ap_json        – last antipattern JSON string
        domain         – snake_case domain from last full generation
        domain_display – human-readable domain
        size           – small / medium / large
        construct_count – integer from CONSTRUCT_COUNT line
    """
    text = audit_path.read_text(encoding="utf-8")

    # Each pair after the first is preceded by a line of ═ characters
    # (optionally containing "REPROCESS — ..." text).
    sections = re.split(r"\n═{10,}[^\n]*\n", text)

    result: dict = {
        "v1_puml": None,
        "v2_puml": None,
        "ap_json": None,
        "domain": None,
        "domain_display": None,
        "size": None,
        "construct_count": None,
    }

    for section in sections:
        resp_marker = "ASSISTANT RESPONSE (raw)\n"
        idx = section.find(resp_marker)
        if idx == -1:
            continue

        after = section[idx + len(resp_marker):]
        # Strip the ────...──── separator line that follows the marker
        after = re.sub(r"^─{10,}\n", "", after)
        response = after.strip()

        if not response:
            continue

        is_generation = "=== VERSION 1:" in response

        if is_generation:
            # Extract header metadata
            for key, pattern in [
                ("domain",         r"^DOMAIN:\s*(.+)$"),
                ("domain_display", r"^DOMAIN_DISPLAY:\s*(.+)$"),
                ("size",           r"^SIZE:\s*(.+)$"),
            ]:
                m = re.search(pattern, response, re.MULTILINE)
                if m:
                    result[key] = m.group(1).strip()

            m = re.search(r"^CONSTRUCT_COUNT:\s*(\d+)$", response, re.MULTILINE)
            if m:
                result["construct_count"] = int(m.group(1))

            # Split at the VERSION 2 header
            v2_idx = response.find("=== VERSION 2:")
            v1_section = response[:v2_idx] if v2_idx != -1 else response
            v2_section = response[v2_idx:] if v2_idx != -1 else ""

            # Extract VERSION 1 PlantUML and the JSON that follows it
            m = re.search(r"```plantuml\s*\n(.*?)```", v1_section, re.DOTALL)
            if m:
                result["v1_puml"] = m.group(1).strip()
                after_puml = v1_section[m.end():].strip()
                extracted = _extract_json_from_block(after_puml)
                if extracted:
                    result["ap_json"] = extracted

            # Extract VERSION 2 PlantUML
            m = re.search(r"```plantuml\s*\n(.*?)```", v2_section, re.DOTALL)
            if m:
                result["v2_puml"] = m.group(1).strip()

        else:
            # REPROCESS — JSON only
            extracted = _extract_json_from_block(response)
            if extracted:
                result["ap_json"] = extracted

    return result


# ── Audit verification ────────────────────────────────────────────────────────


def verify_domain_audit(
    domain_str: str, run_dir: Path, skip_json: bool = False
) -> tuple[list[str], list[str]]:
    """Verify .puml / .jinja / .yaml files for a domain against its audit log.

    skip_json=True skips the JSON comparison between .jinja / .yaml and the audit
    log.  Use this after repair_training_samples.py has intentionally changed the
    answer JSON (the audit log records the original pre-repair generation).

    Returns ``(errors, warnings)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    domain_dir = run_dir / "domains" / domain_str
    audit_file = domain_dir / f"{domain_str}_audit.txt"

    if not audit_file.exists():
        return [f"domain {domain_str}: no audit file"], []

    expected = parse_audit_file(audit_file)

    # ── _ap files ─────────────────────────────────────────────────────────────
    ap_puml_path  = domain_dir / f"{domain_str}_ap.puml"
    ap_jinja_path = domain_dir / f"{domain_str}_ap.jinja"
    ap_yaml_path  = domain_dir / f"{domain_str}_ap.yaml"

    ap_jinja_content: str | None = None
    if ap_jinja_path.exists():
        ap_jinja_content = ap_jinja_path.read_text(encoding="utf-8")

    # PlantUML checks
    if expected["v1_puml"] is None:
        errors.append(f"domain {domain_str}: could not extract VERSION 1 PlantUML from audit")
    else:
        if ap_puml_path.exists():
            actual = ap_puml_path.read_text(encoding="utf-8").strip()
            if actual != expected["v1_puml"]:
                errors.append(f"domain {domain_str}: _ap.puml does not match audit VERSION 1 PlantUML")
        else:
            errors.append(f"domain {domain_str}: _ap.puml not found")

        if ap_jinja_content is not None:
            jinja_puml = extract_puml_from_jinja(ap_jinja_content)
            if jinja_puml is None:
                errors.append(f"domain {domain_str}: could not extract PlantUML from _ap.jinja")
            elif jinja_puml != expected["v1_puml"]:
                errors.append(f"domain {domain_str}: _ap.jinja PlantUML input does not match audit VERSION 1")
        elif ap_jinja_path.exists() is False:
            errors.append(f"domain {domain_str}: _ap.jinja not found")

        # Orphan use case check on VERSION 1
        for alias, name in _find_orphan_use_cases(expected["v1_puml"]):
            errors.append(
                f"domain {domain_str}: _ap.puml orphan use case '{name}' ({alias}) — "
                f"no actor association and not reachable via any relationship"
            )

    # JSON / output checks
    expected_json_obj: object | None = None
    if expected["ap_json"] is None:
        errors.append(f"domain {domain_str}: could not extract antipattern JSON from audit")
    else:
        try:
            expected_json_obj = json.loads(expected["ap_json"])
        except json.JSONDecodeError:
            errors.append(f"domain {domain_str}: audit antipattern JSON is not valid JSON")

        if not skip_json and ap_jinja_content is not None:
            jinja_answer = extract_answer_from_jinja(ap_jinja_content)
            if jinja_answer is None:
                errors.append(f"domain {domain_str}: could not find Answer section in _ap.jinja")
            else:
                try:
                    jinja_json_obj = json.loads(jinja_answer)
                except json.JSONDecodeError:
                    jinja_json_obj = None
                    errors.append(f"domain {domain_str}: _ap.jinja Answer is not valid JSON")
                else:
                    if expected_json_obj is not None and jinja_json_obj != expected_json_obj:
                        errors.append(f"domain {domain_str}: _ap.jinja Answer JSON does not match audit")

    # Always validate the .jinja Answer for internal consistency (even when skip_json=True)
    if ap_jinja_content is not None:
        jinja_answer = extract_answer_from_jinja(ap_jinja_content)
        if jinja_answer is not None:
            errs, warns = verify_output_json_consistency(
                f"domain {domain_str} (_ap.jinja)", jinja_answer,
                puml_source=expected.get("v1_puml"),
            )
            errors.extend(errs)
            warnings.extend(warns)

    # YAML checks
    if ap_yaml_path.exists():
        yaml_data = yaml.safe_load(ap_yaml_path.read_text(encoding="utf-8"))
        ts = yaml_data.get("training_sample", {})
        meta = yaml_data.get("metadata", {})

        # Output JSON — against audit (unless skipped) and always for internal consistency
        yaml_output = ts.get("output", "")
        if not skip_json and expected["ap_json"] is not None and expected_json_obj is not None:
            try:
                yaml_output_obj = json.loads(yaml_output)
            except (json.JSONDecodeError, TypeError):
                yaml_output_obj = None
                errors.append(f"domain {domain_str}: _ap.yaml training_sample.output is not valid JSON")
            else:
                if yaml_output_obj != expected_json_obj:
                    errors.append(f"domain {domain_str}: _ap.yaml training_sample.output does not match audit JSON")
        if yaml_output:
            errs, warns = verify_output_json_consistency(
                f"domain {domain_str} (_ap.yaml)", yaml_output,
                puml_source=expected.get("v1_puml"),
            )
            errors.extend(errs)
            warnings.extend(warns)

        # Input PlantUML (check containment — the input also has the task instruction prefix)
        if expected["v1_puml"] is not None:
            yaml_input = ts.get("input", "")
            if expected["v1_puml"] not in yaml_input:
                errors.append(
                    f"domain {domain_str}: _ap.yaml training_sample.input does not contain audit VERSION 1 PlantUML"
                )

        # Metadata fields
        for field, key in [("domain", "domain"), ("domain_display", "domain_display"), ("size", "size")]:
            if expected[key] is not None and meta.get(field) != expected[key]:
                errors.append(
                    f'domain {domain_str}: _ap.yaml metadata.{field} '
                    f'("{meta.get(field)}") does not match audit ("{expected[key]}")'
                )
        if expected["construct_count"] is not None and meta.get("construct_count") != expected["construct_count"]:
            errors.append(
                f"domain {domain_str}: _ap.yaml metadata.construct_count "
                f"({meta.get('construct_count')}) does not match audit ({expected['construct_count']})"
            )
    else:
        errors.append(f"domain {domain_str}: _ap.yaml not found")

    # ── _re files (only if any exist) ─────────────────────────────────────────
    re_puml_path  = domain_dir / f"{domain_str}_re.puml"
    re_jinja_path = domain_dir / f"{domain_str}_re.jinja"
    re_yaml_path  = domain_dir / f"{domain_str}_re.yaml"

    has_re = re_puml_path.exists() or re_jinja_path.exists() or re_yaml_path.exists()
    if not has_re:
        return errors, warnings

    if expected["v2_puml"] is None:
        errors.append(
            f"domain {domain_str}: _re files exist but could not extract VERSION 2 PlantUML from audit"
        )
        return errors, warnings

    re_jinja_content: str | None = None
    if re_jinja_path.exists():
        re_jinja_content = re_jinja_path.read_text(encoding="utf-8")

    # _re.puml
    if re_puml_path.exists():
        actual_re = re_puml_path.read_text(encoding="utf-8").strip()
        if actual_re != expected["v2_puml"]:
            errors.append(f"domain {domain_str}: _re.puml does not match audit VERSION 2 PlantUML")
    else:
        errors.append(f"domain {domain_str}: _re.puml not found")

    # Orphan use case check on VERSION 2
    for alias, name in _find_orphan_use_cases(expected["v2_puml"]):
        errors.append(
            f"domain {domain_str}: _re.puml orphan use case '{name}' ({alias}) — "
            f"no actor association and not reachable via any relationship"
        )

    # _re.jinja PlantUML input
    if re_jinja_content is not None:
        re_jinja_puml = extract_puml_from_jinja(re_jinja_content)
        if re_jinja_puml is None:
            errors.append(f"domain {domain_str}: could not extract PlantUML from _re.jinja")
        elif re_jinja_puml != expected["v2_puml"]:
            errors.append(f"domain {domain_str}: _re.jinja PlantUML input does not match audit VERSION 2")

    # _re.yaml
    if re_yaml_path.exists():
        re_yaml_data = yaml.safe_load(re_yaml_path.read_text(encoding="utf-8"))
        re_ts   = re_yaml_data.get("training_sample", {})
        re_meta = re_yaml_data.get("metadata", {})

        re_yaml_input = re_ts.get("input", "")
        if expected["v2_puml"] not in re_yaml_input:
            errors.append(
                f"domain {domain_str}: _re.yaml training_sample.input does not contain audit VERSION 2 PlantUML"
            )

        for field, key in [("domain", "domain"), ("domain_display", "domain_display"), ("size", "size")]:
            if expected[key] is not None and re_meta.get(field) != expected[key]:
                errors.append(
                    f'domain {domain_str}: _re.yaml metadata.{field} '
                    f'("{re_meta.get(field)}") does not match audit ("{expected[key]}")'
                )
    else:
        errors.append(f"domain {domain_str}: _re.yaml not found")

    # ── Refactoring scope check ───────────────────────────────────────────────
    # Requires: both .puml files exist, and the jinja answer JSON is parseable.
    if ap_puml_path.exists() and re_puml_path.exists() and ap_jinja_content is not None:
        jinja_answer_for_scope = extract_answer_from_jinja(ap_jinja_content)
        if jinja_answer_for_scope is not None:
            try:
                scope_json = json.loads(jinja_answer_for_scope)
            except json.JSONDecodeError:
                scope_json = None
            if scope_json is not None:
                errors.extend(
                    verify_refactoring_scope(
                        domain_str,
                        ap_puml_path.read_text(encoding="utf-8"),
                        re_puml_path.read_text(encoding="utf-8"),
                        scope_json,
                    )
                )

    return errors, warnings


# ── Explanation template helpers (mirrored from fix_explanations.py) ─────────


def _build_uc_map(plantuml_source: str) -> dict[str, str]:
    """Return {alias: name} from 'usecase "Name" as Alias' lines."""
    return {
        alias.strip(): name.strip()
        for name, alias in re.findall(r'usecase\s+"([^"]+)"\s+as\s+(\S+)', plantuml_source)
    }


def _build_include_rels(plantuml_source: str) -> set[tuple[str, str]]:
    """Return {(base_alias, inclusion_alias)} for all include arrows (.> or ..>)."""
    return set(re.findall(r'(\S+)\s+\.\.?\>\s+(\S+)', plantuml_source))


def _build_actor_ucs(plantuml_source: str) -> set[str]:
    """Return the set of UC aliases that have a direct actor association.

    Handles both quoted-with-alias actors ('actor "Name" as Alias') and bare
    actors ('actor Name') that use the name itself as the alias.
    """
    actors: set[str] = set()
    # 'actor "Full Name" as Alias'  →  alias
    actors |= set(re.findall(r'actor\s+"[^"]+"\s+as\s+(\S+)', plantuml_source))
    # 'actor Name as Alias'         →  alias  (unquoted name, explicit alias)
    actors |= set(re.findall(r'^actor\s+\w+\s+as\s+(\S+)', plantuml_source, re.MULTILINE))
    # 'actor Name'                  →  Name   (bare, name IS the alias)
    actors |= set(re.findall(r'^actor\s+(\w+)\s*$', plantuml_source, re.MULTILINE))
    ucs_with_actor: set[str] = set()
    for a, b in re.findall(r'(\w+)\s+--\s+(\w+)', plantuml_source):
        if a in actors:
            ucs_with_actor.add(b)
        if b in actors:
            ucs_with_actor.add(a)
    return ucs_with_actor


def _find_orphan_use_cases(plantuml_source: str) -> list[tuple[str, str]]:
    """Return (alias, name) pairs for use cases with no actor and no incoming relationship.

    A use case is an orphan if it has no direct actor association AND is not
    reachable through any relationship:
      • not a target of an <<include>> arrow  (inclusion UCs are intentionally actor-free
        in the antipattern, but they're still reachable from an actor via the base UC)
      • not a source of an <<extend>> arrow   (extension UCs trigger off a base UC)
      • not a member of a generalisation      (child/parent UCs share actor coverage)

    UC4 "Record Diagnosis" in domain 017 is the canonical example: declared in the
    model but connected to nothing — no actor, no arrows in or out.
    """
    uc_map = _build_uc_map(plantuml_source)
    if not uc_map:
        return []

    actor_ucs = _build_actor_ucs(plantuml_source)

    # Targets of include arrows (base ..> inclusion [:<<include>>])
    include_targets = {incl for _, incl in _build_include_rels(plantuml_source)}

    # Sources of extend arrows (extension ..> base : <<extend>>)
    extend_sources = set(re.findall(
        r'(\S+)\s+\.\.?\>\s+\S+\s*:\s*<<extend>>', plantuml_source
    ))

    # Members of any generalisation (child --|> parent)
    gen_members: set[str] = set()
    for a, b in re.findall(r'(\w+)\s+--\|>\s+(\w+)', plantuml_source):
        gen_members.add(a)
        gen_members.add(b)

    reachable = actor_ucs | include_targets | extend_sources | gen_members
    return [
        (alias, uc_map[alias])
        for alias in sorted(uc_map)
        if alias not in reachable
    ]


# ── Refactoring-scope helpers ─────────────────────────────────────────────────


def _parse_actors(plantuml_source: str) -> dict[str, str]:
    """Return {alias: display_name} for every actor declaration."""
    actors: dict[str, str] = {}
    # actor "Full Name" as Alias
    for name, alias in re.findall(r'actor\s+"([^"]+)"\s+as\s+(\S+)', plantuml_source):
        actors[alias] = name
    # actor Name as Alias  (unquoted name, explicit alias)
    for m in re.finditer(r'^actor\s+(\w+)\s+as\s+(\S+)', plantuml_source, re.MULTILINE):
        actors.setdefault(m.group(2), m.group(1))
    # actor Name           (bare – name IS the alias)
    for m in re.finditer(r'^actor\s+(\w+)\s*$', plantuml_source, re.MULTILINE):
        actors.setdefault(m.group(1), m.group(1))
    return actors


def _parse_assocs(
    plantuml_source: str,
    actor_aliases: set[str],
    uc_aliases: set[str],
) -> set[tuple[str, str]]:
    """Return {(actor_alias, uc_alias)} for every undirected -- association."""
    assocs: set[tuple[str, str]] = set()
    for a, b in re.findall(r'(\w+)\s+--\s+(\w+)', plantuml_source):
        if a in actor_aliases and b in uc_aliases:
            assocs.add((a, b))
        elif b in actor_aliases and a in uc_aliases:
            assocs.add((b, a))
    return assocs


def _parse_tagged_include_rels(plantuml_source: str) -> set[tuple[str, str]]:
    """Return {(base_alias, incl_alias)} for arrows explicitly tagged <<include>>."""
    return set(re.findall(r'(\S+)\s+\.\.?\>\s+(\S+)\s*:\s*<<include>>', plantuml_source))


def _parse_tagged_extend_rels(plantuml_source: str) -> set[tuple[str, str]]:
    """Return {(ext_alias, base_alias)} for arrows explicitly tagged <<extend>>."""
    return set(re.findall(r'(\S+)\s+\.\.?\>\s+(\S+)\s*:\s*<<extend>>', plantuml_source))


def _parse_gen_rels(plantuml_source: str) -> set[tuple[str, str]]:
    """Return {(child_alias, parent_alias)} for generalisation (--|>) arrows."""
    return set(re.findall(r'(\w+)\s+--\|>\s+(\w+)', plantuml_source))


def verify_refactoring_scope(
    domain_str: str,
    ap_puml: str,
    re_puml: str,
    ap_json_data: dict,
) -> list[str]:
    """Verify that the refactored model is exactly the antipattern model with
    the antipattern inclusion use cases (and their <<include>> relationships)
    dropped and *nothing else changed*.

    For "Functional Decomposition: Using the include relationship" the refactoring
    rule is "Drop Functional Decomposition": remove inclusion UCs that have no
    direct actor association.  All actors, all other use cases, all other
    associations, extends, and generalisations must be identical in both versions.

    Checks
    ──────
    Actors       : same set, same display names
    Use cases    : VERSION 2 == VERSION 1 − dropped_inclusion_UCs
    Associations : VERSION 2 == VERSION 1 − assocs involving dropped UCs
                   (no new associations may be introduced)
    <<include>>  : VERSION 2 == VERSION 1 − antipattern include arrows
    <<extend>>   : identical in both versions
    Generalise   : identical in both versions
    UC names     : retained UCs must keep the same display name
    """
    errors: list[str] = []
    prefix = f"domain {domain_str}"

    # ── Parse VERSION 1 ───────────────────────────────────────────────────────
    v1_uc_map   = _build_uc_map(ap_puml)
    v1_actors   = _parse_actors(ap_puml)
    v1_actor_set = set(v1_actors)
    v1_uc_set   = set(v1_uc_map)
    v1_assocs   = _parse_assocs(ap_puml, v1_actor_set, v1_uc_set)
    v1_includes = _parse_tagged_include_rels(ap_puml)
    v1_extends  = _parse_tagged_extend_rels(ap_puml)
    v1_gens     = _parse_gen_rels(ap_puml)

    # ── Parse VERSION 2 ───────────────────────────────────────────────────────
    v2_uc_map   = _build_uc_map(re_puml)
    v2_actors   = _parse_actors(re_puml)
    v2_actor_set = set(v2_actors)
    v2_uc_set   = set(v2_uc_map)
    v2_assocs   = _parse_assocs(re_puml, v2_actor_set, v2_uc_set)
    v2_includes = _parse_tagged_include_rels(re_puml)
    v2_extends  = _parse_tagged_extend_rels(re_puml)
    v2_gens     = _parse_gen_rels(re_puml)

    # ── Collect dropped UC aliases from the antipattern JSON ─────────────────
    # Elements may be formatted as "Use Case Name (UCx)" OR just "Use Case Name".
    # Build a reverse name→alias map to handle the name-only form.
    v1_name_to_alias = {name: alias for alias, name in v1_uc_map.items()}

    dropped_uc_aliases: set[str] = set()
    for ap in ap_json_data.get("antipatterns", []):
        if "Functional Decomposition" in ap.get("antipattern_name", ""):
            for inst in ap.get("instances", []):
                elems = inst.get("elements", [])
                if len(elems) >= 2:
                    incl_name, incl_alias = _resolve_uc(elems[1], v1_uc_map)
                    if incl_alias is None:
                        incl_alias = v1_name_to_alias.get(incl_name)
                    if incl_alias:
                        dropped_uc_aliases.add(incl_alias)

    # ── Compute expected VERSION 2 ────────────────────────────────────────────
    exp_uc_set   = v1_uc_set - dropped_uc_aliases
    exp_assocs   = {(a, uc) for a, uc in v1_assocs
                   if uc not in dropped_uc_aliases}
    exp_includes = {(b, i) for b, i in v1_includes
                   if i not in dropped_uc_aliases and b not in dropped_uc_aliases}
    # Extends and generalisations must be fully preserved
    exp_extends = v1_extends
    exp_gens    = v1_gens

    # ── Actor checks ──────────────────────────────────────────────────────────
    for alias in sorted(v1_actor_set - v2_actor_set):
        errors.append(f"{prefix}: refactored model removed actor "
                      f"'{v1_actors[alias]}' ({alias})")
    for alias in sorted(v2_actor_set - v1_actor_set):
        errors.append(f"{prefix}: refactored model introduced new actor "
                      f"'{v2_actors.get(alias, alias)}' ({alias})")
    for alias in sorted(v1_actor_set & v2_actor_set):
        if v1_actors[alias] != v2_actors[alias]:
            errors.append(f"{prefix}: actor {alias} renamed "
                          f"'{v1_actors[alias]}' → '{v2_actors[alias]}'")

    # ── Use-case checks ───────────────────────────────────────────────────────
    # Antipattern inclusion UCs that were NOT removed
    for alias in sorted(dropped_uc_aliases & v2_uc_set):
        errors.append(f"{prefix}: antipattern inclusion use case "
                      f"'{v2_uc_map[alias]}' ({alias}) not removed in refactored model")
    # Non-antipattern UCs that were incorrectly removed
    for alias in sorted(exp_uc_set - v2_uc_set):
        errors.append(f"{prefix}: refactored model incorrectly removed use case "
                      f"'{v1_uc_map[alias]}' ({alias})")
    # Brand-new UCs that did not exist in VERSION 1
    for alias in sorted(v2_uc_set - v1_uc_set):
        errors.append(f"{prefix}: refactored model introduced new use case "
                      f"'{v2_uc_map.get(alias, alias)}' ({alias})")
    # Retained UCs with a changed display name
    for alias in sorted(exp_uc_set & v2_uc_set):
        if v1_uc_map[alias] != v2_uc_map[alias]:
            errors.append(f"{prefix}: use case {alias} renamed "
                          f"'{v1_uc_map[alias]}' → '{v2_uc_map[alias]}'")

    # ── Association checks ────────────────────────────────────────────────────
    for actor_a, uc_a in sorted(exp_assocs - v2_assocs):
        errors.append(f"{prefix}: refactored model removed association "
                      f"{actor_a}--{uc_a}")
    for actor_a, uc_a in sorted(v2_assocs - exp_assocs):
        errors.append(f"{prefix}: refactored model introduced new association "
                      f"{actor_a}--{uc_a} not present in original")

    # ── <<include>> checks ────────────────────────────────────────────────────
    for base, incl in sorted(exp_includes - v2_includes):
        errors.append(f"{prefix}: refactored model removed include "
                      f"{base}→{incl} which was not an antipattern element")
    for base, incl in sorted(v2_includes - exp_includes):
        if incl in dropped_uc_aliases or base in dropped_uc_aliases:
            errors.append(f"{prefix}: refactored model still contains antipattern "
                          f"include {base}→{incl}")
        else:
            errors.append(f"{prefix}: refactored model introduced new include "
                          f"{base}→{incl} not in original")

    # ── <<extend>> checks ─────────────────────────────────────────────────────
    for ext, base in sorted(exp_extends - v2_extends):
        errors.append(f"{prefix}: refactored model removed extend {ext}→{base}")
    for ext, base in sorted(v2_extends - exp_extends):
        errors.append(f"{prefix}: refactored model introduced new extend "
                      f"{ext}→{base} not in original")

    # ── Generalisation checks ─────────────────────────────────────────────────
    for child, parent in sorted(exp_gens - v2_gens):
        errors.append(f"{prefix}: refactored model removed generalisation "
                      f"{child}→{parent}")
    for child, parent in sorted(v2_gens - exp_gens):
        errors.append(f"{prefix}: refactored model introduced new generalisation "
                      f"{child}→{parent} not in original")

    return errors


def _resolve_uc(element: str, uc_map: dict[str, str]) -> tuple[str, str | None]:
    """Parse an element string into (name, alias_or_None)."""
    m = re.match(r'^(.+?)\s+\((\S+)\)$', element.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    elem = element.strip()
    if elem in uc_map:
        return uc_map[elem], elem
    return elem, None


def _expected_explanation(base_name: str, base_alias: str | None,
                           incl_name: str, incl_alias: str | None) -> str:
    """Canonical explanation string for a (base, inclusion) antipattern instance."""
    base_ref = f"'{base_name}' ({base_alias})" if base_alias else f"'{base_name}'"
    incl_ref = f"'{incl_name}' ({incl_alias})" if incl_alias else f"'{incl_name}'"
    return (
        f"{incl_ref} is included by {base_ref}, has no direct actor "
        f"association, does not include or extend any other use case, is not extended by "
        f"any use case, and is not part of any generalisation hierarchy. "
        f"It represents an internal sub-step of '{base_name}' rather than a standalone "
        f"service observable to any actor, making it a functional decomposition of "
        f"'{base_name}' via the include relationship."
    )


# ── Instance-count consistency checks ────────────────────────────────────────


def verify_output_json_consistency(
    sample_id: str,
    output_json: str,
    puml_source: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Check internal consistency of an assistant output JSON string.

    Returns ``(errors, warnings)``.  Errors indicate genuine data integrity
    problems; warnings flag style/template deviations that do not affect
    training quality.

      1. total_instances == sum of all antipattern instance_count values
      2. instance_count for each antipattern == len(instances)
      3. Every instance has an "elements" field (not the old "constructs")
      4. Every instance has exactly 2 elements (base + inclusion)
      5. Every instance has a non-empty "explanation"
      6. Every instance explanation matches the canonical template (when puml_source provided)
         → warning only: content is correct even when alias parentheticals are omitted
      7. Every element alias resolves to a real use case in the PlantUML source (when puml_source provided)
      8. Every element name matches the use case name for that alias in the diagram (when puml_source provided)
      9. The include relationship base→inclusion actually exists in the diagram (when puml_source provided)
     10. The inclusion UC has no direct actor association in the diagram (when puml_source provided)
     11. No duplicate (base, inclusion) pairs within an antipattern
     12. total_antipattern_types == number of distinct antipatterns listed
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        obj = json.loads(output_json)
    except json.JSONDecodeError as exc:
        return [f"{sample_id}: assistant JSON is not valid JSON: {exc}"], []

    if not obj.get("detected", False):
        # Refactored sample — just check antipatterns list is empty
        if obj.get("antipatterns"):
            errors.append(f"{sample_id}: detected=false but antipatterns list is non-empty")
        if obj.get("total_instances", 0) != 0:
            errors.append(f"{sample_id}: detected=false but total_instances={obj.get('total_instances')}")
        return errors, warnings

    # Check 12: total_antipattern_types == number of antipatterns listed
    listed_aps = obj.get("antipatterns", [])
    if obj.get("total_antipattern_types", 0) != len(listed_aps):
        errors.append(
            f"{sample_id}: total_antipattern_types={obj.get('total_antipattern_types')} "
            f"but {len(listed_aps)} antipattern(s) listed"
        )

    claimed_total = obj.get("total_instances", 0)
    actual_total = 0
    uc_map       = _build_uc_map(puml_source)       if puml_source is not None else {}
    include_rels = _build_include_rels(puml_source) if puml_source is not None else set()
    actor_ucs    = _build_actor_ucs(puml_source)    if puml_source is not None else set()

    for ap_idx, ap in enumerate(listed_aps, 1):
        ap_name = ap.get("antipattern_name", f"antipattern[{ap_idx}]")
        instances = ap.get("instances", [])
        claimed_ic = ap.get("instance_count", 0)

        if claimed_ic != len(instances):
            errors.append(
                f"{sample_id}: {ap_name}: instance_count={claimed_ic} "
                f"but len(instances)={len(instances)}"
            )

        actual_total += len(instances)
        seen_pairs: set[tuple[str, str]] = set()

        for i, inst in enumerate(instances, 1):
            if "constructs" in inst and "elements" not in inst:
                errors.append(
                    f"{sample_id}: {ap_name} instance {i}: "
                    f"still uses old 'constructs' key — should be 'elements'"
                )
            elements = inst.get("elements") or inst.get("constructs", [])
            if len(elements) != 2:
                errors.append(
                    f"{sample_id}: {ap_name} instance {i}: "
                    f"expected 2 elements (base, inclusion), got {len(elements)}: {elements}"
                )
            if not inst.get("explanation", "").strip():
                errors.append(
                    f"{sample_id}: {ap_name} instance {i}: explanation is empty"
                )
            if puml_source is not None:
                for ei, elem in enumerate(elements, 1):
                    m_alias = re.match(r'^.+\((\S+)\)$', elem.strip())
                    if m_alias:
                        alias = m_alias.group(1)
                        if alias not in uc_map:
                            errors.append(
                                f"{sample_id}: {ap_name} instance {i}: "
                                f"element {ei} alias '{alias}' not found in PlantUML source"
                            )
                        elif uc_map[alias] != re.match(r'^(.+?)\s+\(\S+\)$', elem.strip()).group(1):
                            errors.append(
                                f"{sample_id}: {ap_name} instance {i}: "
                                f"element {ei} name mismatch — "
                                f"got '{elem.strip()}', diagram has '{uc_map[alias]} ({alias})'"
                            )
                    elif uc_map and elem.strip() not in uc_map.values():
                        errors.append(
                            f"{sample_id}: {ap_name} instance {i}: "
                            f"element {ei} '{elem.strip()}' not found as a use case in PlantUML source"
                        )

            if puml_source is not None and len(elements) == 2:
                base_name, base_alias = _resolve_uc(elements[0], uc_map)
                incl_name, incl_alias = _resolve_uc(elements[1], uc_map)

                # Check 11: no duplicate (base, inclusion) pairs
                pair = (elements[0], elements[1])
                if pair in seen_pairs:
                    errors.append(
                        f"{sample_id}: {ap_name} instance {i}: "
                        f"duplicate instance {pair}"
                    )
                seen_pairs.add(pair)

                # Check 9: include relationship exists in diagram
                if base_alias and incl_alias:
                    if (base_alias, incl_alias) not in include_rels:
                        errors.append(
                            f"{sample_id}: {ap_name} instance {i}: "
                            f"include relationship {base_alias}..>{incl_alias} not in diagram"
                        )
                    # Check 10: inclusion UC has no direct actor association
                    if incl_alias in actor_ucs:
                        errors.append(
                            f"{sample_id}: {ap_name} instance {i}: "
                            f"inclusion UC '{incl_alias}' has a direct actor association"
                        )

                # Check 6: explanation matches canonical template (warning only —
                # content is correct even when alias parentheticals are omitted)
                expected_expl = _expected_explanation(
                    base_name, base_alias, incl_name, incl_alias
                )
                if inst.get("explanation", "") != expected_expl:
                    warnings.append(
                        f"{sample_id}: {ap_name} instance {i}: "
                        f"explanation does not match canonical template"
                    )

    if claimed_total != actual_total:
        errors.append(
            f"{sample_id}: total_instances={claimed_total} "
            f"but sum of instance_counts={actual_total}"
        )

    return errors, warnings


# ── JSONL verification ────────────────────────────────────────────────────────


def verify_sample(
    sample: dict,
    line_num: int,
    run_dir: Path,
    task_instr: str,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for this sample."""
    errors: list[str] = []
    warnings: list[str] = []
    sample_id = sample.get("sample_id", f"<line {line_num}>")
    messages = sample.get("messages", [])

    log.debug("Checking %s", sample_id)

    # ── Parse sample_id ───────────────────────────────────────────────────────
    parts = sample_id.split("_")
    if len(parts) < 2:
        errors.append(f"{sample_id}: cannot parse sample_id (expected NNN_ap/re_...)")
        return errors, warnings

    domain_str = parts[0]       # e.g. "001"
    sample_type = parts[1]      # "ap" or "re"
    domain_dir = run_dir / "domains" / domain_str
    puml_file = domain_dir / f"{domain_str}_{sample_type}.puml"
    jinja_file = domain_dir / f"{domain_str}_{sample_type}.jinja"

    # ── Message count ─────────────────────────────────────────────────────────
    if len(messages) != 3:
        errors.append(
            f"{sample_id}: expected 3 messages, got {len(messages)}"
        )
        return errors, warnings

    sys_msg, user_msg, asst_msg = messages

    # ── Message 1: system ─────────────────────────────────────────────────────
    if sys_msg.get("role") != "system":
        errors.append(
            f"{sample_id}: message[0] role is '{sys_msg.get('role')}', expected 'system'"
        )
    elif sys_msg.get("content") != _TRAINING_SYSTEM_PROMPT:
        errors.append(f"{sample_id}: message[0] (system) content mismatch")

    # ── Message 2: user ───────────────────────────────────────────────────────
    if user_msg.get("role") != "user":
        errors.append(
            f"{sample_id}: message[1] role is '{user_msg.get('role')}', expected 'user'"
        )
    else:
        if not puml_file.exists():
            errors.append(f"{sample_id}: PUML file not found: {puml_file}")
        else:
            puml_content = puml_file.read_text(encoding="utf-8")
            expected_user = f"{task_instr}\n\nPlantUML Model:\n{puml_content}"
            actual_user = user_msg.get("content", "")
            if actual_user != expected_user:
                # Narrow down where the mismatch is
                prefix = f"{task_instr}\n\nPlantUML Model:\n"
                if not actual_user.startswith(task_instr):
                    errors.append(f"{sample_id}: message[1] instruction prefix mismatch")
                elif not actual_user.startswith(prefix):
                    errors.append(f"{sample_id}: message[1] missing 'PlantUML Model:' separator")
                else:
                    actual_puml = actual_user[len(prefix):]
                    if actual_puml != puml_content:
                        errors.append(f"{sample_id}: message[1] PlantUML source mismatch")
                    else:
                        errors.append(f"{sample_id}: message[1] (user) content mismatch")

    # ── Message 3: assistant ──────────────────────────────────────────────────
    if asst_msg.get("role") != "assistant":
        errors.append(
            f"{sample_id}: message[2] role is '{asst_msg.get('role')}', expected 'assistant'"
        )
    else:
        asst_content = asst_msg.get("content", "")

        # Cross-check against .jinja Answer block
        if not jinja_file.exists():
            errors.append(f"{sample_id}: Jinja file not found: {jinja_file}")
        else:
            jinja_content = jinja_file.read_text(encoding="utf-8")
            expected_output = extract_answer_from_jinja(jinja_content)
            if expected_output is None:
                errors.append(
                    f"{sample_id}: could not find 'Answer:' section in {jinja_file.name}"
                )
            elif asst_content != expected_output:
                errors.append(f"{sample_id}: message[2] (assistant) JSON output mismatch")

        # Internal consistency: instance counts, elements field, explanation template
        puml_marker = "PlantUML Model:\n"
        user_content = user_msg.get("content", "")
        puml_idx = user_content.find(puml_marker)
        puml_src = user_content[puml_idx + len(puml_marker):] if puml_idx != -1 else None
        errs, warns = verify_output_json_consistency(
            sample_id, asst_content, puml_source=puml_src,
        )
        errors.extend(errs)
        warnings.extend(warns)

    return errors, warnings


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    script_dir = Path(__file__).parent
    default_run = script_dir / "output" / "run_20260325_012118"

    parser = argparse.ArgumentParser(
        description="Verify training samples against source files."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=str(default_run),
        help="Path to the run directory (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-audit-json",
        action="store_true",
        help=(
            "Skip comparing .jinja / _ap.yaml answer JSON against the audit log. "
            "Use this after running repair_training_samples.py, which intentionally "
            "changes the answer JSON while audit logs record the original generation."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        log.error("Run directory not found: %s", run_dir)
        return

    jsonl_path = run_dir / "samples.jsonl"
    if not jsonl_path.exists():
        log.error("samples.jsonl not found in %s", run_dir)
        return

    state_path = run_dir / "samples_generate_state.json"
    task_mode = "detect"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        task_mode = state.get("task_mode") or state.get("args", {}).get("task_mode", "detect")

    if task_mode not in _TASK_INSTRUCTION:
        log.error("Unknown task_mode '%s' in samples_generate_state.json", task_mode)
        return

    task_instr = _TASK_INSTRUCTION[task_mode].replace("\n", " ")

    log.info("Run dir        : %s", run_dir)
    log.info("JSONL          : %s", jsonl_path.name)
    log.info("Task mode      : %s", task_mode)
    log.info("Skip audit JSON: %s", args.skip_audit_json)

    # ── JSONL verification ────────────────────────────────────────────────────
    all_errors: list[str] = []
    all_warnings: list[str] = []
    total = 0
    failed_domains: dict[str, list[str]] = {}
    warned_domains: dict[str, list[str]] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"<line {line_num}>: JSON parse error: {exc}"
                log.error(msg)
                all_errors.append(msg)
                continue

            total += 1
            sample_id = sample.get("sample_id", f"<line {line_num}>")
            domain_str = sample_id.split("_")[0] if "_" in sample_id else sample_id

            errs, warns = verify_sample(sample, line_num, run_dir, task_instr)
            for w in warns:
                log.warning("WARN  %s", w)
                all_warnings.append(w)
            if errs:
                for e in errs:
                    log.warning("FAIL  %s", e)
                    all_errors.append(e)
                failed_domains.setdefault(domain_str, []).extend(errs)
            elif warns:
                # warn-only (no errors): track separately for the summary
                warned_domains.setdefault(domain_str, []).extend(warns)
            else:
                log.info("OK    %s", sample_id)

    log.info("")
    log.info("═" * 60)
    log.info("Verified %d samples — %d passed, %d failed, %d with warnings",
             total, total - len(failed_domains) - len(warned_domains),
             len(failed_domains), len(warned_domains))

    if failed_domains:
        log.info("")
        log.info("Failed domains:")
        for ds in sorted(failed_domains):
            log.info("  Domain %s:", ds)
            for reason in failed_domains[ds]:
                log.info("    • %s", reason)
    else:
        log.info("All samples passed verification.")

    if warned_domains:
        log.info("")
        log.info("Domains with warnings:")
        for ds in sorted(warned_domains):
            log.info("  Domain %s:", ds)
            for reason in warned_domains[ds]:
                log.info("    ~ %s", reason)

    # ── Audit verification ────────────────────────────────────────────────────
    log.info("")
    log.info("Running audit verification...")
    log.info("")

    domains_dir = run_dir / "domains"
    if not domains_dir.exists():
        log.error("domains/ directory not found in %s", run_dir)
        return

    domain_dirs = sorted(d for d in domains_dir.iterdir() if d.is_dir())
    audit_checked = 0
    audit_skipped = 0
    audit_failed: dict[str, list[str]] = {}
    audit_warned: dict[str, list[str]] = {}
    audit_no_file: list[str] = []

    for domain_dir in domain_dirs:
        ds = domain_dir.name
        audit_file = domain_dir / f"{ds}_audit.txt"

        if not audit_file.exists():
            audit_skipped += 1
            audit_no_file.append(ds)
            continue

        audit_checked += 1
        errs, warns = verify_domain_audit(ds, run_dir, skip_json=args.skip_audit_json)
        for w in warns:
            log.warning("AUDIT WARN  %s", w)
        if errs:
            for e in errs:
                log.warning("AUDIT FAIL  %s", e)
            audit_failed[ds] = errs
        elif warns:
            # warn-only (no errors): track separately for the summary
            audit_warned[ds] = warns
        else:
            log.info("AUDIT OK    domain %s", ds)

    log.info("")
    log.info("═" * 60)
    log.info(
        "Audit: %d domains checked — %d passed, %d failed, %d with warnings"
        " | %d skipped (no audit file)",
        audit_checked,
        audit_checked - len(audit_failed) - len(audit_warned),
        len(audit_failed),
        len(audit_warned),
        audit_skipped,
    )

    if audit_failed:
        log.info("")
        log.info("Audit failed domains:")
        for ds in sorted(audit_failed):
            log.info("  Domain %s:", ds)
            for reason in audit_failed[ds]:
                log.info("    • %s", reason)

    if audit_warned:
        log.info("")
        log.info("Audit domains with warnings:")
        for ds in sorted(audit_warned):
            log.info("  Domain %s:", ds)
            for reason in audit_warned[ds]:
                log.info("    ~ %s", reason)

    if audit_no_file:
        log.info("")
        log.info("Domains without audit file (%d): %s",
                 len(audit_no_file), ", ".join(audit_no_file))


if __name__ == "__main__":
    main()
