#!/usr/bin/env python3
"""
samples_generate.py

Simulates a multi-turn conversation with Claude Opus 4.6 to generate
PlantUML use case models for antipattern detection research and LLM fine-tuning.

Usage:
    python samples_generate.py \
        --config antipatterns.yaml \
        --plantuml-jar /path/to/plantuml.jar \
        --num-prompts 10 \
        --task-mode detect
"""

import anthropic
import yaml
import argparse
import csv
import os
import re
import time
import subprocess
import random
import sys
from pathlib import Path
from datetime import datetime
import logging
import json
import signal
import atexit
import jinja2

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
))
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(_console_handler)
logger = logging.getLogger(__name__)

# Shared state written by main() so the atexit handler can always print a summary,
# even when the process is killed via SIGTERM / debugpy stop button.
_run_context: dict = {}


def _atexit_summary() -> None:
    ctx = _run_context
    if not ctx:
        return
    run_dir               = ctx["run_dir"]
    completed_prompts     = ctx["completed_prompts"]
    failed_prompts        = ctx["failed_prompts"]
    reprocess_prompt_nums = ctx.get("reprocess_prompt_nums", set())
    sizes                 = ctx["sizes"]
    args_ns               = ctx["args"]
    ap_usage_counts       = ctx["ap_usage_counts"]
    csv_fields            = ctx["csv_fields"]
    interrupted           = not ctx.get("completed", False)

    if interrupted:
        logger.warning("")
        logger.warning(f"{'─'*60}")
        logger.warning("Run interrupted — partial results saved.")
    else:
        logger.info("")
        logger.info(f"{'─'*60}")
    size_counts = {s: sizes.count(s) for s in args_ns.sizes}
    logger.info("Size distribution: " + "  ".join(f"{s}={c}" for s, c in size_counts.items()))
    logger.info(f"Prompts completed: {len(completed_prompts)} / {args_ns.num_prompts}")
    if reprocess_prompt_nums:
        regenerated = reprocess_prompt_nums & completed_prompts
        logger.info(f"Regenerated      : {len(regenerated)} / {len(reprocess_prompt_nums)} flagged")
    if failed_prompts:
        logger.warning(f"Prompts failed   : {len(failed_prompts)}  →  {sorted(failed_prompts)}")
    else:
        logger.info("Prompts failed   : 0")
    if interrupted:
        logger.warning(f"Resume with      : --resume {run_dir}")

    all_rows = read_csv_rows(run_dir / "samples_stats.csv")
    ap_rows  = [r for r in all_rows if r.get("sample_type") == "antipattern"]

    # Antipattern usage
    logger.info("")
    logger.info("Antipattern usage across run:")
    for name, count in ap_usage_counts.items():
        logger.info(f"  {count:3d}×  {name}")


    logger.info("")
    logger.info(f"{'═'*60}")
    if interrupted:
        logger.warning(f"Partial output under: {run_dir}")
    else:
        logger.info(f"Done. All output under: {run_dir}")
    logger.info(f"  models/           → PlantUML + image pairs")
    logger.info(f"  training_samples/ → .jinja + .yaml training data")


def setup_file_logging(run_dir: Path) -> None:
    """Add a DEBUG-level file handler writing to run_dir/samples_generate.log."""
    log_path = run_dir / "samples_generate.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log file      : {log_path}")

# ── Size ranges (construct counts) ────────────────────────────────────────────

SIZE_RANGES = {
    "small":  (9,  13),
    "medium": (24, 30),
    "large":  (31, 54),
}

# ����─ Antipattern instance caps by size ����───────��─��──────────���───────────────────

# A "group" is one base use case with 1-N qualifying inclusion use cases.
# Under the new instance rule, each (base, inclusion) pair = 1 instance,
# so a group with 3 inclusions contributes 3 instances.
# This constant limits the number of *groups* embedded per size tier.
MAX_INSTANCE_GROUPS_BY_SIZE = {
    "small":  1,
    "medium": 3,
    "large":  5,
}

# Keep the old name as an alias for backward-compat with any callers
MAX_INSTANCES_BY_SIZE = MAX_INSTANCE_GROUPS_BY_SIZE

# ── CLI ���─────────────────────���───────���───────────────���────────────����─����─���������──────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate UML use case model pairs (antipattern + refactored) "
                    "for antipattern detection research and LLM fine-tuning."
    )
    p.add_argument("--config",        default=None,
                   help="Path to YAML antipattern/refactoring config file. "
                        "Required for new runs; ignored when --resume is used.")
    p.add_argument("--domains-config", default=None,
                   help="Path to YAML domains file (domains.yaml). "
                        "When provided, domains are assigned sequentially from the list. "
                        "When omitted, Claude selects domains freely.")
    p.add_argument("--plantuml-jar",  required=True,
                   help="Path to plantuml.jar for image conversion.")
    p.add_argument("--output-dir",    default=None,
                   help="Root output directory (default: ./output next to this script).")
    p.add_argument("--num-prompts",   type=int, default=10,
                   help="Number of model pairs to generate (default: 10).")
    p.add_argument("--sizes", nargs="+", choices=["small", "medium", "large"],
                   default=["small"],
                   help="Which model sizes to generate (default: small). "
                        "Allowed values: small medium large. "
                        "Example: --sizes small medium")
    p.add_argument("--size-weights", nargs="+", type=float, default=None,
                   help="Sampling weights for each size in --sizes (default: equal). "
                        "Must have the same number of values as --sizes. "
                        "Values are relative (need not sum to 1). "
                        "Example: --sizes small medium --size-weights 0.7 0.3")
    p.add_argument("--task-mode",     choices=["detect", "detect-and-refactor"],
                   default="detect",
                   help="Training sample task: 'detect' (default) or 'detect-and-refactor'.")
    p.add_argument("--rate-limit",    type=float, default=2.5,
                   help="Seconds to wait between API calls (default: 2.5).")
    p.add_argument("--api-key",       default=None,
                   help="Anthropic API key (default: ANTHROPIC_API_KEY env var).")
    p.add_argument("--seed",          type=int, default=None,
                   help="Random seed for reproducible size distribution.")
    p.add_argument("--resume",        default=None, metavar="RUN_DIR",
                   help="Resume an interrupted run. Pass the path to the existing run directory.")
    p.add_argument("--reprocess-flagged", action="store_true", default=False,
                   help="With --resume: regenerate prompts marked 'bad' or 'needs-rework' in samples_review.json.")
    p.add_argument("--reprocess-json", action="store_true", default=False,
                   help="With --resume: for each 'needs-rework' prompt that has reviewer notes, "
                        "call the LLM to regenerate only the JSON detection response. "
                        "The model (.puml/.png) is not touched. Audit file is appended to.")
    args = p.parse_args()
    if args.resume is None and args.config is None:
        p.error("--config is required for new runs.")
    if args.reprocess_json and not args.resume:
        p.error("--reprocess-json requires --resume.")
    return args


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_domains(path: str) -> list[dict]:
    """Load the ordered domain list from a domains YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["domains"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert arbitrary text to snake_case with no spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def determine_sizes(n: int, allowed_sizes: list[str], weights: list[float] | None = None) -> list[str]:
    """Return a shuffled list of n size labels drawn from allowed_sizes.

    weights: relative sampling weights (same length as allowed_sizes).
             If None, all sizes are equally likely.
    """
    sizes = random.choices(allowed_sizes, weights=weights, k=n)
    random.shuffle(sizes)
    return sizes


def assign_antipatterns_for_prompt(
    all_antipatterns: list[dict],
    size: str,
    usage_counts: dict[str, int],
) -> list[dict]:
    """Select antipatterns and instance counts for one prompt.

    Returns a list of {"antipattern": {...}, "instance_count": int}.
    Prefers antipattern types with lower usage counts to keep balance across the run.
    """
    max_instances = MAX_INSTANCES_BY_SIZE[size]
    n_available   = len(all_antipatterns)

    # Number of distinct antipattern types to embed
    if size == "small":
        n_types = 1
    elif size == "medium":
        n_types = random.randint(1, min(2, n_available))
    else:  # large
        n_types = random.randint(1, min(3, n_available))

    # Can't have more types than the instance budget allows
    n_types = min(n_types, max_instances)

    # Sort by usage count ascending; random.random() breaks ties
    ranked = sorted(
        range(n_available),
        key=lambda idx: (usage_counts.get(all_antipatterns[idx]["name"], 0), random.random()),
    )
    selected = [all_antipatterns[idx] for idx in ranked[:n_types]]

    # Give each selected type at least 1 instance, then distribute remainder randomly
    total_instances = random.randint(n_types, max_instances)
    counts    = [1] * n_types
    remaining = total_instances - n_types
    for _ in range(remaining):
        counts[random.randrange(n_types)] += 1

    return [{"antipattern": ap, "instance_count": c} for ap, c in zip(selected, counts)]


def write_file(content: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info(f"    Wrote  {path}")
    return path


def convert_to_image(puml_path: Path, jar_path: str) -> Path:
    """Run PlantUML to generate a PNG beside the .puml file."""
    img_path = puml_path.with_suffix(".png")
    try:
        proc = subprocess.run(
            ["java", "-Djava.awt.headless=true", "-jar", str(jar_path), "-tpng", str(puml_path)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            logger.warning(f"    PlantUML warning: {proc.stderr.strip()[:200]}")
        else:
            logger.info(f"    PNG  {img_path}")
    except FileNotFoundError:
        logger.error("    'java' not found on PATH. Image conversion skipped.")
    except subprocess.TimeoutExpired:
        logger.error(f"    PlantUML timed out for {puml_path}")
    except Exception as exc:
        logger.error(f"    Image conversion failed: {exc}")
    return img_path


# ── Audit log ─────────────────────────────────────────────────────────────────

def write_audit(
    prompt_dir: Path,
    prompt_num: int,
    system_prompt: str,
    messages: list[dict],
    raw_response: str,
    model: str,
    usage: dict | None,
) -> Path:
    """Write a human-readable audit file for one API round-trip."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"{prompt_num:03d}_audit.txt"

    sep = "─" * 72
    lines = [
        f"AUDIT LOG — Prompt #{prompt_num:03d}",
        f"Timestamp : {datetime.now().isoformat()}",
        f"Model     : {model}",
    ]
    if usage:
        lines.append(
            f"Tokens    : input={usage.get('input_tokens')}  "
            f"output={usage.get('output_tokens')}"
        )
    lines += [
        "",
        sep,
        "SYSTEM PROMPT",
        sep,
        system_prompt,
        "",
    ]
    for idx, msg in enumerate(messages, start=1):
        role = msg["role"].upper()
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        lines += [
            sep,
            f"MESSAGE {idx} — {role}",
            sep,
            content,
            "",
        ]
    lines += [
        sep,
        "ASSISTANT RESPONSE (raw)",
        sep,
        raw_response,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug(f"  Audit      : {path}")
    return path


def append_audit(
    prompt_dir: Path,
    prompt_num: int,
    system_prompt: str,
    messages: list[dict],
    raw_response: str,
    model: str,
    usage: dict | None,
    label: str = "REPROCESS — JSON only",
) -> Path:
    """Append a new audit section to an existing audit file."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"{prompt_num:03d}_audit.txt"

    sep  = "─" * 72
    bold = "═" * 72
    lines = [
        "",
        bold,
        f"{label}",
        f"Timestamp : {datetime.now().isoformat()}",
        f"Model     : {model}",
    ]
    if usage:
        lines.append(
            f"Tokens    : input={usage.get('input_tokens')}  "
            f"output={usage.get('output_tokens')}"
        )
    lines += ["", sep, "SYSTEM PROMPT", sep, system_prompt, ""]
    for idx, msg in enumerate(messages, start=1):
        role    = msg["role"].upper()
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        lines  += [sep, f"MESSAGE {idx} — {role}", sep, content, ""]
    lines += [sep, "ASSISTANT RESPONSE (raw)", sep, raw_response, ""]

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.debug(f"  Audit (appended) : {path}")
    return path


# ── Prompt builders ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE: str | None = None

def _load_system_prompt_template() -> str:
    """Load and cache the system prompt template from disk."""
    global _SYSTEM_PROMPT_TEMPLATE
    if _SYSTEM_PROMPT_TEMPLATE is None:
        path = Path(__file__).parent / "generation_system_prompt.txt"
        _SYSTEM_PROMPT_TEMPLATE = path.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT_TEMPLATE


def build_system_prompt(assigned_antipatterns: list[dict], task_mode: str) -> str:
    """Build the system prompt for one prompt turn.

    assigned_antipatterns: list of {"antipattern": {...}, "instance_count": int}

    The static skeleton is in generation_system_prompt.txt; this function
    assembles the three dynamic sections and substitutes them in.
    """
    task_line = (
        "detect antipatterns — name each one and list the exact constructs involved"
        if task_mode == "detect"
        else "detect antipatterns (name each one and list the exact constructs involved) "
             "AND provide the full refactored PlantUML model"
    )

    # ── Antipattern + refactoring section ─────────────────────────────────────
    ap_section_lines = []
    for entry in assigned_antipatterns:
        ap = entry["antipattern"]
        n  = entry["instance_count"]
        ap_section_lines += [
            f"─── Antipattern: {ap['name']} ({'%d instance' % n if n == 1 else '%d instances' % n}) ───",
            f"Description : {ap['description'].strip()}",
            "",
            "Refactoring strategy:",
        ]
        for rf in ap["refactorings"]:
            ap_section_lines += [
                f"  Name        : {rf['name']}",
                f"  Description : {rf['description'].strip()}",
            ]
        ap_section_lines.append("")
    ap_section = "\n".join(ap_section_lines).rstrip()

    # ── JSON response template (dynamic per assignment) ────────────────────────
    template_antipatterns = []
    for entry in assigned_antipatterns:
        ap = entry["antipattern"]
        n  = entry["instance_count"]
        template_instances = [
            {
                "elements": ["<base use case (UCx)>", "<inclusion use case (UCy)>"],
                "explanation": f"<why these two use cases form the \"{ap['name']}\" antipattern>",
            }
            for _ in range(n)
        ]
        template_antipatterns.append({
            "antipattern_name": ap["name"],
            "instance_count":   n,
            "instances":        template_instances,
        })
    template_obj = {
        "detected":                True,
        "total_antipattern_types": len(assigned_antipatterns),
        "total_instances":         sum(e["instance_count"] for e in assigned_antipatterns),
        "antipatterns":            template_antipatterns,
    }
    v1_template = json.dumps(template_obj, indent=2)

    return (
        _load_system_prompt_template()
        .replace("<<<AP_SECTION>>>", ap_section)
        .replace("<<<V1_TEMPLATE>>>", v1_template)
        .replace("<<<TASK_LINE>>>", task_line)
    )


def load_prior_puml(run_dir: Path, prompt_num: int) -> tuple[str | None, str | None]:
    """Return (antipattern_puml, refactored_puml) for a previously generated prompt, or (None, None)."""
    domain_dir = run_dir / "domains" / f"{prompt_num:03d}"
    ap_path  = domain_dir / f"{prompt_num:03d}_ap.puml"
    ref_path = domain_dir / f"{prompt_num:03d}_re.puml"
    ap_puml  = ap_path.read_text(encoding="utf-8")  if ap_path.exists()  else None
    ref_puml = ref_path.read_text(encoding="utf-8") if ref_path.exists() else None
    return ap_puml, ref_puml


_USER_MESSAGE_TEMPLATE: jinja2.Template | None = None

def _load_user_message_template() -> jinja2.Template:
    """Load and cache the user message Jinja template from disk."""
    global _USER_MESSAGE_TEMPLATE
    if _USER_MESSAGE_TEMPLATE is None:
        path = Path(__file__).parent / "generation_user_message.jinja"
        env  = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(path.parent)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _USER_MESSAGE_TEMPLATE = env.get_template(path.name)
    return _USER_MESSAGE_TEMPLATE


def build_user_message(
    prompt_num: int, size: str, construct_count: int,
    domain_name: str | None = None,
    reviewer_notes: str | None = None,
    prior_puml: tuple[str | None, str | None] | None = None,
) -> str:
    lo, hi = SIZE_RANGES[size]
    ap_puml, ref_puml = prior_puml if prior_puml else (None, None)
    return _load_user_message_template().render(
        prompt_num=prompt_num,
        size=size,
        construct_count=construct_count,
        lo=lo,
        hi=hi,
        domain_name=domain_name,
        reviewer_notes=reviewer_notes,
        ap_puml=ap_puml,
        ref_puml=ref_puml,
    ).rstrip()


# ── Response parser ───────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """Extract all structured fields from Claude's formatted response."""

    def first_group(pattern, src=text, default=""):
        m = re.search(pattern, src)
        return m.group(1).strip() if m else default

    domain_raw     = first_group(r"DOMAIN:\s*(\S+)", default="unknown_domain")
    domain_display = first_group(r"DOMAIN_DISPLAY:\s*(.+)", default=domain_raw)
    size           = first_group(r"SIZE:\s*(\S+)", default="unknown")

    counts   = re.findall(r"CONSTRUCT_COUNT:\s*(\d+)", text)
    count_v1 = int(counts[0]) if counts else None

    def extract_puml(section: str) -> str | None:
        m = re.search(r"```plantuml\s*(.*?)```", section, re.DOTALL)
        return m.group(1).strip() if m else None

    v1_sec = re.search(r"=== VERSION 1.*?===(.*?)=== VERSION 2", text, re.DOTALL)
    v2_sec = re.search(r"=== VERSION 2.*?===(.*?)$",             text, re.DOTALL)
    v1_text = v1_sec.group(1) if v1_sec else ""

    # ── Parse JSON answer block from VERSION 1 ────────────────────────────────
    antipatterns_detected: list[dict] = []
    json_m = re.search(r"```json\s*(.*?)```", v1_text, re.DOTALL)
    # For the raw-JSON fallback, strip plantuml blocks first to avoid the greedy
    # r"(\{.*\})" regex matching '{' inside a `rectangle "..." { }` PlantUML construct.
    v1_text_no_puml = re.sub(r"```plantuml\s.*?```", "", v1_text, flags=re.DOTALL) if not json_m else ""
    raw_json_m = re.search(r"(\{.*\})", v1_text_no_puml, re.DOTALL) if not json_m else None
    json_source = json_m.group(1) if json_m else (raw_json_m.group(1) if raw_json_m else None)
    if json_source:
        try:
            answer_json = json.loads(json_source.strip())
            for ap in answer_json.get("antipatterns", []):
                antipatterns_detected.append({
                    "name": ap.get("antipattern_name", "Unknown"),
                    "instances": [
                        {
                            "elements": inst.get("elements") or inst.get("constructs", []),
                            "explanation": inst.get("explanation", ""),
                        }
                        for inst in ap.get("instances", [])
                    ],
                })
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # leave antipatterns_detected empty; logged downstream

    v2_text = v2_sec.group(1) if v2_sec else ""
    refactoring_rationale = first_group(
        r"REFACTORING_RATIONALE:\s*(.*?)(?=```|$)",
        src=v2_text, default="",
    )

    # Flat summary strings for logging / CSV
    antipattern_names          = "; ".join(ap["name"] for ap in antipatterns_detected)
    antipattern_instance_counts = "; ".join(str(len(ap["instances"])) for ap in antipatterns_detected)
    total_antipattern_instances = sum(len(ap["instances"]) for ap in antipatterns_detected)

    return {
        "domain":                       slugify(domain_raw),
        "domain_display":               domain_display,
        "size":                         size.lower(),
        "antipatterns_detected":        antipatterns_detected,
        "antipattern_names":            antipattern_names,
        "antipattern_instance_counts":  antipattern_instance_counts,
        "total_antipattern_instances":  total_antipattern_instances,
        "refactoring_rationale":        refactoring_rationale,
        "construct_count_v1":           count_v1,
        "antipattern_puml":             extract_puml(v1_sec.group(1)) if v1_sec else None,
        "refactored_puml":              extract_puml(v2_sec.group(1)) if v2_sec else None,
    }


# ── Training sample content generators ───────────────────────────────────────

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


def make_jinja_antipattern(
    puml: str,
    antipatterns_detected: list[dict],
    refactoring_rationale: str,
    task_mode: str,
    refactored_puml: str | None = None,
) -> str:
    """Return the content of a .jinja training-sample file for the antipattern version."""
    instruction = _TASK_INSTRUCTION[task_mode]

    answer_obj = {
        "detected": bool(antipatterns_detected),
        "total_antipattern_types": len(antipatterns_detected),
        "total_instances": sum(len(ap["instances"]) for ap in antipatterns_detected),
        "antipatterns": [
            {
                "antipattern_name": ap["name"],
                "instance_count":   len(ap["instances"]),
                "instances": [
                    {
                        "elements":    inst.get("elements") or inst.get("constructs", []),
                        "explanation": inst["explanation"],
                    }
                    for inst in ap["instances"]
                ],
            }
            for ap in antipatterns_detected
        ],
    }
    answer = json.dumps(answer_obj, indent=2)

    if task_mode == "detect-and-refactor" and refactored_puml:
        answer += "\n\nRefactored Model:\n" + refactored_puml

    return (
        f"{instruction}\n"
        "\n\n"
        "PlantUML Model:\n"
        f"{puml}"
        "\n\n\n"
        "Answer:\n"
        f"{answer}\n"
    )


_NO_ANTIPATTERN_JSON = json.dumps(
    {"detected": False, "total_antipattern_types": 0, "total_instances": 0, "antipatterns": []},
    indent=2,
)

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



def make_jinja_refactored(puml: str, task_mode: str) -> str:
    """Return the content of a .jinja training-sample file for the refactored version."""
    instruction = _TASK_INSTRUCTION[task_mode]
    answer = _NO_ANTIPATTERN_JSON

    return (
        f"{instruction}\n"
        "\n\n"
        "PlantUML Model:\n"
        f"{puml}"
        "\n\n\n"
        "Answer:\n"
        f"{answer}\n"
    )


def make_yaml_record(
    *,
    domain_id: int,
    domain: str,
    domain_display: str,
    size: str,
    antipatterns_detected: list[dict],
    refactoring_rationale: str | None,
    construct_count: int | None,
    puml: str,
    expected_output: str,
    sample_type: str,           # "antipattern" | "refactored"
    task_mode: str,
    generated_at: str,
) -> dict:

    task_instr = _TASK_INSTRUCTION[task_mode].replace("\n", " ")

    return {
        "metadata": {
            "domain_id":              domain_id,
            "domain":                 domain,
            "domain_display":         domain_display,
            "size":                   size,
            "antipatterns_detected":  antipatterns_detected,
            "sample_type":            sample_type,
            "task_mode":              task_mode,
            "construct_count":        construct_count,
            "refactoring_rationale":  refactoring_rationale or "",
            "generated_at":           generated_at,
        },
        "training_sample": {
            "input":  f"{task_instr}\n\nPlantUML Model:\n{puml}",
            "output": expected_output,
        },
    }


# ── PlantUML construct counter ────────────────────────────────────────────────

def parse_puml_stats(puml: str) -> dict:
    """Count UML constructs and detect system boundary in a PlantUML source string."""
    lines = puml.splitlines()

    actors          = sum(1 for l in lines if re.match(r"^\s*actor\b", l, re.IGNORECASE))
    use_cases       = sum(1 for l in lines if re.match(r"^\s*usecase\b", l, re.IGNORECASE)
                          or re.match(r"^\s*\(", l))
    includes        = sum(1 for l in lines if re.search(r"<<include>>", l, re.IGNORECASE)
                          or re.search(r"\.include\b", l, re.IGNORECASE))
    extends         = sum(1 for l in lines if re.search(r"<<extend>>", l, re.IGNORECASE)
                          or re.search(r"\.extend\b", l, re.IGNORECASE))
    generalizations = sum(1 for l in lines if re.search(r"<\|--|--\|>", l))
    has_boundary    = any(re.match(r"^\s*rectangle\b", l, re.IGNORECASE) for l in lines)
    total_parsed    = actors + use_cases + includes + extends + generalizations

    return {
        "actors":              actors,
        "use_cases":           use_cases,
        "includes":            includes,
        "extends":             extends,
        "generalizations":     generalizations,
        "total_parsed":        total_parsed,
        "has_system_boundary": has_boundary,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"    Wrote  {path}")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to a CSV, writing the header only when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_training_record(
    yaml_path: Path,
    jsonl_path: Path,
    record: dict,
    fields: list[str],
) -> None:
    """Append one training record to both samples.yaml and .jsonl."""
    flat = {k: record[k] for k in fields if k in record}
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or []
    existing.append(flat)
    existing.sort(key=lambda r: (int(r.get("domain_id", 0)), r.get("sample_type", "")))
    yaml_path.write_text(
        yaml.dump(existing, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        messages_record = {
            "sample_id": record.get("sample_id", ""),
            "domain_id": record.get("domain_id"),
            "messages": [
                {"role": "system",    "content": _TRAINING_SYSTEM_PROMPT},
                {"role": "user",      "content": record["input"]},
                {"role": "assistant", "content": record["output"]},
            ],
        }
        f.write(json.dumps(messages_record, ensure_ascii=False) + "\n")


def save_run_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_run_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def remove_csv_rows(path: Path, prompt_nums: set[int], fieldnames: list[str]) -> None:
    """Rewrite a per-prompt CSV file omitting rows whose prompt_num is in prompt_nums."""
    rows = [r for r in read_csv_rows(path) if int(r["domain_id"]) not in prompt_nums]
    write_csv(path, rows, fieldnames)


def remove_training_rows(yaml_path: Path, jsonl_path: Path, prompt_nums: set[int]) -> None:
    """Rewrite samples.yaml and .jsonl omitting records for prompt_nums."""
    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or []
        kept = [r for r in existing if int(r.get("domain_id", -1)) not in prompt_nums]
        yaml_path.write_text(
            yaml.dump(kept, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    if jsonl_path.exists():
        kept_lines = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if int(rec.get("domain_id", -1)) not in prompt_nums:
                    kept_lines.append(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
        jsonl_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""),
                               encoding="utf-8")


# ── JSON-only reprocess ────────────────────────────────────────────────────────

_REPROCESS_SYSTEM_PROMPT = (
    "You are an expert in UML use case diagram analysis specialising in antipattern detection.\n"
    "You will be given:\n"
    "  1. A PlantUML use case model\n"
    "  2. The current JSON detection response for that model\n"
    "  3. Reviewer notes describing exactly what needs to be corrected\n\n"
    "Your task: produce a corrected JSON detection response that addresses the reviewer's notes.\n"
    "Respond with ONLY the JSON object — no explanation, no markdown fences.\n"
    "Use this exact structure:\n"
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


def reprocess_json_only(
    run_dir: Path,
    run_ts: str,
    client,
    training_yaml_path: Path,
    training_jsonl_path: Path,
    rate_limit: float,
) -> None:
    """Regenerate the JSON detection response for all needs-rework prompts that have notes."""
    review_path = run_dir / "samples_review.json"
    if not review_path.exists():
        logger.error("No samples_review.json found in run directory. Cannot reprocess.")
        sys.exit(1)

    review = json.loads(review_path.read_text(encoding="utf-8"))
    flagged: dict[int, str] = {}
    skipped_no_notes: list[int] = []

    for k, v in review.get("prompts", {}).items():
        if v.get("status") == "needs-rework":
            notes = v.get("notes", "").strip()
            if notes:
                flagged[int(k)] = notes
            else:
                skipped_no_notes.append(int(k))

    if skipped_no_notes:
        logger.warning(
            f"Skipping {len(skipped_no_notes)} needs-rework prompt(s) with no notes: "
            f"{sorted(skipped_no_notes)}"
        )
    if not flagged:
        logger.info("No needs-rework prompts with notes found — nothing to do.")
        return

    logger.info(f"Reprocessing JSON for {len(flagged)} prompt(s): {sorted(flagged.keys())}")

    for prompt_num, notes in sorted(flagged.items()):
        prefix     = f"{prompt_num:03d}"
        prompt_dir = run_dir / "domains" / prefix
        jinja_path = prompt_dir / f"{prefix}_ap.jinja"
        yaml_path  = prompt_dir / f"{prefix}_ap.yaml"
        puml_path  = prompt_dir / f"{prefix}_ap.puml"

        logger.info("")
        logger.info(f"{'─'*60}")
        logger.info(f"Prompt {prompt_num}  [json-only reprocess]")
        logger.info(f"  Notes : {notes[:120]}{'…' if len(notes) > 120 else ''}")

        if not puml_path.exists():
            logger.warning(f"  {puml_path.name} not found — skipping.")
            continue

        puml = puml_path.read_text(encoding="utf-8")

        current_json = ""
        if jinja_path.exists():
            text = jinja_path.read_text(encoding="utf-8")
            idx  = text.find("\nAnswer:\n")
            if idx >= 0:
                current_json = text[idx + len("\nAnswer:\n"):].strip()

        user_msg = (
            f"PlantUML Model:\n{puml}\n\n"
            f"Current JSON Response:\n{current_json}\n\n"
            f"Reviewer Notes:\n{notes}"
        )
        messages = [{"role": "user", "content": user_msg}]

        _model = "claude-opus-4-6"
        try:
            response = client.messages.create(
                model=_model,
                max_tokens=2048,
                system=_REPROCESS_SYSTEM_PROMPT,
                messages=messages,
            )
            raw = response.content[0].text
        except anthropic.RateLimitError:
            logger.warning("  Rate-limited. Sleeping 30 s then retrying once…")
            time.sleep(30)
            try:
                response = client.messages.create(
                    model=_model,
                    max_tokens=2048,
                    system=_REPROCESS_SYSTEM_PROMPT,
                    messages=messages,
                )
                raw = response.content[0].text
            except Exception as exc:
                logger.error(f"  Retry failed: {exc} — skipping.")
                continue
        except Exception as exc:
            logger.error(f"  API error: {exc} — skipping.")
            continue

        usage = vars(response.usage) if hasattr(response, "usage") else None
        if usage:
            logger.debug(
                f"  Tokens : input={usage.get('input_tokens')}  "
                f"output={usage.get('output_tokens')}"
            )

        # Strip optional markdown fences
        raw_stripped = raw.strip()
        json_m = re.search(r"```(?:json)?\s*(.*?)```", raw_stripped, re.DOTALL)
        json_str = json_m.group(1).strip() if json_m else raw_stripped

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"  Response is not valid JSON — skipping. Raw: {raw[:300]}")
            continue

        canonical = json.dumps(parsed_json, indent=2, ensure_ascii=False)

        # Append to audit file
        append_audit(prompt_dir, prompt_num, _REPROCESS_SYSTEM_PROMPT, messages, raw, _model, usage)

        # Update _ap.jinja
        if jinja_path.exists():
            text = jinja_path.read_text(encoding="utf-8")
            idx  = text.find("\nAnswer:\n")
            if idx >= 0:
                jinja_path.write_text(
                    text[:idx + len("\nAnswer:\n")] + canonical + "\n", encoding="utf-8"
                )
                logger.info(f"  Updated : {jinja_path.name}")
            else:
                logger.warning(f"  Answer marker not found in {jinja_path.name} — skipped.")
        else:
            logger.warning(f"  {jinja_path.name} not found — skipped.")

        # Update _ap.yaml — replace only the output field, preserving all other formatting
        if yaml_path.exists():
            try:
                text = yaml_path.read_text(encoding="utf-8")
                marker = "\n  output:"
                idx = text.rfind(marker)
                if idx != -1:
                    new_kv = yaml.dump(
                        {"output": canonical},
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                    # Indent by 2 spaces to match training_sample nesting
                    indented = "\n".join("  " + line for line in new_kv.rstrip("\n").split("\n"))
                    yaml_path.write_text(text[:idx] + "\n" + indented + "\n", encoding="utf-8")
                    logger.info(f"  Updated : {yaml_path.name}")
                else:
                    logger.warning(f"  output marker not found in {yaml_path.name} — skipped.")
            except Exception as exc:
                logger.warning(f"  {yaml_path.name} update failed: {exc}")
        else:
            logger.warning(f"  {yaml_path.name} not found — skipped.")

        # Update samples.yaml — replace only the output field of the matching entry
        if training_yaml_path.exists():
            try:
                text = training_yaml_path.read_text(encoding="utf-8")
                # Find the entry block: each starts with "- sample_id:", contains
                # "  domain_id: N" and "_ap_" in the sample_id line
                import re as _re
                pattern = _re.compile(r'^- sample_id:', _re.MULTILINE)
                entry_starts = [m.start() for m in pattern.finditer(text)]
                matched = False
                for i, start in enumerate(entry_starts):
                    end = entry_starts[i + 1] if i + 1 < len(entry_starts) else len(text)
                    block = text[start:end]
                    if (f"  domain_id: {prompt_num}\n" in block
                            and "_ap_" in block):
                        out_idx = block.rfind("\n  output:")
                        if out_idx != -1:
                            new_kv = yaml.dump(
                                {"output": canonical},
                                allow_unicode=True,
                                default_flow_style=False,
                                sort_keys=False,
                            )
                            indented = "\n".join("  " + line for line in new_kv.rstrip("\n").split("\n"))
                            new_block = block[:out_idx] + "\n" + indented + "\n"
                            text = text[:start] + new_block + text[end:]
                            training_yaml_path.write_text(text, encoding="utf-8")
                            matched = True
                        else:
                            logger.warning(f"  output marker not found in entry — skipped.")
                        break
                if not matched:
                    logger.warning(f"  No matching entry in {training_yaml_path.name}")
            except Exception as exc:
                logger.warning(f"  {training_yaml_path.name} update failed: {exc}")

        # Update samples.jsonl
        if training_jsonl_path.exists():
            try:
                lines = training_jsonl_path.read_text(encoding="utf-8").splitlines()
                new_lines = []
                matched = False
                for line in lines:
                    if not line.strip():
                        new_lines.append(line)
                        continue
                    obj = json.loads(line)
                    if (obj.get("domain_id") == prompt_num
                            and "_ap_" in obj.get("sample_id", "")):
                        for msg in obj["messages"]:
                            if msg["role"] == "assistant":
                                msg["content"] = canonical
                                break
                        new_lines.append(json.dumps(obj, ensure_ascii=False))
                        matched = True
                    else:
                        new_lines.append(line)
                if matched:
                    training_jsonl_path.write_text(
                        "\n".join(new_lines) + "\n", encoding="utf-8"
                    )
                    logger.info(f"  Updated : {training_jsonl_path.name}")
                else:
                    logger.warning(f"  No matching entry in {training_jsonl_path.name}")
            except Exception as exc:
                logger.warning(f"  {training_jsonl_path.name} update failed: {exc}")

        if prompt_num < max(flagged.keys()):
            logger.info(f"  Sleeping {rate_limit} s …")
            time.sleep(rate_limit)

    logger.info("")
    logger.info(f"{'═'*60}")
    logger.info(f"JSON reprocess complete. {len(flagged)} prompt(s) processed.")
    logger.info("Review the changes in the reviewer UI before marking as approved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Validate size weights (only for new runs; resume restores these from state)
    if args.resume is None and args.size_weights is not None:
        if len(args.size_weights) != len(args.sizes):
            logger.error(
                f"--size-weights has {len(args.size_weights)} value(s) "
                f"but --sizes has {len(args.sizes)}; they must match."
            )
            sys.exit(1)
        if any(w < 0 for w in args.size_weights):
            logger.error("--size-weights values must be non-negative.")
            sys.exit(1)

    # ── Output directories & run state ──���─────────────────────────────────────
    _csv_fields = [
        "sample_id", "domain_id", "domain_display", "size", "sample_type", "task_mode",
        "antipattern_codes", "antipattern_instance_counts", "total_antipattern_instances",
        "actors", "use_cases", "includes", "extends", "generalizations",
        "total_parsed",
    ]
    _training_fields = [
        "sample_id", "domain_id", "domain_display", "size",
        "antipattern_names", "antipattern_instance_counts", "total_antipattern_instances",
        "sample_type", "task_mode", "generated_at",
        "input", "output",
    ]

    if args.resume:
        run_dir    = Path(args.resume)
        run_ts     = run_dir.name.removeprefix("generate_models_")
        state_path = run_dir / "samples_generate_state.json"
        if not state_path.exists():
            logger.error(f"No samples_generate_state.json found in {run_dir}. Cannot resume.")
            sys.exit(1)
        state = load_run_state(state_path)
        saved = state["args"]
        # Restore run parameters from saved state — CLI values are ignored
        args.num_prompts   = saved["num_prompts"]
        args.sizes         = saved["sizes"]
        args.size_weights  = saved["size_weights"]
        args.task_mode     = saved["task_mode"]
        args.config        = saved["config"]
        args.domains_config = saved["domains_config"]
        sizes                  = state["sizes"]
        completed_prompts      = set(state["completed_prompts"])
        failed_prompts         = set(state.get("failed_prompts", []))
        reprocess_prompt_nums: set[int] = set()
        setup_file_logging(run_dir)
        logger.info(f"Resuming run : {run_dir}")
        logger.info(f"Completed    : {len(completed_prompts)} / {args.num_prompts}")

        reprocess_prompt_nums: set[int] = set()
        reprocess_notes: dict[int, str] = {}
        if args.reprocess_flagged:
            review_path = run_dir / "samples_review.json"
            if review_path.exists():
                review = json.loads(review_path.read_text(encoding="utf-8"))
                for k, v in review.get("prompts", {}).items():
                    if v.get("status") == "needs-rework":
                        num = int(k)
                        reprocess_prompt_nums.add(num)
                        notes = v.get("notes", "").strip()
                        if notes:
                            reprocess_notes[num] = notes
            if reprocess_prompt_nums:
                logger.info(
                    f"Reprocessing : {len(reprocess_prompt_nums)} flagged prompt(s) "
                    f"→ {sorted(reprocess_prompt_nums)}"
                )
                logger.info(
                    f"With notes   : {len(reprocess_notes)} prompt(s) have reviewer notes"
                )
                completed_prompts -= reprocess_prompt_nums
            else:
                logger.info("Reprocessing : no flagged prompts found in samples_review.json")
    else:
        script_dir  = Path(__file__).parent
        output_root = Path(args.output_dir) if args.output_dir else script_dir / "output"
        run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir     = output_root / f"generate_models_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_file_logging(run_dir)
        sizes                  = determine_sizes(args.num_prompts, args.sizes, args.size_weights)
        completed_prompts      = set()
        failed_prompts         = set()
        reprocess_prompt_nums  = set()
        reprocess_notes        = {}
        state_path             = run_dir / "samples_generate_state.json"

    # ── JSON-only reprocess — early exit ──────────────────────────────────────
    if args.reprocess_json:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("No API key. Set ANTHROPIC_API_KEY or use --api-key.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        reprocess_json_only(
            run_dir=run_dir,
            run_ts=run_ts,
            client=client,
            training_yaml_path=run_dir / "samples.yaml",
            training_jsonl_path=run_dir / "samples.jsonl",
            rate_limit=args.rate_limit,
        )
        return

    cfg              = load_config(args.config)
    all_antipatterns = cfg["antipatterns"]
    ap_code_lookup   = {ap["name"]: ap.get("code", ap["name"]) for ap in all_antipatterns}

    # Store run meta fields — written into state on first save
    _run_meta = {
        "run_ts":       run_ts,
        "antipatterns": [
            {"name": ap["name"], "code": ap.get("code", ap["name"])}
            for ap in all_antipatterns
        ],
    }

    domain_pool: list[dict] | None = None
    if args.domains_config:
        domain_pool = load_domains(args.domains_config)
        logger.info(f"Domains      : {len(domain_pool)} loaded from {args.domains_config}")

    if args.resume is None:
        ap_usage_counts = {ap["name"]: 0 for ap in all_antipatterns}
    else:
        ap_usage_counts = state["ap_usage_counts"]

    training_yaml_path  = run_dir / "samples.yaml"
    training_jsonl_path = run_dir / "samples.jsonl"

    # Strip old data for any prompts being reprocessed
    if reprocess_prompt_nums:
        # Read counts before removing rows so we can adjust ap_usage_counts
        prior_ap_rows = [
            r for r in read_csv_rows(run_dir / "samples_stats.csv")
            if int(r.get("domain_id", -1)) in reprocess_prompt_nums
            and r.get("sample_type") == "antipattern"
        ]
        remove_csv_rows(run_dir / "samples_stats.csv", reprocess_prompt_nums, _csv_fields)
        remove_training_rows(training_yaml_path, training_jsonl_path, reprocess_prompt_nums)
        for row in prior_ap_rows:
            for name in row.get("antipattern_codes", "").split(", "):
                name = name.strip()
                if name in ap_usage_counts:
                    ap_usage_counts[name] = max(0, ap_usage_counts[name] - 1)

    logger.info(f"Output root  : {run_dir}")
    logger.info(f"Antipatterns : {', '.join(ap['name'] for ap in all_antipatterns)}")
    logger.info(f"Task mode    : {args.task_mode}")
    logger.info(f"Num prompts  : {args.num_prompts}")
    if args.size_weights:
        size_dist = "  ".join(f"{s}={w}" for s, w in zip(args.sizes, args.size_weights))
        logger.info(f"Sizes        : {size_dist} (weighted)")
    else:
        logger.info(f"Sizes        : {', '.join(args.sizes)} (equal weight)")

    # Anthropic client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("No API key. Set ANTHROPIC_API_KEY or use --api-key.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Main generation loop ───────────────────────────────────────────────────

    # Populate shared context so _atexit_summary() always has what it needs.
    _run_context.update({
        "run_dir":               run_dir,
        "completed_prompts":     completed_prompts,
        "failed_prompts":        failed_prompts,
        "reprocess_prompt_nums": reprocess_prompt_nums,
        "sizes":                 sizes,
        "args":                  args,
        "ap_usage_counts":       ap_usage_counts,
        "csv_fields":            _csv_fields,
    })
    atexit.register(_atexit_summary)

    # sys.exit() triggers atexit; raising KeyboardInterrupt does not always work
    # when the process is stopped via debugpy / SIGTERM.
    def _handle_stop(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT,  _handle_stop)

    try:
        for i, size in enumerate(sizes, start=1):
            if i in completed_prompts:
                logger.info(f"Prompt {i}/{args.num_prompts}  [already completed, skipping]")
                continue
    
            lo, hi          = SIZE_RANGES[size]
            construct_count = random.randint(lo, hi)
    
            # Select antipatterns for this prompt and build system prompt
            assigned      = assign_antipatterns_for_prompt(all_antipatterns, size, ap_usage_counts)
            system_prompt = build_system_prompt(assigned, args.task_mode)
    
            assigned_summary = ", ".join(
                f"{e['antipattern']['name']} ×{e['instance_count']}" for e in assigned
            )
    
            domain_name: str | None = None
            if domain_pool:
                domain_entry = domain_pool[(i - 1) % len(domain_pool)]
                domain_name  = domain_entry["name"]
    
            prior_puml = load_prior_puml(run_dir, i) if i in reprocess_prompt_nums else None
            user_msg = build_user_message(
                i, size, construct_count, domain_name,
                reviewer_notes=reprocess_notes.get(i),
                prior_puml=prior_puml,
            )
            messages = [{"role": "user", "content": user_msg}]
    
            logger.info("")
            logger.info(f"{'─'*60}")
            logger.info(f"Prompt {i}/{args.num_prompts}  size={size}  target_constructs={construct_count}")
            logger.info(f"  Antipatterns : {assigned_summary}")
            if reprocess_notes.get(i):
                logger.info(f"  Reviewer notes : {reprocess_notes[i]}")
    
            # ── API call ───────────────────────────────────────────────────────────
            _model      = "claude-opus-4-6"
            _max_tokens = 4096
            logger.debug(f"  API request  model={_model}  max_tokens={_max_tokens}")
            try:
                response = client.messages.create(
                    model=_model,
                    max_tokens=_max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                raw = response.content[0].text
            except anthropic.RateLimitError:
                logger.warning("  Rate-limited by API. Sleeping 30 s then retrying once …")
                time.sleep(30)
                try:
                    response = client.messages.create(
                        model=_model,
                        max_tokens=_max_tokens,
                        system=system_prompt,
                        messages=messages,
                    )
                    raw = response.content[0].text
                except Exception as exc:
                    logger.error(f"  Retry failed: {exc}. Skipping prompt {i}.")
                    failed_prompts.add(i)
                    continue
            except Exception as exc:
                logger.error(f"  API error: {exc}. Skipping prompt {i}.")
                time.sleep(args.rate_limit)
                failed_prompts.add(i)
                continue
    
            usage = vars(response.usage) if hasattr(response, "usage") else None
            if usage:
                logger.debug(
                    f"  API response input_tokens={usage.get('input_tokens')}  "
                    f"output_tokens={usage.get('output_tokens')}"
                )
    
            prompt_dir = run_dir / "domains" / f"{i:03d}"
            write_audit(prompt_dir, i, system_prompt, messages, raw, _model, usage)
    
            # Update usage counts now that this prompt succeeded
            for entry in assigned:
                ap_usage_counts[entry["antipattern"]["name"]] += 1
    
            # ── Parse ──────────────────────────────────────────────────────────────
            parsed         = parse_response(raw)
            domain         = parsed["domain"] or f"prompt_{i:03d}"
            domain_display = parsed["domain_display"]
            gen_at         = datetime.now().isoformat()
    
            logger.info(f"  Domain       : {domain_display}")
            logger.info(f"  V1 constructs: {parsed['construct_count_v1']}")
            logger.debug(f"  Antipatterns : {parsed['antipattern_names']}")
            logger.debug(f"  Instances    : {parsed['antipattern_instance_counts']}")
            logger.debug(f"  V1 PUML parsed: {parsed['antipattern_puml'] is not None}")
            logger.debug(f"  V2 PUML parsed: {parsed['refactored_puml'] is not None}")
    
            if not parsed["antipattern_puml"] or not parsed["refactored_puml"]:
                logger.warning("  Could not parse both PlantUML blocks — skipping file output for this prompt.")
                time.sleep(args.rate_limit)
                continue
    

            # ── PlantUML files ─────────────────────────────────────────────────────
            v1_puml = write_file(parsed["antipattern_puml"], prompt_dir / f"{i:03d}_ap.puml")
            v2_puml = write_file(parsed["refactored_puml"],  prompt_dir / f"{i:03d}_re.puml")
    
            v1_img = convert_to_image(v1_puml, args.plantuml_jar)
            v2_img = convert_to_image(v2_puml, args.plantuml_jar)
    
            # ── Descriptive statistics ─────────────────────────────────────────────
            v1_stats = parse_puml_stats(parsed["antipattern_puml"])
            v2_stats = parse_puml_stats(parsed["refactored_puml"])
    
            antipattern_codes = "; ".join(
                ap_code_lookup.get(ap["name"], ap["name"])
                for ap in parsed["antipatterns_detected"]
            )
            _base = dict(
                domain_id=i,
                domain=domain,
                domain_display=domain_display,
                size=size,
                antipattern_codes=antipattern_codes,
                antipattern_instance_counts=parsed["antipattern_instance_counts"],
                total_antipattern_instances=parsed["total_antipattern_instances"],
                task_mode=args.task_mode,
                generated_at=gen_at,
            )
            ap_stats_row = {
                "sample_id":                f"{i:03d}_ap_{run_ts}",
                **_base,
                "sample_type":              "antipattern",
                **v1_stats,
                "puml_file":                str(v1_puml),
                "img_file":                 str(v1_img),
            }
            ref_stats_row = {
                "sample_id":                f"{i:03d}_re_{run_ts}",
                **_base,
                "sample_type":              "refactored",
                **v2_stats,
                "puml_file":                str(v2_puml),
                "img_file":                 str(v2_img),
            }
            append_csv_row(run_dir / "samples_stats.csv", ap_stats_row,  _csv_fields)
            append_csv_row(run_dir / "samples_stats.csv", ref_stats_row, _csv_fields)
    
            # ── Training sample – antipattern (negative) ───────────────────────────
            write_file(
                make_jinja_antipattern(
                    parsed["antipattern_puml"],
                    parsed["antipatterns_detected"],
                    parsed["refactoring_rationale"],
                    args.task_mode,
                    parsed["refactored_puml"],
                ),
                prompt_dir / f"{i:03d}_ap.jinja",
            )
    
            ap_answer = make_jinja_antipattern(
                parsed["antipattern_puml"],
                parsed["antipatterns_detected"],
                parsed["refactoring_rationale"],
                args.task_mode,
                parsed["refactored_puml"],
            ).split("Answer:\n", 1)[-1].strip()
    
            _task_instr = _TASK_INSTRUCTION[args.task_mode].replace("\n", " ")
    
            write_file(
                yaml.dump(
                    make_yaml_record(
                        domain_id=i,
                        domain=domain,
                        domain_display=domain_display,
                        size=size,
                        antipatterns_detected=parsed["antipatterns_detected"],
                        refactoring_rationale=parsed["refactoring_rationale"],
                        construct_count=parsed["construct_count_v1"],
                        puml=parsed["antipattern_puml"],
                        expected_output=ap_answer,
                        sample_type="antipattern",
                        task_mode=args.task_mode,
                        generated_at=gen_at,
                    ),
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                ),
                prompt_dir / f"{i:03d}_ap.yaml",
            )
            append_training_record(training_yaml_path, training_jsonl_path, {
                "sample_id":                f"{i:03d}_ap_{run_ts}",
                "domain_id":                i,
                "domain_display":           domain_display,
                "size":                     size,
                "antipattern_names":        parsed["antipattern_names"],
                "antipattern_instance_counts": parsed["antipattern_instance_counts"],
                "total_antipattern_instances": parsed["total_antipattern_instances"],
                "sample_type":              "antipattern",
                "task_mode":                args.task_mode,
                "generated_at":             gen_at,
                "input":                    f"{_task_instr}\n\nPlantUML Model:\n{parsed['antipattern_puml']}",
                "output":                   ap_answer,
            }, _training_fields)
    
            # ── Training sample – refactored (positive) ────────────────────────────
            rf_answer = _NO_ANTIPATTERN_JSON
    
            write_file(
                make_jinja_refactored(parsed["refactored_puml"], args.task_mode),
                prompt_dir / f"{i:03d}_re.jinja",
            )
    
            write_file(
                yaml.dump(
                    make_yaml_record(
                        domain_id=i,
                        domain=domain,
                        domain_display=domain_display,
                        size=size,
                        antipatterns_detected=[],
                        refactoring_rationale=None,
                        construct_count=None,
                        puml=parsed["refactored_puml"],
                        expected_output=rf_answer,
                        sample_type="refactored",
                        task_mode=args.task_mode,
                        generated_at=gen_at,
                    ),
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                ),
                prompt_dir / f"{i:03d}_re.yaml",
            )
            append_training_record(training_yaml_path, training_jsonl_path, {
                "sample_id":                f"{i:03d}_re_{run_ts}",
                "domain_id":                i,
                "domain_display":           domain_display,
                "size":                     size,
                "antipattern_names":        "",
                "antipattern_instance_counts": "",
                "total_antipattern_instances": 0,
                "sample_type":              "refactored",
                "task_mode":                args.task_mode,
                "generated_at":             gen_at,
                "input":                    f"{_task_instr}\n\nPlantUML Model:\n{parsed['refactored_puml']}",
                "output":                   rf_answer,
            }, _training_fields)
    
            # ── Save run state ─────────────────────────────────────────────────────
            completed_prompts.add(i)
            save_run_state(state_path, {
                **_run_meta,
                "args": {
                    "num_prompts":    args.num_prompts,
                    "sizes":          args.sizes,
                    "size_weights":   args.size_weights,
                    "task_mode":      args.task_mode,
                    "config":         str(Path(args.config).relative_to(Path.cwd())) if args.config and Path(args.config).is_absolute() else args.config,
                    "domains_config": str(Path(args.domains_config).relative_to(Path.cwd())) if args.domains_config and Path(args.domains_config).is_absolute() else args.domains_config,
                },
                "sizes":             sizes,
                "ap_usage_counts":   ap_usage_counts,
                "completed_prompts": sorted(completed_prompts),
                "failed_prompts":    sorted(failed_prompts),
            })
    
            # ── Rate limit ─────────────────────────────────────────────────────────
            if i < args.num_prompts:
                logger.info(f"  Sleeping {args.rate_limit} s …")
                time.sleep(args.rate_limit)

    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)  # triggers atexit → _atexit_summary

    # Normal completion — mark as done so _atexit_summary knows it wasn't interrupted.
    _run_context["completed"] = True


if __name__ == "__main__":
    main()
