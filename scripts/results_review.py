#!/usr/bin/env python3
"""
results_review.py

Tkinter GUI for reviewing model test results.
Left panel: PlantUML diagram PNG (from run's domains/ folder).
Right panels: expected vs predicted JSON, diff-highlighted.
Match badge: MATCH / NO MATCH.
Per-sample notes saved alongside the test file.

Usage:
    python results_review.py [results.jsonl]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required: uv add pillow")
    sys.exit(1)

# ── Palette (Catppuccin Latte) ────────────────────────────────────────────────
BG      = "#eff1f5"
BG_ALT  = "#e6e9ef"
SURFACE = "#ccd0da"
OVERLAY = "#acb0be"
TEXT    = "#4c4f69"
SUBTEXT = "#6c6f85"
RED     = "#d20f39"
GREEN   = "#40a02b"
ORANGE  = "#fe640b"
BLUE    = "#1e66f5"
CHIP_FG = "#ffffff"  # text on coloured chip/badge backgrounds


# ── Check states ─────────────────────────────────────────────────────────────
AUTO_OK   = "auto_ok"    # exact match, auto-verified, not togglable
AUTO_FAIL = "auto_fail"  # detected mismatch, auto-failed, not togglable
MANUAL_OK = "manual_ok"  # user confirmed, togglable
UNCHECKED = "unchecked"  # pending review, togglable

_CHECK_GLYPH = {
    AUTO_OK:   ("✓", GREEN,   False),
    AUTO_FAIL: ("✗", RED,     False),
    MANUAL_OK: ("✓", BLUE,    True),
    UNCHECKED: ("○", OVERLAY, True),
}

ZOOM_MIN  = 0.05
ZOOM_MAX  = 8.0
ZOOM_STEP = 1.20


# ── Match logic ───────────────────────────────────────────────────────────────



def _instance_matched(ap_name: str, i: int, ei: dict, pi: dict, check_states: dict) -> bool:
    """Return True if one instance is correctly matched or every mismatched field is manually verified."""
    ec = ei.get("elements") or ei.get("constructs", []); pc = pi.get("elements") or pi.get("constructs", [])
    if ec != pc and check_states.get(f"ap.{ap_name}.inst.{i}.elements") != MANUAL_OK:
        return False
    ee = ei.get("explanation", ""); pe = pi.get("explanation", "")
    if ee != pe and check_states.get(f"ap.{ap_name}.inst.{i}.explanation") != MANUAL_OK:
        return False
    return True


def compute_contributions(expected_str: str, predicted_str: str, check_states: dict) -> list[str]:
    """Return per-instance TP/TN/FP/FN contributions for metrics.

    RE samples (no expected antipatterns): 1 TN if nothing predicted, else 1 FP per extra AP.
    AP samples: for each paired antipattern, each expected instance → TP or FN; each extra
    predicted instance → FP. Antipatterns missing from predicted → 1 FN per expected instance.
    Extra predicted antipatterns → 1 FP per predicted instance.
    """
    try:
        exp  = json.loads(expected_str)
        pred = json.loads(predicted_str)
    except (json.JSONDecodeError, TypeError):
        return []
    exp_aps  = {a["antipattern_name"]: a for a in exp.get("antipatterns",  [])
                if "antipattern_name" in a}
    pred_aps = {a["antipattern_name"]: a for a in pred.get("antipatterns", [])
                if "antipattern_name" in a}
    results: list[str] = []
    if not exp_aps:
        if not pred_aps:
            results.append("TN")
        else:
            results.extend("FP" for _ in pred_aps)
    else:
        for ap_name, ea in exp_aps.items():
            ei_list = ea.get("instances", [])
            if ap_name not in pred_aps:
                results.extend("FN" for _ in ei_list or [""])
            else:
                pi_list = pred_aps[ap_name].get("instances", [])
                for i in range(max(len(ei_list), len(pi_list))):
                    ei = ei_list[i] if i < len(ei_list) else {}
                    pi = pi_list[i] if i < len(pi_list) else {}
                    if i >= len(ei_list):
                        results.append("FP")
                    elif i >= len(pi_list):
                        results.append("FN")
                    else:
                        results.append("TP" if _instance_matched(ap_name, i, ei, pi, check_states) else "FN")
        for ap_name, pa in pred_aps.items():
            if ap_name not in exp_aps:
                results.extend("FP" for _ in pa.get("instances", []) or [""])
    return results


def normalize_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return s


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["_exp_norm"] = normalize_json(r.get("expected",  ""))
        r["_pred_norm"]= normalize_json(r.get("predicted", ""))
        records.append(r)
    records.sort(key=lambda r: (r["domain_id"], r["sample_id"]))
    return records


def find_png(run_dir: Path, domain_id: int, sample_id: str) -> Path | None:
    suffix  = "ap" if "_ap_" in sample_id else "re"
    png     = run_dir / "domains" / f"{domain_id:03d}" / f"{domain_id:03d}_{suffix}.png"
    return png if png.exists() else None


def notes_path(jsonl_path: Path) -> Path:
    ts = jsonl_path.stem.removeprefix("finetune_eval_")
    return jsonl_path.with_name(f"results_review_{ts}.json")


def load_notes(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_notes(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Image panel ───────────────────────────────────────────────────────────────

class ImagePanel(tk.Frame):
    def __init__(self, parent, on_scroll, **kw):
        super().__init__(parent, bg=BG, **kw)

        header = tk.Frame(self, bg=SURFACE, pady=3)
        header.pack(fill="x")
        tk.Label(header, text="Input Model", fg=BLUE, bg=SURFACE,
                 font=("Helvetica", 13, "bold"), padx=8).pack(side="left")

        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True)
        self._vscroll = tk.Scrollbar(wrap, orient="vertical",   bg=SURFACE)
        self._hscroll = tk.Scrollbar(wrap, orient="horizontal", bg=SURFACE)
        self._vscroll.pack(side="right",  fill="y")
        self._hscroll.pack(side="bottom", fill="x")
        self.canvas = tk.Canvas(wrap, bg=SURFACE, highlightthickness=0,
                                xscrollcommand=self._hscroll.set,
                                yscrollcommand=self._vscroll.set)
        self.canvas.pack(fill="both", expand=True)
        self._vscroll.config(command=self.canvas.yview)
        self._hscroll.config(command=self.canvas.xview)
        self.canvas.bind("<MouseWheel>", lambda e: on_scroll(+1 if (e.num==4 or e.delta>0) else -1))
        self.canvas.bind("<Button-4>",   lambda e: on_scroll(+1))
        self.canvas.bind("<Button-5>",   lambda e: on_scroll(-1))

        self._orig: Image.Image | None = None
        self._photo = None

    def load(self, path: Path | None) -> None:
        self.canvas.delete("all")
        self._orig  = None
        self._photo = None
        if path is None or not path.exists():
            self.canvas.create_text(10, 10, text="No image available",
                                    fill=SUBTEXT, anchor="nw")
            return
        self._orig = Image.open(path)

    def render(self, fit: bool, zoom: float) -> None:
        if self._orig is None:
            return
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)
        if fit:
            img = self._orig.copy()
            img.thumbnail((cw - 4, ch - 4), Image.Resampling.LANCZOS)
            iw, ih = img.size
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")
            self.canvas.configure(scrollregion=(0, 0, cw, ch))
            self._vscroll.pack_forget()
            self._hscroll.pack_forget()
        else:
            w = max(int(self._orig.width  * zoom), 1)
            h = max(int(self._orig.height * zoom), 1)
            img = self._orig.resize((w, h), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(2, 2, image=self._photo, anchor="nw")
            self.canvas.configure(scrollregion=(0, 0, w + 4, h + 4))
            self._vscroll.pack(side="right",  fill="y")
            self._hscroll.pack(side="bottom", fill="x")


# ── Rich comparison panel ─────────────────────────────────────────────────────

class RichComparePanel(tk.Frame):
    """Field-by-field comparison with per-field check/verify indicators."""

    def __init__(self, parent, on_toggle, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._on_toggle = on_toggle

        header = tk.Frame(self, bg=SURFACE, pady=3)
        header.pack(fill="x")
        tk.Label(header, text="Comparison", fg=TEXT, bg=SURFACE,
                 font=("Helvetica", 13, "bold"), padx=8).pack(side="left")

        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True)
        vscroll = tk.Scrollbar(wrap, orient="vertical", bg=SURFACE)
        vscroll.pack(side="right", fill="y")
        self._canvas = tk.Canvas(wrap, bg=BG_ALT, highlightthickness=0,
                                  yscrollcommand=vscroll.set)
        self._canvas.pack(fill="both", expand=True)
        vscroll.config(command=self._canvas.yview)

        self._inner = tk.Frame(self._canvas, bg=BG_ALT)
        self._win_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>",
                         lambda e: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(self._win_id, width=e.width))
        self._bind_scroll(self._canvas)
        self._check_btns: dict[str, tk.Button] = {}

    def _bind_scroll(self, w):
        w.bind("<MouseWheel>", self._on_scroll)
        w.bind("<Button-4>",   lambda e: self._canvas.yview_scroll(-1, "units"))
        w.bind("<Button-5>",   lambda e: self._canvas.yview_scroll(+1, "units"))

    def _on_scroll(self, event):
        direction = -1 if (event.num == 5 or getattr(event, "delta", 0) < 0) else 1
        self._canvas.yview_scroll(-direction, "units")

    def update_check(self, key: str, state: str) -> None:
        btn = self._check_btns.get(key)
        if btn is None:
            return
        glyph, color, togglable = _CHECK_GLYPH[state]
        btn.config(text=glyph, fg=color,
                   cursor="hand2" if togglable else "arrow",
                   state="normal" if togglable else "disabled",
                   disabledforeground=color)

    def load(self, expected_str: str, predicted_str: str, check_states: dict) -> None:
        for w in self._inner.winfo_children():
            w.destroy()
        self._check_btns.clear()

        try:
            exp = json.loads(expected_str)
        except (json.JSONDecodeError, TypeError):
            exp = {}
        try:
            pred = json.loads(predicted_str)
        except (json.JSONDecodeError, TypeError):
            pred = {}

        self._row = 0

        # Column header
        hdr = tk.Frame(self._inner, bg=OVERLAY)
        hdr.grid(row=self._row, column=0, columnspan=4, sticky="ew")
        hdr.columnconfigure(1, weight=1)
        hdr.columnconfigure(2, weight=1)
        for col, text in enumerate(["Field", "Expected", "Predicted", ""]):
            tk.Label(hdr, text=text, fg=SUBTEXT, bg=OVERLAY,
                     font=("Helvetica", 14, "bold"), padx=8, pady=3,
                     anchor="w").grid(row=0, column=col, sticky="ew")
        self._bind_scroll(hdr)
        self._row += 1

        def section(text, color=BLUE):
            f = tk.Frame(self._inner, bg=SURFACE)
            f.grid(row=self._row, column=0, columnspan=4, sticky="ew", pady=(4, 0))
            tk.Label(f, text=text, fg=color, bg=SURFACE,
                     font=("Helvetica", 14, "bold"), padx=10, pady=3).pack(side="left")
            self._bind_scroll(f)
            self._row += 1

        def val_cell(parent, text, fg, bg, height):
            t = tk.Text(parent, height=height, bg=bg, fg=fg,
                        font=("Courier", 14), relief="flat", borderwidth=0,
                        wrap="word", state="disabled", cursor="xterm",
                        selectbackground=OVERLAY, selectforeground=TEXT,
                        insertwidth=0, padx=6, pady=6,
                        spacing1=4, spacing2=4, spacing3=4)
            t.config(state="normal")
            t.insert("1.0", text)
            t.config(state="disabled")
            self._bind_scroll(t)
            return t

        def data_row(key, label, ev, pv, auto_state=None, height=1):
            if auto_state is not None:
                state = auto_state
            elif check_states.get(key) == MANUAL_OK:
                state = MANUAL_OK
            else:
                state = UNCHECKED

            bg      = BG_ALT if self._row % 2 == 0 else BG
            matched = (ev == pv)
            exp_fg  = TEXT if matched else GREEN
            pred_fg = TEXT if matched else ORANGE

            fr = tk.Frame(self._inner, bg=bg)
            fr.grid(row=self._row, column=0, columnspan=4, sticky="ew")
            fr.columnconfigure(1, weight=1)
            fr.columnconfigure(2, weight=1)
            self._bind_scroll(fr)

            tk.Label(fr, text=label, fg=SUBTEXT, bg=bg, anchor="nw",
                     font=("Helvetica", 14), padx=8, pady=4,
                     width=22).grid(row=0, column=0, sticky="nw")
            val_cell(fr, ev, exp_fg,  bg, height).grid(row=0, column=1, sticky="ew", padx=2)
            val_cell(fr, pv, pred_fg, bg, height).grid(row=0, column=2, sticky="ew", padx=2)

            glyph, color, togglable = _CHECK_GLYPH[state]
            btn = tk.Button(fr, text=glyph, fg=color, bg=bg,
                            font=("Helvetica", 15), relief="flat", width=2, pady=4,
                            cursor="hand2" if togglable else "arrow",
                            state="normal" if togglable else "disabled",
                            disabledforeground=color, activebackground=bg)
            if togglable:
                btn.config(command=lambda k=key: self._on_toggle(k))
            btn.grid(row=0, column=3, sticky="ne", padx=4, pady=2)
            self._check_btns[key] = btn
            self._row += 1

        # detected
        ev = exp.get("detected"); pv = pred.get("detected")
        data_row("detected", "detected",
                 str(ev) if ev is not None else "—",
                 str(pv) if pv is not None else "—",
                 auto_state=AUTO_OK if ev == pv else AUTO_FAIL)

        # totals
        for field, lbl in [("total_antipattern_types", "total_ap_types"),
                            ("total_instances",         "total_instances")]:
            ev = exp.get(field); pv = pred.get(field)
            data_row(field, lbl,
                     str(ev) if ev is not None else "—",
                     str(pv) if pv is not None else "—",
                     auto_state=AUTO_OK if ev == pv else None)

        # antipatterns
        exp_aps  = {a["antipattern_name"]: a for a in exp.get("antipatterns",  [])
                    if "antipattern_name" in a}
        pred_aps = {a["antipattern_name"]: a for a in pred.get("antipatterns", [])
                    if "antipattern_name" in a}
        all_names = list(exp_aps) + [n for n in pred_aps if n not in exp_aps]

        for ap_name in all_names:
            ea = exp_aps.get(ap_name, {}); pa = pred_aps.get(ap_name, {})
            section(f"  {ap_name}",
                    GREEN if (ap_name in exp_aps and ap_name in pred_aps) else ORANGE)

            e_name = ea.get("antipattern_name", ap_name if ap_name in exp_aps  else "—")
            p_name = pa.get("antipattern_name", ap_name if ap_name in pred_aps else "—")
            data_row(f"ap.{ap_name}.antipattern_name", "antipattern_name",
                     e_name, p_name,
                     auto_state=AUTO_OK if e_name == p_name else AUTO_FAIL)

            ev = ea.get("instance_count"); pv = pa.get("instance_count")
            data_row(f"ap.{ap_name}.instance_count", "instance_count",
                     str(ev) if ev is not None else "—",
                     str(pv) if pv is not None else "—",
                     auto_state=AUTO_OK if (ev is not None and ev == pv) else None)

            ei_list = ea.get("instances", []); pi_list = pa.get("instances", [])
            for i in range(max(len(ei_list), len(pi_list))):
                ei = ei_list[i] if i < len(ei_list) else {}
                pi = pi_list[i] if i < len(pi_list) else {}
                section(f"    Instance {i + 1}", SUBTEXT)

                ec = ei.get("elements") or ei.get("constructs", []); pc = pi.get("elements") or pi.get("constructs", [])
                data_row(f"ap.{ap_name}.inst.{i}.elements", "elements",
                         "\n".join(f"• {c}" for c in ec) if ec else "—",
                         "\n".join(f"• {c}" for c in pc) if pc else "—",
                         auto_state=AUTO_OK if ec == pc else None,
                         height=max(len(ec), len(pc), 1))

                ee = ei.get("explanation", ""); pe = pi.get("explanation", "")
                data_row(f"ap.{ap_name}.inst.{i}.explanation", "explanation",
                         ee or "—", pe or "—",
                         auto_state=AUTO_OK if ee == pe else None,
                         height=5)


# ── Main app ──────────────────────────────────────────────────────────────────

class TestReviewApp(tk.Tk):
    def __init__(self, jsonl_path: Path | None = None):
        super().__init__()
        self.title("Test Reviewer")
        self.configure(bg=BG)
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

        self.records:     list[dict]   = []
        self.current_idx: int          = 0
        self.notes:       dict         = {}
        self._notes_path: Path | None  = None
        self._run_dir:    Path | None  = None
        self._current_note_text: str   = ""
        self._notes_popup:        tk.Toplevel | None = None
        self._input_popup:        tk.Toplevel | None = None
        self._diagram_popup:      tk.Toplevel | None = None
        self._current_png:        Path | None        = None
        self._json_popup:         tk.Toplevel | None = None
        self._contribs_edit_mode: bool               = False

        self._build_ui()

        if jsonl_path:
            self._load_file(jsonl_path)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Single header row
        hdr = tk.Frame(self, bg=BG, pady=4)
        hdr.pack(fill="x", padx=8)
        self._lbl_progress = tk.Label(hdr, text="", fg=BLUE, bg=BG,
                                       font=("Helvetica", 13, "bold"))
        self._lbl_progress.pack(side="left")
        self._outcome_frame = tk.Frame(hdr, bg=BG)
        self._outcome_frame.pack(side="left", padx=(4, 4))
        self._verified_var = tk.BooleanVar()
        tk.Checkbutton(hdr, text="Verified", variable=self._verified_var,
                       command=self._on_verified_toggle,
                       bg=BG, fg=BLUE, selectcolor=SURFACE,
                       activebackground=BG, activeforeground=BLUE,
                       font=("Helvetica", 11), cursor="hand2").pack(side="left", padx=4)
        self._lbl_meta = tk.Label(hdr, text="", fg=SUBTEXT, bg=BG,
                                   font=("Helvetica", 11), justify="left", anchor="w")
        self._lbl_meta.pack(side="left", padx=(8, 0))

        # Right-side buttons (packed right-to-left, so first = rightmost)
        _btn = dict(bg=SURFACE, fg=TEXT, relief="flat", padx=6, pady=2,
                    font=("Helvetica", 11), cursor="hand2")
        tk.Button(hdr, text="Open File…", command=self._open_file, **_btn
                  ).pack(side="right", padx=(3, 0))
        tk.Button(hdr, text="⟳", command=self._refresh, **_btn
                  ).pack(side="right", padx=3)
        tk.Label(hdr, text="|", fg=SUBTEXT, bg=BG,
                 font=("Helvetica", 11)).pack(side="right", padx=2)
        tk.Button(hdr, text="Next ▶", command=self._next, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="Unverif ▶", command=self._next_unverified, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="FP/FN ▶", command=self._next_fp_fn,
                  bg=SURFACE, fg=RED, relief="flat", padx=6, pady=2,
                  font=("Helvetica", 11, "bold"), cursor="hand2"
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="Go", command=self._jump, **_btn
                  ).pack(side="right", padx=3)
        self._jump_var = tk.StringVar()
        je = tk.Entry(hdr, textvariable=self._jump_var, width=4,
                      bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                      relief="flat", font=("Helvetica", 11))
        je.pack(side="right")
        je.bind("<Return>", lambda _: self._jump())
        tk.Label(hdr, text="Jump:", fg=SUBTEXT, bg=BG,
                 font=("Helvetica", 11)).pack(side="right", padx=(8, 2))
        tk.Button(hdr, text="◀ Prev", command=self._prev, **_btn
                  ).pack(side="right", padx=3)
        tk.Label(hdr, text="|", fg=SUBTEXT, bg=BG,
                 font=("Helvetica", 11)).pack(side="right", padx=2)
        tk.Button(hdr, text="Notes", command=self._open_notes_popup, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="Stats", command=self._open_stats_popup, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="JSON", command=self._open_json_popup, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="Diagram", command=self._open_diagram_popup, **_btn
                  ).pack(side="right", padx=3)
        tk.Button(hdr, text="Input", command=self._open_input_popup, **_btn
                  ).pack(side="right", padx=3)

        # Main panel area
        self._paned = tk.PanedWindow(self, orient="horizontal", bg=OVERLAY,
                                      sashwidth=6, sashrelief="flat", sashpad=2,
                                      handlesize=0)
        self._paned.pack(fill="both", expand=True, padx=12, pady=(4, 0))

        self._rich_panel = RichComparePanel(self._paned, on_toggle=self._on_check_toggle)
        self._paned.add(self._rich_panel, stretch="always")

        self._json_exp_txt:  tk.Text | None = None
        self._json_pred_txt: tk.Text | None = None



    # ── File loading ───────────────────────────────────────���──────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select test results JSONL",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._load_file(Path(path))

    def _load_file(self, path: Path):
        self.records     = load_records(path)
        self._notes_path = notes_path(path)
        self.notes       = load_notes(self._notes_path)
        self._run_dir    = path.parent
        self.current_idx = 0
        self.title(f"Test Reviewer — {path.name}")
        if not self.records:
            messagebox.showerror("Empty", "No records found in file.")
            return
        self._migrate_notes()
        self._show_current()

    def _migrate_notes(self):
        """Strip stale fields and recompute contributions for all notes entries."""
        STALE = {"outcome", "match"}
        rec_by_id = {r["sample_id"]: r for r in self.records}
        dirty = False
        for sample_id, note in self.notes.items():
            if sample_id.startswith("_"):
                continue
            for field in STALE:
                if field in note:
                    del note[field]
                    dirty = True
            rec = rec_by_id.get(sample_id)
            if rec:
                check_states = note.get("checks", {})
                contribs = compute_contributions(
                    rec.get("expected", ""), rec.get("predicted", ""), check_states
                )
                if note.get("contributions") != contribs:
                    note["contributions"] = contribs
                    dirty = True
        if dirty and self._notes_path:
            save_notes(self._notes_path, self.notes)

    def _refresh(self):
        if not self._notes_path:
            return
        jsonl = self._notes_path.with_name(
            "finetune_eval_" + self._notes_path.stem.removeprefix("results_review_") + ".jsonl"
        )
        if jsonl.exists():
            idx = self.current_idx
            self._load_file(jsonl)
            self.current_idx = min(idx, len(self.records) - 1)
            self._show_current()

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_current(self):
        if not self.records:
            return
        rec = self.records[self.current_idx]
        key = rec["sample_id"]

        # Progress + badge
        sample_type = "AP" if "_ap_" in key else "RE"
        self._lbl_progress.config(
            text=f"Domain {rec['domain_id']}  [{sample_type}]  "
                 f"·  {self.current_idx + 1} of {len(self.records)}"
        )
        check_states = self.notes.get(key, {}).get("checks", {})
        auto_contribs = compute_contributions(rec.get("expected", ""), rec.get("predicted", ""), check_states)

        # Persist auto contributions and ensure png_path is set
        if self._notes_path:
            note = self.notes.get(key, {})
            note["contributions"] = auto_contribs
            domain_id  = note.get("domain_id", "")
            suffix     = "ap" if note.get("sample_type") == "antipattern" else "re"
            padded     = f"{int(domain_id):03d}" if domain_id else "000"
            note["png_path"] = f"domains/{padded}/{padded}_{suffix}.png"
            self.notes[key] = note
            save_notes(self._notes_path, self.notes)

        self._contribs_edit_mode = False
        self._refresh_outcome_badge(self._effective_contribs(key), key)

        # Metadata
        self._lbl_meta.config(text=key)

        # PNG — stored so diagram popup can load it
        self._current_png = find_png(self._run_dir, rec["domain_id"], key) if self._run_dir else None
        self._update_diagram_popup()

        # Rich comparison panel
        check_states = self.notes.get(key, {}).get("checks", {})
        exp_str  = rec.get("expected",  "")
        pred_str = rec.get("predicted", "")
        self._rich_panel.load(exp_str, pred_str, check_states)

        self._update_input_popup()
        self._update_json_popup()

        # Verified checkbox
        self._verified_var.set(self.notes.get(key, {}).get("verified", False))

        # Notes backing store + preview
        self._current_note_text = self.notes.get(key, {}).get("text", "")
        self._update_summary()

    def _update_summary(self):
        pass

    # ── Input popup ───────────────────────────────────────────────────────────

    def _open_input_popup(self):
        if not self.records:
            return
        if self._input_popup and self._input_popup.winfo_exists():
            self._input_popup.lift()
            self._input_popup.focus_force()
            self._update_input_popup()
            return
        popup = tk.Toplevel(self, bg=BG)
        self._input_popup = popup
        popup.geometry("900x520")
        popup.update_idletasks()
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        popup.geometry(f"900x520+{(sw - 900) // 2}+{(sh - 520) // 2}")

        inner = tk.Frame(popup, bg=BG)
        inner.pack(fill="both", expand=True, padx=12, pady=(12, 4))
        vscroll = tk.Scrollbar(inner, bg=SURFACE)
        vscroll.pack(side="right", fill="y")
        self._input_popup_txt = tk.Text(
            inner, bg=BG_ALT, fg=TEXT, font=("Courier", 13),
            relief="flat", wrap="word", state="disabled",
            padx=8, pady=6, yscrollcommand=vscroll.set,
            selectbackground=OVERLAY, selectforeground=TEXT,
            insertwidth=0, cursor="xterm",
        )
        self._input_popup_txt.pack(fill="both", expand=True)
        vscroll.config(command=self._input_popup_txt.yview)
        tk.Button(popup, text="Close", command=popup.destroy,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=16, pady=4,
                  font=("Helvetica", 12), cursor="hand2").pack(pady=8)
        self._update_input_popup()

    def _update_input_popup(self):
        if not self._input_popup or not self._input_popup.winfo_exists():
            return
        rec = self.records[self.current_idx]
        self._input_popup.title(f"Input — {rec['sample_id']}")
        self._input_popup_txt.config(state="normal")
        self._input_popup_txt.delete("1.0", "end")
        self._input_popup_txt.insert("1.0", rec.get("input", ""))
        self._input_popup_txt.config(state="disabled")

    # ── Diagram popup ─────────────────────────────────────────────────────────

    def _open_diagram_popup(self):
        if not self.records:
            return
        if self._diagram_popup and self._diagram_popup.winfo_exists():
            self._diagram_popup.lift()
            self._diagram_popup.focus_force()
            self._update_diagram_popup()
            return
        popup = tk.Toplevel(self, bg=BG)
        self._diagram_popup = popup
        popup.update_idletasks()
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        pw, ph = min(1000, sw - 80), min(800, sh - 80)
        popup.geometry(f"{pw}x{ph}+{(sw - pw) // 2}+{(sh - ph) // 2}")

        # Zoom state local to this popup window
        popup._fit_mode = True
        popup._zoom     = 1.0

        ctrl = tk.Frame(popup, bg=BG, pady=4)
        ctrl.pack(fill="x", padx=12)

        fit_btn  = tk.Button(ctrl, text="Fit  ✓", bg=BLUE, fg=CHIP_FG, relief="flat",
                             padx=10, pady=3, font=("Helvetica", 12, "bold"), cursor="hand2")
        fit_btn.pack(side="left", padx=(0, 6))
        zoom_minus = tk.Button(ctrl, text="−", bg=SURFACE, fg=TEXT, relief="flat",
                               padx=10, pady=3, font=("Helvetica", 14), cursor="hand2")
        zoom_minus.pack(side="left")
        lbl_zoom = tk.Label(ctrl, text="fit", fg=TEXT, bg=SURFACE,
                            font=("Helvetica", 12), padx=8, pady=3, width=6)
        lbl_zoom.pack(side="left", padx=2)
        zoom_plus = tk.Button(ctrl, text="+", bg=SURFACE, fg=TEXT, relief="flat",
                              padx=10, pady=3, font=("Helvetica", 14), cursor="hand2")
        zoom_plus.pack(side="left")

        img_panel = ImagePanel(popup, on_scroll=lambda d: _zoom_by(ZOOM_STEP if d > 0 else 1 / ZOOM_STEP))
        img_panel.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        popup._img_panel = img_panel

        def _render():
            img_panel.render(popup._fit_mode, popup._zoom)

        def _zoom_by(factor):
            if popup._fit_mode:
                popup._fit_mode = False
                fit_btn.config(text="Fit", bg=SURFACE, fg=TEXT)
            popup._zoom = max(ZOOM_MIN, min(popup._zoom * factor, ZOOM_MAX))
            lbl_zoom.config(text=f"{popup._zoom * 100:.0f}%")
            _render()

        def _toggle_fit():
            popup._fit_mode = not popup._fit_mode
            if popup._fit_mode:
                fit_btn.config(text="Fit  ✓", bg=BLUE, fg=CHIP_FG)
                lbl_zoom.config(text="fit")
            else:
                fit_btn.config(text="Fit", bg=SURFACE, fg=TEXT)
                lbl_zoom.config(text=f"{popup._zoom * 100:.0f}%")
            _render()

        fit_btn.config(command=_toggle_fit)
        zoom_minus.config(command=lambda: _zoom_by(1 / ZOOM_STEP))
        zoom_plus.config(command=lambda: _zoom_by(ZOOM_STEP))
        popup._render = _render

        self._update_diagram_popup()

    def _update_diagram_popup(self):
        if not self._diagram_popup or not self._diagram_popup.winfo_exists():
            return
        rec = self.records[self.current_idx]
        self._diagram_popup.title(f"Diagram — {rec['sample_id']}")
        self._diagram_popup._img_panel.load(self._current_png)
        self._diagram_popup.after(60, self._diagram_popup._render)

    # ── JSON popup ────────────────────────────────────────────────────────────

    def _open_json_popup(self):
        if not self.records:
            return
        if self._json_popup and self._json_popup.winfo_exists():
            self._json_popup.lift()
            self._json_popup.focus_force()
            self._update_json_popup()
            return
        popup = tk.Toplevel(self, bg=BG)
        self._json_popup = popup
        popup.geometry("1100x540")
        popup.update_idletasks()
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        popup.geometry(f"1100x540+{(sw - 1100) // 2}+{(sh - 540) // 2}")

        inner = tk.Frame(popup, bg=BG)
        inner.pack(fill="both", expand=True, padx=12, pady=(12, 4))
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        for col, lbl in enumerate(["Expected", "Predicted"]):
            tk.Label(inner, text=lbl, fg=SUBTEXT, bg=BG,
                     font=("Helvetica", 12, "bold"), anchor="w").grid(
                         row=0, column=col, sticky="ew", padx=(0, 6 if col == 0 else 0))
        self._json_exp_txt = tk.Text(
            inner, bg=BG_ALT, fg=GREEN, font=("Courier", 12), relief="flat",
            wrap="none", state="disabled", padx=6, pady=4,
            selectbackground=OVERLAY, selectforeground=TEXT,
            insertwidth=0, cursor="xterm",
        )
        self._json_pred_txt = tk.Text(
            inner, bg=BG_ALT, fg=ORANGE, font=("Courier", 12), relief="flat",
            wrap="none", state="disabled", padx=6, pady=4,
            selectbackground=OVERLAY, selectforeground=TEXT,
            insertwidth=0, cursor="xterm",
        )
        exp_scroll   = tk.Scrollbar(inner, orient="vertical",   bg=SURFACE, command=self._json_exp_txt.yview)
        pred_scroll  = tk.Scrollbar(inner, orient="vertical",   bg=SURFACE, command=self._json_pred_txt.yview)
        exp_hscroll  = tk.Scrollbar(inner, orient="horizontal", bg=SURFACE, command=self._json_exp_txt.xview)
        pred_hscroll = tk.Scrollbar(inner, orient="horizontal", bg=SURFACE, command=self._json_pred_txt.xview)
        self._json_exp_txt.config( yscrollcommand=exp_scroll.set,  xscrollcommand=exp_hscroll.set)
        self._json_pred_txt.config(yscrollcommand=pred_scroll.set, xscrollcommand=pred_hscroll.set)
        self._json_exp_txt.grid( row=1, column=0, sticky="nsew", padx=(0, 2))
        exp_scroll.grid(         row=1, column=0, sticky="nse",  padx=(0, 2))
        exp_hscroll.grid(        row=2, column=0, sticky="ew",   padx=(0, 2))
        self._json_pred_txt.grid(row=1, column=1, sticky="nsew")
        pred_scroll.grid(        row=1, column=1, sticky="nse")
        pred_hscroll.grid(       row=2, column=1, sticky="ew")
        inner.rowconfigure(1, weight=1)

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=16, pady=4,
                  font=("Helvetica", 12), cursor="hand2").pack(pady=8)
        self._update_json_popup()

    def _update_json_popup(self):
        if not self._json_popup or not self._json_popup.winfo_exists():
            return
        if not self.records:
            return
        rec = self.records[self.current_idx]
        self._json_popup.title(f"JSON — {rec['sample_id']}")
        for widget, text in ((self._json_exp_txt, rec["_exp_norm"]),
                              (self._json_pred_txt, rec["_pred_norm"])):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.config(state="disabled")

    def _toggle_diagram(self):
        self._diagram_visible = not self._diagram_visible
        if self._diagram_visible:
            # insert before rich panel: forget rich, add img, re-add rich
            self._paned.forget(self._rich_panel)
            self._paned.add(self._img_panel,  stretch="always")
            self._paned.add(self._rich_panel, stretch="always")
            self._diagram_btn.config(text="Hide Diagram ▴", bg=BLUE, fg=CHIP_FG)
            self.after(60, self._render_image)
        else:
            self._paned.forget(self._img_panel)
            self._diagram_btn.config(text="Show Diagram ▾", bg=SURFACE, fg=TEXT)

    # ── Check toggle ──────────────────────────────────────────────────────────

    def _on_check_toggle(self, field_key: str):
        if not self.records or not self._notes_path:
            return
        sample_id = self.records[self.current_idx]["sample_id"]
        checks = self.notes.setdefault(sample_id, {}).setdefault("checks", {})
        new_state = UNCHECKED if checks.get(field_key) == MANUAL_OK else MANUAL_OK
        checks[field_key] = new_state
        self._rich_panel.update_check(field_key, new_state)
        rec = self.records[self.current_idx]
        auto_contribs = compute_contributions(rec.get("expected", ""), rec.get("predicted", ""), checks)
        self.notes[sample_id]["contributions"] = auto_contribs
        save_notes(self._notes_path, self.notes)
        self._refresh_outcome_badge(self._effective_contribs(sample_id), sample_id)
        self._update_summary()

    _CONTRIB_COLORS = {"TP": GREEN, "TN": BLUE, "FP": RED, "FN": ORANGE}

    def _effective_contribs(self, key: str) -> list[str]:
        note = self.notes.get(key, {})
        override = note.get("contributions_override")
        return override if override is not None else note.get("contributions", [])

    def _refresh_outcome_badge(self, contribs: list[str], key: str) -> None:
        for w in self._outcome_frame.winfo_children():
            w.destroy()
        editing = self._contribs_edit_mode
        for i, label in enumerate(contribs):
            color = self._CONTRIB_COLORS.get(label, SURFACE)
            chip = tk.Frame(self._outcome_frame, bg=color)
            chip.pack(side="left", padx=2, pady=1)
            lbl = tk.Label(chip, text=f" {label} ", fg=CHIP_FG, bg=color,
                           font=("Helvetica", 12, "bold"), pady=3,
                           cursor="hand2" if editing else "arrow")
            lbl.pack(side="left")
            if editing:
                lbl.bind("<Button-1>", lambda e, idx=i: self._contrib_menu(e, key, idx))
                x = tk.Label(chip, text="×", fg=CHIP_FG, bg=color,
                             font=("Helvetica", 10), padx=3, pady=3, cursor="hand2")
                x.pack(side="left")
                x.bind("<Button-1>", lambda e, idx=i: self._del_contrib(key, idx))
        if editing:
            tk.Button(self._outcome_frame, text="+", bg=SURFACE, fg=TEXT, relief="flat",
                      font=("Helvetica", 12), cursor="hand2", padx=6, pady=1,
                      command=lambda: self._add_contrib(key)).pack(side="left", padx=(4, 0))
            if self.notes.get(key, {}).get("contributions_override") is not None:
                tk.Button(self._outcome_frame, text="↺", bg=SURFACE, fg=SUBTEXT, relief="flat",
                          font=("Helvetica", 11), cursor="hand2", padx=6, pady=1,
                          command=lambda: self._reset_contribs(key)).pack(side="left", padx=2)
        # Lock / unlock toggle — always visible
        lock_text = "🔓" if editing else "🔒"
        tk.Button(self._outcome_frame, text=lock_text, bg=BG, fg=SUBTEXT, relief="flat",
                  font=("Helvetica", 11), cursor="hand2", padx=4, pady=1,
                  command=lambda: self._toggle_contribs_edit(key)).pack(side="left", padx=(6, 0))

    def _toggle_contribs_edit(self, key: str) -> None:
        self._contribs_edit_mode = not self._contribs_edit_mode
        self._refresh_outcome_badge(self._effective_contribs(key), key)

    def _contrib_menu(self, event, key: str, idx: int) -> None:
        menu = tk.Menu(self, tearoff=0, bg=SURFACE, fg=TEXT,
                       activebackground=OVERLAY, activeforeground=TEXT)
        for option in ("TP", "TN", "FP", "FN"):
            menu.add_command(label=option, command=lambda o=option: self._set_contrib(key, idx, o))
        menu.add_separator()
        menu.add_command(label="Delete", command=lambda: self._del_contrib(key, idx))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _save_contrib_override(self, key: str, contribs: list[str]) -> None:
        note = self.notes.setdefault(key, {})
        note["contributions_override"] = contribs
        if self._notes_path:
            save_notes(self._notes_path, self.notes)
        self._refresh_outcome_badge(contribs, key)

    def _set_contrib(self, key: str, idx: int, new_type: str) -> None:
        contribs = list(self._effective_contribs(key))
        if idx < len(contribs):
            contribs[idx] = new_type
        self._save_contrib_override(key, contribs)

    def _del_contrib(self, key: str, idx: int) -> None:
        contribs = list(self._effective_contribs(key))
        if idx < len(contribs):
            contribs.pop(idx)
        self._save_contrib_override(key, contribs)

    def _add_contrib(self, key: str) -> None:
        contribs = list(self._effective_contribs(key))
        contribs.append("TP")
        self._save_contrib_override(key, contribs)

    def _reset_contribs(self, key: str) -> None:
        note = self.notes.get(key, {})
        note.pop("contributions_override", None)
        if self._notes_path:
            save_notes(self._notes_path, self.notes)
        self._refresh_outcome_badge(self._effective_contribs(key), key)

    def _on_verified_toggle(self):
        if not self.records or not self._notes_path:
            return
        key  = self.records[self.current_idx]["sample_id"]
        note = self.notes.get(key, {})
        if self._verified_var.get():
            note["verified"] = True
        else:
            note.pop("verified", None)
        if note:
            self.notes[key] = note
        elif key in self.notes:
            self.notes.pop(key)
        save_notes(self._notes_path, self.notes)
        self._flush_metrics()
        self._update_summary()

    # ── Persistence ───────────────────────────────────────────────────────────

    # ── Metrics computation ───────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        import math
        rec_by_id = {r["sample_id"]: r for r in self.records}
        tp = tn = fp = fn = 0
        n_verified = 0
        for sample_id, note in self.notes.items():
            if sample_id.startswith("_"):
                continue
            if not note.get("verified"):
                continue
            rec = rec_by_id.get(sample_id)
            if rec is None:
                continue
            n_verified += 1
            for c in self._effective_contribs(sample_id):
                if c == "TP":   tp += 1
                elif c == "TN": tn += 1
                elif c == "FP": fp += 1
                elif c == "FN": fn += 1
        total     = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy  = (tp + tn) / total if total else 0.0
        denom     = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc       = (tp * tn - fp * fn) / denom if denom else 0.0
        return dict(
            n_verified=n_verified, total_units=total,
            ap_units=tp + fn, re_units=tn + fp,
            tp=tp, tn=tn, fp=fp, fn=fn,
            precision=precision, recall=recall,
            f1=f1, accuracy=accuracy, mcc=mcc,
        )

    def _flush_metrics(self) -> None:
        """Recompute and embed metrics inside the review file under '_metrics'."""
        if not self._notes_path:
            return
        m = self._compute_metrics()
        self.notes["_metrics"] = dict(m, saved_at=datetime.now().isoformat())
        save_notes(self._notes_path, self.notes)

    def _open_stats_popup(self):
        m = self._compute_metrics()
        self._flush_metrics()
        tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
        precision, recall = m["precision"], m["recall"]
        f1, accuracy, mcc = m["f1"], m["accuracy"], m["mcc"]
        n_verified = m["n_verified"]

        pop = tk.Toplevel(self)
        pop.title("Confusion Matrix & Metrics")
        pop.configure(bg=BG)
        pop.resizable(False, False)

        tk.Label(pop, text=f"Verified samples: {n_verified}  ·  AP units: {tp + fn}  ·  RE units: {tn + fp}",
                 fg=SUBTEXT, bg=BG, font=("Helvetica", 12)).pack(anchor="w", padx=16, pady=(12, 4))

        # Confusion matrix
        mat = tk.Frame(pop, bg=BG)
        mat.pack(padx=16, pady=(0, 8))
        for c, h in enumerate(["", "Predicted +", "Predicted −"]):
            tk.Label(mat, text=h, fg=SUBTEXT, bg=BG,
                     font=("Helvetica", 12, "bold"), width=14,
                     anchor="center").grid(row=0, column=c, padx=4, pady=2)
        cell_colors = [[None, GREEN, ORANGE], [None, RED, BLUE]]
        for r, (row_label, v0, v1) in enumerate([
            ("Actual +  (AP)", tp, fn),
            ("Actual −  (RE)", fp, tn),
        ]):
            tk.Label(mat, text=row_label, fg=SUBTEXT, bg=BG,
                     font=("Helvetica", 12), anchor="e", width=16
                     ).grid(row=r + 1, column=0, padx=4, pady=2)
            for c, val in enumerate((v0, v1), start=1):
                tk.Label(mat, text=str(val), fg=CHIP_FG, bg=cell_colors[r][c],
                         font=("Helvetica", 13, "bold"), width=10,
                         anchor="center", pady=6
                         ).grid(row=r + 1, column=c, padx=4, pady=2)

        # Metrics
        mf = tk.Frame(pop, bg=BG)
        mf.pack(padx=16, pady=(0, 12))
        for i, (name, val) in enumerate([
            ("Precision", f"{precision:.3f}"),
            ("Recall",    f"{recall:.3f}"),
            ("F1",        f"{f1:.3f}"),
            ("Accuracy",  f"{accuracy:.3f}"),
            ("MCC",       f"{mcc:.3f}"),
        ]):
            tk.Label(mf, text=name, fg=SUBTEXT, bg=BG,
                     font=("Helvetica", 12), anchor="e", width=12
                     ).grid(row=i, column=0, padx=(0, 8), pady=1)
            tk.Label(mf, text=val, fg=TEXT, bg=BG,
                     font=("Helvetica", 12, "bold"), anchor="w"
                     ).grid(row=i, column=1, pady=1)

        # Counts
        cf = tk.Frame(pop, bg=BG)
        cf.pack(padx=16, pady=(0, 12))
        for c, (name, val, color) in enumerate([("TP", tp, GREEN), ("TN", tn, BLUE),
                                                ("FP", fp, RED),   ("FN", fn, ORANGE)]):
            tk.Label(cf, text=name, fg=color, bg=BG,
                     font=("Helvetica", 12, "bold"), width=6,
                     anchor="center").grid(row=0, column=c, padx=4)
            tk.Label(cf, text=str(val), fg=TEXT, bg=SURFACE,
                     font=("Helvetica", 12), width=6, pady=4,
                     anchor="center").grid(row=1, column=c, padx=4)

        tk.Button(pop, text="Close", command=pop.destroy,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=16, pady=4,
                  font=("Helvetica", 12), cursor="hand2").pack(pady=12)

    def _open_notes_popup(self):
        if not self.records:
            return
        # If already open, just bring it to front
        if self._notes_popup and self._notes_popup.winfo_exists():
            self._notes_popup.lift()
            self._notes_popup.focus_force()
            return

        rec = self.records[self.current_idx]
        key = rec["sample_id"]

        popup = tk.Toplevel(self, bg=BG)
        popup.title(f"Notes — {key}")
        popup.geometry("860x520")
        popup.configure(bg=BG)
        # Centre on screen
        popup.update_idletasks()
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        pw = 860; ph = 520
        popup.geometry(f"{pw}x{ph}+{(sw - pw) // 2}+{(sh - ph) // 2}")
        self._notes_popup = popup

        txt = tk.Text(
            popup, bg=SURFACE, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Helvetica", 18), wrap="word",
            padx=16, pady=14,
            spacing1=6, spacing2=4, spacing3=8,
        )
        txt.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        txt.insert("1.0", self._current_note_text)
        txt.focus_set()

        btn_bar = tk.Frame(popup, bg=BG, pady=8)
        btn_bar.pack(fill="x", padx=12)

        def _save_and_close():
            self._current_note_text = txt.get("1.0", "end-1c")
            self._persist()
            popup.destroy()

        tk.Button(btn_bar, text="Save & Close", command=_save_and_close,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=14, pady=6,
                  font=("Helvetica", 14), cursor="hand2").pack(side="right", padx=4)
        tk.Button(btn_bar, text="Discard", command=popup.destroy,
                  bg=SURFACE, fg=SUBTEXT, relief="flat", padx=14, pady=6,
                  font=("Helvetica", 14), cursor="hand2").pack(side="right")

        popup.bind("<Escape>", lambda _: _save_and_close())
        popup.protocol("WM_DELETE_WINDOW", _save_and_close)

    def _persist(self):
        if not self.records or not self._notes_path:
            return
        key  = self.records[self.current_idx]["sample_id"]
        text = self._current_note_text
        note = self.notes.get(key, {})
        if text.strip():
            note["text"] = text
            note["updated_at"] = datetime.now().isoformat()
            self.notes[key] = note
        else:
            note.pop("text", None)
            note.pop("updated_at", None)
            if note:                        # keep entry if checks remain
                self.notes[key] = note
            elif key in self.notes:
                self.notes.pop(key)
        save_notes(self._notes_path, self.notes)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if not self.records:
            return
        self._persist()
        self.current_idx = max(0, self.current_idx - 1)
        self._show_current()

    def _next(self):
        if not self.records:
            return
        self._persist()
        self.current_idx = min(len(self.records) - 1, self.current_idx + 1)
        self._show_current()

    def _next_unverified(self):
        if not self.records:
            return
        self._persist()
        for offset in range(1, len(self.records)):
            idx = (self.current_idx + offset) % len(self.records)
            sid = self.records[idx]["sample_id"]
            if not self.notes.get(sid, {}).get("verified", False):
                self.current_idx = idx
                self._show_current()
                return
        messagebox.showinfo("All verified", "All samples are verified.")

    def _next_fp_fn(self):
        if not self.records:
            return
        self._persist()
        for offset in range(1, len(self.records)):
            idx = (self.current_idx + offset) % len(self.records)
            rec = self.records[idx]
            sid = rec["sample_id"]
            contribs = self._effective_contribs(sid)
            if not contribs:
                checks   = self.notes.get(sid, {}).get("checks", {})
                contribs = compute_contributions(
                    rec.get("expected", ""), rec.get("predicted", ""), checks
                )
            if any(c in ("FP", "FN") for c in contribs):
                self.current_idx = idx
                self._show_current()
                return
        messagebox.showinfo("No FP/FN", "No samples with FP or FN found after this one.")

    def _jump(self):
        if not self.records:
            return
        raw = self._jump_var.get().strip()
        try:
            target = int(raw)
        except ValueError:
            return
        if 1 <= target <= len(self.records):
            self._persist()
            self.current_idx = target - 1
            self._show_current()
        else:
            messagebox.showwarning("Not found", f"Record {target} out of range (1–{len(self.records)}).")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = TestReviewApp(path)
    app.mainloop()
