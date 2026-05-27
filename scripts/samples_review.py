#!/usr/bin/env python3
"""
samples_review.py

Desktop GUI for reviewing generated antipattern/refactored model pairs.
Shows both diagrams side by side with metadata, lets you add notes and mark
each prompt as good / bad / needs-rework.  State is saved to samples_review.json
inside the run directory and restored on reopen.

Usage:
    python samples_review.py [run_dir]
    python samples_review.py output/generate_models_20260321_004201
"""

import csv
import json
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required: uv add pillow")
    sys.exit(1)

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ── Palette (Catppuccin Mocha) ────────────────────────────────────────────────
BG      = "#1e1e2e"
BG_ALT  = "#181825"
SURFACE = "#313244"
OVERLAY = "#45475a"
TEXT    = "#cdd6f4"
SUBTEXT = "#a6adc8"
RED     = "#f38ba8"
GREEN   = "#a6e3a1"
ORANGE  = "#fab387"
BLUE    = "#89b4fa"

STATUS_COLORS = {"approved": GREEN, "needs-rework": ORANGE}
STATUS_LABELS = {"approved": "Approved", "needs-rework": "Needs Rework"}

ZOOM_MIN  = 0.05
ZOOM_MAX  = 8.0
ZOOM_STEP = 1.20   # multiply/divide per click or scroll tick


# ── Data helpers ──────────────────────────────────────────────────────────────

def find_prompts(run_dir: Path) -> list[dict]:
    prompts = []
    domains_dir = run_dir / "domains"
    search_dir = domains_dir if domains_dir.is_dir() else run_dir
    for folder in sorted(search_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            num = int(folder.name)
        except ValueError:
            continue
        prefix = f"{num:03d}"
        ap_img  = folder / f"{prefix}_ap.png"
        ref_img = folder / f"{prefix}_re.png"
        prompts.append({
            "num":    num,
            "folder": folder,
            "ap_img":  ap_img  if ap_img.exists()  else None,
            "ref_img": ref_img if ref_img.exists() else None,
        })
    prompts.sort(key=lambda p: p["num"])
    return prompts


def load_stats(run_dir: Path) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    csv_path = run_dir / "stats_models.csv"
    if not csv_path.exists():
        return stats
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("sample_type") == "antipattern":
                    stats[int(row["domain_id"])] = row
            except (KeyError, ValueError):
                pass
    return stats


def load_review(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("prompts", {}).values():
            if entry.get("status") == "good":
                entry["status"] = "approved"
            elif entry.get("status") == "bad":
                entry.pop("status", None)
                entry.pop("reviewed_at", None)
        return data
    return {"prompts": {}}


def save_review(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── ImagePanel ────────────────────────────────────────────────────────────────

class ImagePanel(tk.Frame):
    """
    A labelled canvas panel that shows one image.
    Supports fit-to-panel mode and zoom+pan mode.
    Clicking the image opens a fullscreen popup.
    Scroll wheel zooms (delegates to on_scroll callback).
    """

    def __init__(self, parent, title: str, title_color: str,
                 on_scroll, on_click, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._on_scroll_cb = on_scroll
        self._on_click_cb  = on_click

        tk.Label(self, text=title, fg=title_color, bg=BG,
                 font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 2))

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

        self._path:     Path | None        = None
        self._orig_img: Image.Image | None = None
        self._photo:    ImageTk.PhotoImage | None = None

        self.canvas.bind("<Button-1>",   lambda _: self._on_click_cb(self._path))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>",   self._wheel)   # Linux scroll up
        self.canvas.bind("<Button-5>",   self._wheel)   # Linux scroll down

    def _wheel(self, event):
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._on_scroll_cb(+1)
        else:
            self._on_scroll_cb(-1)

    def load(self, path: Path | None) -> None:
        self._path     = path
        self._orig_img = None
        self._photo    = None
        self.canvas.delete("all")
        if path is None or not path.exists():
            self.canvas.create_text(10, 10, text="No image",
                                    fill=SUBTEXT, anchor="nw")
            return
        self._orig_img = Image.open(path)

    def render(self, fit: bool, zoom: float) -> None:
        if self._orig_img is None:
            return
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)

        if fit:
            img = self._orig_img.copy()
            img.thumbnail((cw - 4, ch - 4), Image.Resampling.LANCZOS)
            iw, ih = img.size
            x, y = cw // 2, ch // 2
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(x, y, image=self._photo, anchor="center")
            self.canvas.configure(scrollregion=(0, 0, cw, ch))
            self._vscroll.pack_forget()
            self._hscroll.pack_forget()
        else:
            w = max(int(self._orig_img.width  * zoom), 1)
            h = max(int(self._orig_img.height * zoom), 1)
            img = self._orig_img.resize((w, h), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(2, 2, image=self._photo, anchor="nw")
            self.canvas.configure(scrollregion=(0, 0, w + 4, h + 4))
            self._vscroll.pack(side="right",  fill="y")
            self._hscroll.pack(side="bottom", fill="x")


# ── Popup ─────────────────────────────────────────────────────────────────────

def open_popup(root: tk.Tk, path: Path) -> None:
    """Open a near-fullscreen popup for a single image with zoom + pan."""
    if not path.exists():
        return

    top = tk.Toplevel(root)
    top.title(path.stem.replace("_", " ").title())
    top.configure(bg=BG)

    orig = Image.open(path)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ctrl_h = 52   # approximate height of the control bar
    win_w  = min(orig.width  + 20, sw - 40)
    win_h  = min(orig.height + 20 + ctrl_h, sh - 40)
    x_off  = (sw - win_w) // 2
    y_off  = (sh - win_h) // 2
    top.geometry(f"{win_w}x{win_h}+{x_off}+{y_off}")
    zoom  = [1.0]
    photo = [None]

    # ── Canvas + scrollbars
    wrap   = tk.Frame(top, bg=BG)
    wrap.pack(fill="both", expand=True)
    vscroll = tk.Scrollbar(wrap, orient="vertical",   bg=SURFACE)
    hscroll = tk.Scrollbar(wrap, orient="horizontal", bg=SURFACE)
    vscroll.pack(side="right",  fill="y")
    hscroll.pack(side="bottom", fill="x")
    canvas = tk.Canvas(wrap, bg=SURFACE, highlightthickness=0,
                       xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
    canvas.pack(fill="both", expand=True)
    vscroll.config(command=canvas.yview)
    hscroll.config(command=canvas.xview)

    def render():
        top.update_idletasks()
        w = max(int(orig.width  * zoom[0]), 1)
        h = max(int(orig.height * zoom[0]), 1)
        photo[0] = ImageTk.PhotoImage(orig.resize((w, h), Image.Resampling.LANCZOS))
        canvas.delete("all")
        canvas.create_image(2, 2, image=photo[0], anchor="nw")
        canvas.configure(scrollregion=(0, 0, w + 4, h + 4))
        zoom_btn.config(text=f"{zoom[0]*100:.0f}%")

    def fit():
        top.update_idletasks()
        cw = max(canvas.winfo_width()  - 4, 100)
        ch = max(canvas.winfo_height() - 4, 100)
        zoom[0] = min(cw / orig.width, ch / orig.height)
        render()

    def zoom_by(factor):
        zoom[0] = max(ZOOM_MIN, min(zoom[0] * factor, ZOOM_MAX))
        render()

    def on_wheel(event):
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            zoom_by(ZOOM_STEP)
        else:
            zoom_by(1 / ZOOM_STEP)

    canvas.bind("<MouseWheel>", on_wheel)
    canvas.bind("<Button-4>",   on_wheel)
    canvas.bind("<Button-5>",   on_wheel)

    # ── Control bar
    ctrl = tk.Frame(top, bg=BG_ALT, pady=6)
    ctrl.pack(fill="x", side="bottom")

    tk.Button(ctrl, text="−", command=lambda: zoom_by(1 / ZOOM_STEP),
              bg=SURFACE, fg=TEXT, relief="flat", padx=12, pady=4,
              font=("Helvetica", 12), cursor="hand2").pack(side="left", padx=4)
    zoom_btn = tk.Button(ctrl, text="100%", command=lambda: zoom_by(1.0 / zoom[0]),
                         bg=SURFACE, fg=TEXT, relief="flat", padx=8, pady=4,
                         font=("Helvetica", 10), cursor="hand2")
    zoom_btn.pack(side="left")
    tk.Button(ctrl, text="+", command=lambda: zoom_by(ZOOM_STEP),
              bg=SURFACE, fg=TEXT, relief="flat", padx=12, pady=4,
              font=("Helvetica", 12), cursor="hand2").pack(side="left", padx=4)
    tk.Button(ctrl, text="Fit", command=fit,
              bg=OVERLAY, fg=TEXT, relief="flat", padx=10, pady=4,
              cursor="hand2").pack(side="left", padx=(12, 0))

    tk.Button(ctrl, text="Close  ✕", command=top.destroy,
              bg=RED, fg=BG_ALT, relief="flat", padx=12, pady=4,
              font=("Helvetica", 10, "bold"), cursor="hand2"
              ).pack(side="right", padx=8)

    top.after(60, render)   # render at 1:1 after layout settles


# ── App ───────────────────────────────────────────────────────────────────────

class ReviewApp(tk.Tk):
    def __init__(self, run_dir: Path | None = None):
        super().__init__()
        self.title("Model Reviewer")
        self.configure(bg=BG)
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

        self.run_dir:     Path | None  = None
        self.prompts:     list[dict]   = []
        self.current_idx: int          = 0
        self.review_data: dict         = {"prompts": {}}
        self.review_path: Path | None  = None
        self.stats:       dict[int, dict] = {}

        self._fit_mode: bool  = True
        self._zoom:     float = 1.0

        self._build_ui()

        if run_dir:
            self._load_run(run_dir)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar
        top = tk.Frame(self, bg=BG_ALT, pady=6)
        top.pack(fill="x")

        self._lbl_run = tk.Label(top, text="No run loaded",
                                  fg=TEXT, bg=BG_ALT, font=("Helvetica", 11, "bold"))
        self._lbl_run.pack(side="left", padx=12)

        tk.Button(top, text="Open Run…", command=self._open_run,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=8, pady=3,
                  activebackground=OVERLAY, cursor="hand2"
                  ).pack(side="right", padx=12)

        tk.Button(top, text="⟳ Refresh", command=self._refresh,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=8, pady=3,
                  activebackground=OVERLAY, cursor="hand2"
                  ).pack(side="right", padx=4)

        # ── Prompt header
        hdr = tk.Frame(self, bg=BG, pady=4)
        hdr.pack(fill="x", padx=12)

        self._lbl_progress = tk.Label(hdr, text="", fg=BLUE, bg=BG,
                                       font=("Helvetica", 12, "bold"))
        self._lbl_progress.pack(side="left")

        self._lbl_badge = tk.Label(hdr, text="", fg=BG_ALT, bg=SURFACE,
                                    font=("Helvetica", 10, "bold"), padx=10, pady=2)
        self._lbl_badge.pack(side="left", padx=10)

        # ── Metadata
        self._lbl_meta = tk.Label(self, text="", fg=SUBTEXT, bg=BG,
                                   font=("Helvetica", 10), justify="left",
                                   anchor="w", wraplength=1600)
        self._lbl_meta.pack(fill="x", padx=12, pady=(0, 4))

        # ── Zoom controls
        zf = tk.Frame(self, bg=BG, pady=3)
        zf.pack(fill="x", padx=12)

        self._fit_btn = tk.Button(zf, text="Fit  ✓", command=self._toggle_fit,
                                   bg=BLUE, fg=BG_ALT, relief="flat",
                                   padx=10, pady=3, font=("Helvetica", 9, "bold"),
                                   cursor="hand2")
        self._fit_btn.pack(side="left", padx=(0, 8))

        tk.Button(zf, text="−", command=lambda: self._zoom_by(1 / ZOOM_STEP),
                  bg=SURFACE, fg=TEXT, relief="flat", padx=10, pady=3,
                  font=("Helvetica", 11), cursor="hand2").pack(side="left")

        self._lbl_zoom = tk.Label(zf, text="fit", fg=TEXT, bg=SURFACE,
                                   font=("Helvetica", 9), padx=8, pady=3, width=6)
        self._lbl_zoom.pack(side="left", padx=2)

        tk.Button(zf, text="+", command=lambda: self._zoom_by(ZOOM_STEP),
                  bg=SURFACE, fg=TEXT, relief="flat", padx=10, pady=3,
                  font=("Helvetica", 11), cursor="hand2").pack(side="left")

        tk.Label(zf, text="  scroll wheel to zoom  ·  click image to enlarge",
                 fg=OVERLAY, bg=BG, font=("Helvetica", 9)).pack(side="left", padx=12)

        self._notes_toggle_btn = tk.Button(
            zf, text="Notes",
            command=self._toggle_notes_panel,
            bg=SURFACE, fg="gold", relief="flat", padx=10, pady=3,
            font=("Helvetica", 9, "bold"), cursor="hand2",
        )
        self._notes_toggle_btn.pack(side="right", padx=(4, 0))

        self._json_toggle_btn = tk.Button(
            zf, text="Expected JSON",
            command=self._toggle_json_panel,
            bg=SURFACE, fg=ORANGE, relief="flat", padx=10, pady=3,
            font=("Helvetica", 9, "bold"), cursor="hand2",
        )
        self._json_toggle_btn.pack(side="right", padx=(4, 0))

        tk.Button(
            zf, text="AP Source",
            command=self._toggle_puml_panel,
            bg=SURFACE, fg=BLUE, relief="flat", padx=10, pady=3,
            font=("Helvetica", 9, "bold"), cursor="hand2",
        ).pack(side="right", padx=(4, 0))

        # ── Status buttons
        sf = tk.Frame(self, bg=BG_ALT, pady=5)
        sf.pack(fill="x", padx=12)

        tk.Button(sf, text="Clear Status",
                  command=lambda: self._set_status(None),
                  bg=OVERLAY, fg=TEXT, relief="flat",
                  font=("Helvetica", 10), padx=10, pady=4,
                  cursor="hand2"
                  ).pack(side="right", padx=(4, 0))

        for key, label in reversed(list(STATUS_LABELS.items())):
            color = STATUS_COLORS[key]
            tk.Button(sf, text=label,
                      command=lambda k=key: self._set_status(k),
                      bg=color, fg=BG_ALT, relief="flat",
                      font=("Helvetica", 10, "bold"), padx=12, pady=4,
                      activebackground=color, cursor="hand2"
                      ).pack(side="right", padx=4)

        tk.Label(sf, text="Mark as:", fg=SUBTEXT, bg=BG_ALT,
                 font=("Helvetica", 10)).pack(side="right", padx=(0, 8))

        # ── Images
        paned = tk.PanedWindow(self, orient="horizontal", bg=OVERLAY,
                               sashwidth=6, sashrelief="flat", sashpad=2,
                               handlesize=0)
        paned.pack(fill="both", expand=True, padx=12)

        self._ap_panel = ImagePanel(
            paned, "ANTIPATTERN", RED,
            on_scroll=self._on_panel_scroll,
            on_click=self._on_image_click,
        )
        paned.add(self._ap_panel, stretch="always")

        self._ref_panel = ImagePanel(
            paned, "REFACTORED", GREEN,
            on_scroll=self._on_panel_scroll,
            on_click=self._on_image_click,
        )
        paned.add(self._ref_panel, stretch="always")

        self._json_win:    tk.Toplevel | None = None
        self._json_text:   tk.Text     | None = None
        self._json_status: tk.Label    | None = None

        self._puml_win:  tk.Toplevel | None = None
        self._puml_text: tk.Text     | None = None

        self._notes_win:  tk.Toplevel | None = None
        self._notes_text: tk.Text     | None = None
        self._current_notes: str = ""

        # ── Bottom bar
        bot = tk.Frame(self, bg=BG_ALT, pady=8)
        bot.pack(fill="x", side="bottom")

        nf = tk.Frame(bot, bg=BG_ALT)
        nf.pack(side="right", padx=16)

        tk.Button(nf, text="◀  Prev", command=self._prev,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=12, pady=5,
                  cursor="hand2").pack(side="left", padx=4)

        tk.Button(nf, text="Next Flagged  ⚑", command=self._next_flagged,
                  bg=ORANGE, fg=BG_ALT, relief="flat", padx=12, pady=5,
                  font=("Helvetica", 10, "bold"),
                  cursor="hand2").pack(side="left", padx=4)

        tk.Label(nf, text="Jump to:", fg=SUBTEXT, bg=BG_ALT,
                 font=("Helvetica", 10)).pack(side="left", padx=(16, 4))

        self._jump_var = tk.StringVar()
        je = tk.Entry(nf, textvariable=self._jump_var, width=5,
                      bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                      relief="flat", font=("Helvetica", 10))
        je.pack(side="left")
        je.bind("<Return>", lambda _: self._jump())

        tk.Button(nf, text="Go", command=self._jump,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=8, pady=5,
                  cursor="hand2").pack(side="left", padx=4)

        tk.Button(nf, text="Next  ▶", command=self._next,
                  bg=SURFACE, fg=TEXT, relief="flat", padx=12, pady=5,
                  cursor="hand2").pack(side="left", padx=4)

        self._lbl_summary = tk.Label(bot, text="", fg=SUBTEXT, bg=BG_ALT,
                                      font=("Helvetica", 10))
        self._lbl_summary.pack(side="right", padx=(0, 32))

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _on_panel_scroll(self, direction: int):
        """Called by either panel's scroll wheel (+1 = up, -1 = down)."""
        if direction > 0:
            self._zoom_by(ZOOM_STEP)
        else:
            self._zoom_by(1 / ZOOM_STEP)

    def _zoom_by(self, factor: float):
        if self._fit_mode:
            # Leave fit mode, seed zoom from current fit ratio
            self._fit_mode = False
            self._fit_btn.config(text="Fit", bg=SURFACE, fg=TEXT)
        self._zoom = max(ZOOM_MIN, min(self._zoom * factor, ZOOM_MAX))
        self._lbl_zoom.config(text=f"{self._zoom * 100:.0f}%")
        self._render_images()

    def _toggle_fit(self):
        self._fit_mode = not self._fit_mode
        if self._fit_mode:
            self._fit_btn.config(text="Fit  ✓", bg=BLUE, fg=BG_ALT)
            self._lbl_zoom.config(text="fit")
        else:
            self._fit_btn.config(text="Fit", bg=SURFACE, fg=TEXT)
            self._lbl_zoom.config(text=f"{self._zoom * 100:.0f}%")
        self._render_images()

    def _render_images(self):
        self._ap_panel.render(self._fit_mode, self._zoom)
        self._ref_panel.render(self._fit_mode, self._zoom)

    def _on_image_click(self, path: Path | None):
        if path and path.exists():
            open_popup(self, path)

    # ── Run loading ───────────────────────────────────────────────────────────

    def _open_run(self):
        d = filedialog.askdirectory(title="Select run directory")
        if d:
            self._load_run(Path(d))

    def _refresh(self):
        if self.run_dir is None:
            return
        idx = self.current_idx
        self._load_run(self.run_dir)
        if self.prompts:
            self.current_idx = min(idx, len(self.prompts) - 1)
            self._show_current()

    def _load_run(self, run_dir: Path):
        self.run_dir     = run_dir
        self.review_path = run_dir / "samples_review.json"
        self.prompts     = find_prompts(run_dir)
        self.review_data = load_review(self.review_path)
        self.stats       = load_stats(run_dir)
        self.current_idx = 0

        self._lbl_run.config(text=f"Run: {run_dir.name}")
        self.title(f"Model Reviewer — {run_dir.name}")

        if not self.prompts:
            messagebox.showerror("No prompts",
                                  f"No domain folders found in\n{run_dir}")
            return

        self._show_current()

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_current(self):
        if not self.prompts:
            return

        p   = self.prompts[self.current_idx]
        num = p["num"]
        key = str(num)

        self._lbl_progress.config(
            text=f"Prompt {num}  ·  {self.current_idx + 1} of {len(self.prompts)}"
        )

        review = self.review_data["prompts"].get(key, {})
        status = review.get("status")
        if status:
            self._lbl_badge.config(text=f"  {STATUS_LABELS[status]}  ",
                                    bg=STATUS_COLORS[status], fg=BG_ALT)
        else:
            self._lbl_badge.config(text="  Not Reviewed  ", bg=SURFACE, fg=SUBTEXT)

        row        = self.stats.get(num, {})
        domain     = row.get("domain_display", f"Domain {num}")
        size       = row.get("size", "—")
        mode       = row.get("task_mode", "—")
        ap_codes   = row.get("antipattern_codes", "—")
        counts     = row.get("antipattern_instance_counts", "—")
        total      = row.get("total_antipattern_instances", "—")

        rev_at     = review.get("reviewed_at", "")
        rev_str    = f"  ·  Reviewed {rev_at[:16]}" if rev_at else ""

        self._lbl_meta.config(text=(
            f"Domain: {domain}    Size: {size}    Mode: {mode}    "
            f"Antipatterns: {ap_codes}    Instances per AP: {counts}    "
            f"Total instances: {total}{rev_str}"
        ))

        self._ap_panel.load(p["ap_img"])
        self._ref_panel.load(p["ref_img"])
        self.after(60, self._render_images)

        self._current_notes = review.get("notes", "")
        if self._notes_win and self._notes_win.winfo_exists():
            self._notes_text.delete("1.0", "end")
            self._notes_text.insert("1.0", self._current_notes)

        if self._json_win and self._json_win.winfo_exists():
            self._load_expected_json()
        if self._puml_win and self._puml_win.winfo_exists():
            self._load_puml()

        self._update_summary()

    def _update_summary(self):
        if not self.review_data["prompts"]:
            self._lbl_summary.config(text="")
            return
        counts = {k: 0 for k in STATUS_LABELS}
        for v in self.review_data["prompts"].values():
            s = v.get("status")
            if s in counts:
                counts[s] += 1
        total    = len(self.prompts)
        reviewed = sum(counts.values())
        parts    = [f"{STATUS_LABELS[k]}: {counts[k]}" for k in STATUS_LABELS]
        self._lbl_summary.config(
            text=f"Reviewed {reviewed}/{total}    " + "    ".join(parts)
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self):
        if not self.prompts or not self.review_path:
            return
        key   = str(self.prompts[self.current_idx]["num"])
        entry = self.review_data["prompts"].setdefault(key, {})
        if self._notes_win and self._notes_win.winfo_exists() and self._notes_text:
            self._current_notes = self._notes_text.get("1.0", "end-1c")
        entry["notes"] = self._current_notes
        save_review(self.review_path, self.review_data)

    def _set_status(self, status: str | None):
        if not self.prompts or not self.review_path:
            return
        self._persist()
        key   = str(self.prompts[self.current_idx]["num"])
        entry = self.review_data["prompts"].setdefault(key, {})
        if status is None:
            entry.pop("status", None)
            entry.pop("reviewed_at", None)
        else:
            entry["status"]      = status
            entry["reviewed_at"] = datetime.now().isoformat()
        save_review(self.review_path, self.review_data)
        self._show_current()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if not self.prompts:
            return
        self._persist()
        self.current_idx = max(0, self.current_idx - 1)
        self._show_current()

    def _next(self):
        if not self.prompts:
            return
        self._persist()
        self.current_idx = min(len(self.prompts) - 1, self.current_idx + 1)
        self._show_current()

    def _next_flagged(self):
        if not self.prompts:
            return
        flagged = {"bad", "needs-rework"}
        for i in range(self.current_idx + 1, len(self.prompts)):
            key = str(self.prompts[i]["num"])
            if self.review_data["prompts"].get(key, {}).get("status") in flagged:
                self._persist()
                self.current_idx = i
                self._show_current()
                return
        messagebox.showinfo("No more flagged",
                            "No bad / needs-rework prompts found after this one.")

    def _jump(self):
        if not self.prompts:
            return
        try:
            target = int(self._jump_var.get())
        except ValueError:
            return
        for idx, p in enumerate(self.prompts):
            if p["num"] == target:
                self._persist()
                self.current_idx = idx
                self._show_current()
                return
        messagebox.showwarning("Not found", f"Prompt {target} not found in this run.")

    # ── Expected JSON floating window ─────────────────────────────────────────

    def _toggle_notes_panel(self):
        if self._notes_win and self._notes_win.winfo_exists():
            self._current_notes = self._notes_text.get("1.0", "end-1c")
            self._persist()
            self._notes_win.destroy()
            self._notes_win = None
            self._notes_text = None
            return

        win = tk.Toplevel(self)
        win.title("Notes")
        win.configure(bg=BG)
        win.attributes("-topmost", True)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.55)
        x = (sw - w) // 2
        y = (sh - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")

        hdr = tk.Frame(win, bg=BG_ALT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Notes", fg="gold", bg=BG_ALT,
                 font=("Helvetica", 15, "bold")).pack(side="left", padx=14)

        edit = tk.Frame(win, bg=BG)
        edit.pack(fill="both", expand=True, padx=10, pady=8)

        vscroll = tk.Scrollbar(edit, orient="vertical", bg=SURFACE)
        vscroll.pack(side="right", fill="y")
        hscroll = tk.Scrollbar(edit, orient="horizontal", bg=SURFACE)
        hscroll.pack(side="bottom", fill="x")

        self._notes_text = tk.Text(
            edit, bg=BG_ALT, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Courier", 16), wrap="word",
            padx=12, pady=10,
            spacing1=6, spacing2=4, spacing3=6,
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )
        self._notes_text.pack(fill="both", expand=True)
        vscroll.config(command=self._notes_text.yview)
        hscroll.config(command=self._notes_text.xview)

        self._notes_text.insert("1.0", self._current_notes)
        self._notes_text.bind("<KeyRelease>", lambda _: self._persist())

        self._notes_win = win
        win.protocol("WM_DELETE_WINDOW", self._toggle_notes_panel)

    def _toggle_json_panel(self):
        if self._json_win and self._json_win.winfo_exists():
            self._json_win.destroy()
            self._json_win = None
            return

        win = tk.Toplevel(self)
        win.title("Expected JSON  (AP)")
        win.configure(bg=BG)
        win.attributes("-topmost", True)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.70)
        x = (sw - w) // 2
        y = (sh - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")

        # header
        hdr = tk.Frame(win, bg=BG_ALT, pady=8)
        hdr.pack(fill="x")

        tk.Label(hdr, text="Expected JSON  (AP)", fg=ORANGE, bg=BG_ALT,
                 font=("Helvetica", 15, "bold")).pack(side="left", padx=14)

        self._json_status = tk.Label(hdr, text="", fg=GREEN, bg=BG_ALT,
                                      font=("Helvetica", 13))
        self._json_status.pack(side="left", padx=10)

        tk.Button(hdr, text="Save", command=self._save_expected_json,
                  bg=GREEN, fg=BG_ALT, relief="flat", padx=14, pady=5,
                  font=("Helvetica", 13, "bold"), cursor="hand2",
                  ).pack(side="right", padx=10)

        # text area
        edit = tk.Frame(win, bg=BG)
        edit.pack(fill="both", expand=True, padx=10, pady=8)

        vscroll = tk.Scrollbar(edit, orient="vertical", bg=SURFACE)
        vscroll.pack(side="right", fill="y")
        hscroll = tk.Scrollbar(edit, orient="horizontal", bg=SURFACE)
        hscroll.pack(side="bottom", fill="x")

        self._json_text = tk.Text(
            edit, bg=BG_ALT, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Courier", 16), wrap="none",
            padx=10, pady=8,
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )
        self._json_text.pack(fill="both", expand=True)
        vscroll.config(command=self._json_text.yview)
        hscroll.config(command=self._json_text.xview)

        self._json_win = win
        win.protocol("WM_DELETE_WINDOW", self._toggle_json_panel)

        self._load_expected_json()

    def _load_expected_json(self):
        if not self._json_text:
            return
        self._json_text.delete("1.0", "end")
        self._json_status.config(text="", fg=GREEN)
        if not self.prompts:
            return
        p = self.prompts[self.current_idx]
        prefix = f"{p['num']:03d}"
        jinja_path = p["folder"] / f"{prefix}_ap.jinja"
        if not jinja_path.exists():
            self._json_text.insert("1.0", "# No .jinja file found")
            return
        text = jinja_path.read_text(encoding="utf-8")
        marker = "\nAnswer:\n"
        idx = text.find(marker)
        if idx < 0:
            self._json_text.insert("1.0", "# 'Answer:' marker not found in .jinja")
            return
        json_str = text[idx + len(marker):].strip()
        try:
            parsed = json.loads(json_str)
            self._json_text.insert("1.0", json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            self._json_text.insert("1.0", json_str)

    def _save_expected_json(self):
        if not self.prompts or not self.run_dir:
            return
        p   = self.prompts[self.current_idx]
        num = p["num"]
        prefix = f"{num:03d}"
        jinja_path = p["folder"] / f"{prefix}_ap.jinja"
        yaml_path  = p["folder"] / f"{prefix}_ap.yaml"
        jsonl_path = self.run_dir / "training_samples.jsonl"

        raw = self._json_text.get("1.0", "end-1c").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            self._json_status.config(text=f"Invalid JSON: {e}", fg=RED)
            return

        canonical = json.dumps(parsed, indent=2, ensure_ascii=False)
        errors = []

        # 1. Update .jinja
        if jinja_path.exists():
            text = jinja_path.read_text(encoding="utf-8")
            marker = "\nAnswer:\n"
            idx = text.find(marker)
            if idx >= 0:
                jinja_path.write_text(
                    text[:idx + len(marker)] + canonical + "\n", encoding="utf-8"
                )
            else:
                errors.append("jinja: Answer marker not found")
        else:
            errors.append("jinja: file not found")

        # 2. Update _ap.yaml
        if yaml_path.exists():
            if _YAML_OK:
                try:
                    data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                    data["training_sample"]["output"] = canonical
                    yaml_path.write_text(
                        _yaml.dump(data, allow_unicode=True, default_flow_style=False,
                                   sort_keys=False, width=120),
                        encoding="utf-8",
                    )
                except Exception as e:
                    errors.append(f"yaml: {e}")
            else:
                errors.append("yaml: pyyaml not installed (skipped)")
        else:
            errors.append("yaml: file not found")

        # 3. Update training_samples.yaml
        samples_yaml_path = self.run_dir / "training_samples.yaml"
        if samples_yaml_path.exists():
            if _YAML_OK:
                try:
                    entries = _yaml.safe_load(samples_yaml_path.read_text(encoding="utf-8"))
                    updated = False
                    for entry in entries:
                        if (entry.get("domain_id") == num
                                and "_ap_" in entry.get("sample_id", "")):
                            entry["output"] = canonical
                            updated = True
                            break
                    if not updated:
                        errors.append("training_samples.yaml: matching entry not found")
                    else:
                        samples_yaml_path.write_text(
                            _yaml.dump(entries, allow_unicode=True,
                                       default_flow_style=False, sort_keys=False,
                                       width=120),
                            encoding="utf-8",
                        )
                except Exception as e:
                    errors.append(f"training_samples.yaml: {e}")
            else:
                errors.append("training_samples.yaml: pyyaml not installed (skipped)")
        else:
            errors.append("training_samples.yaml: file not found")

        # 4. Update training_samples.jsonl
        if jsonl_path.exists():
            try:
                lines = jsonl_path.read_text(encoding="utf-8").splitlines()
                new_lines = []
                updated = False
                for line in lines:
                    if not line.strip():
                        new_lines.append(line)
                        continue
                    obj = json.loads(line)
                    if (obj.get("domain_id") == num
                            and "_ap_" in obj.get("sample_id", "")):
                        for msg in obj["messages"]:
                            if msg["role"] == "assistant":
                                msg["content"] = canonical
                                break
                        new_lines.append(json.dumps(obj, ensure_ascii=False))
                        updated = True
                    else:
                        new_lines.append(line)
                if not updated:
                    errors.append("jsonl: matching entry not found")
                else:
                    jsonl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            except Exception as e:
                errors.append(f"jsonl: {e}")
        else:
            errors.append("jsonl: file not found")

        if errors:
            self._json_status.config(text="Errors: " + "; ".join(errors), fg=RED)
        else:
            self._json_status.config(text="Saved ✓", fg=GREEN)

    # ── AP PUML viewer ────────────────────────────────────────────────────────

    def _toggle_puml_panel(self):
        if self._puml_win and self._puml_win.winfo_exists():
            self._puml_win.destroy()
            self._puml_win = None
            return

        win = tk.Toplevel(self)
        win.title("AP Source  (.puml)")
        win.configure(bg=BG)
        win.attributes("-topmost", True)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.70)
        x = (sw - w) // 2
        y = (sh - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")

        hdr = tk.Frame(win, bg=BG_ALT, pady=8)
        hdr.pack(fill="x")

        tk.Label(hdr, text="AP Source  (.puml)", fg=BLUE, bg=BG_ALT,
                 font=("Helvetica", 15, "bold")).pack(side="left", padx=14)

        tk.Label(hdr, text="read-only · select text to copy", fg=SUBTEXT, bg=BG_ALT,
                 font=("Helvetica", 12)).pack(side="left", padx=6)

        edit = tk.Frame(win, bg=BG)
        edit.pack(fill="both", expand=True, padx=10, pady=8)

        vscroll = tk.Scrollbar(edit, orient="vertical", bg=SURFACE)
        vscroll.pack(side="right", fill="y")
        hscroll = tk.Scrollbar(edit, orient="horizontal", bg=SURFACE)
        hscroll.pack(side="bottom", fill="x")

        self._puml_text = tk.Text(
            edit, bg=BG_ALT, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Courier", 16), wrap="none",
            padx=10, pady=8,
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
            state="disabled",
        )
        self._puml_text.pack(fill="both", expand=True)
        vscroll.config(command=self._puml_text.yview)
        hscroll.config(command=self._puml_text.xview)

        self._puml_win = win
        win.protocol("WM_DELETE_WINDOW", self._toggle_puml_panel)

        self._load_puml()

    def _load_puml(self):
        if not self._puml_text:
            return
        if not self.prompts:
            return
        p = self.prompts[self.current_idx]
        prefix = f"{p['num']:03d}"
        puml_path = p["folder"] / f"{prefix}_ap.puml"

        self._puml_text.config(state="normal")
        self._puml_text.delete("1.0", "end")
        if puml_path.exists():
            self._puml_text.insert("1.0", puml_path.read_text(encoding="utf-8"))
            if self._puml_win:
                self._puml_win.title(f"AP Source  —  {prefix}_ap.puml")
        else:
            self._puml_text.insert("1.0", f"# File not found: {puml_path.name}")
        self._puml_text.config(state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = ReviewApp(run_dir)
    app.mainloop()
