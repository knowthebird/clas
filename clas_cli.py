#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Combination Lock Analysis Suite (CLAS)
#
# An open-source utility for recording, visualizing, and analyzing mechanical
# combination lock measurements for educational and locksport purposes.
#
# Copyright (C) 2026 knowthebird
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# USE POLICY:
# This software is intended ONLY for:
#   - Educational use
#   - Locksport
#   - Locksmith training
#   - Locks that you own or have explicit permission to work on
#
# Misuse of this software may violate local, state, or federal law.
# The authors and contributors accept no liability for misuse.
#
# Module: clas_cli.py
# Purpose: CLI adapter (terminal UI, save/recovery, global commands).
#
# This adapter owns ALL filesystem behavior (saving and recovery).
# The core engine should never write files or print to the terminal.

"""
CLAS CLI Adapter

This module provides the command-line interface for CLAS.

Responsibilities:
- Render prompts from clas_core.get_prompt()
- Read user input and translate it into core actions
- Implement global commands at any prompt:
    q = quit   s = save   u = undo   a = abort/back
- Persist the full session as ONE JSON file (manual save)
- Maintain a crash recovery mirror file alongside the session:
    session-name-YYYYMMDD-HHMMSS.json.recovery-data
  updated after every accepted action, deleted on clean exit.

Navigation guide (search for these headers / functions):
  - Session file I/O (atomic writes, loading)
  - Recovery mirror management
  - Global command handling
  - Main CLI loop (run_cli)
"""


from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional



def _plot_output_path(session_path: Path, kind: str, ident: int) -> Path:
    # Save plots next to the session file (per project convention).
    base = session_path.stem
    if kind == "sweep":
        return session_path.parent / f"{base}_sweep_{int(ident)}.png"
    if kind == "highlow":
        return session_path.parent / f"{base}_high_low_test_{int(ident)}.png"
    return session_path.parent / f"{base}-{kind}-{int(ident)}.png"


def _unwrap_series(vals, dial_min: float = 0.0, dial_max: float = 99.0, anchor=None):
    """Unwrap circular dial values so they plot continuously around an anchor."""
    try:
        n = float(dial_max - dial_min + 1)
        if n <= 0:
            n = 100.0
    except Exception:
        n = 100.0

    if vals is None or len(vals) == 0:
        return vals, 100

    if anchor is None:
        anchor = float(vals[0])

    out = []
    for v in vals:
        try:
            v = float(v)
            delta = v - anchor
            delta_wrapped = ((delta + n / 2.0) % n) - n / 2.0
            out.append(anchor + delta_wrapped)
        except Exception:
            out.append(v)
    return out, int(round(n))


def _format_dial_tick(y: float, n: int, dial_min: float = 0.0) -> str:
    try:
        v = (float(y) - dial_min) % n + dial_min
        return f"{v:.1f}"
    except Exception:
        return str(y)


def _plot_sweep_png(measurements: list, wheel_swept: int, sweep_id: int, out_path: Path, session_name: str = "") -> None:
    # Headless / CLI-safe plotting.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    if not measurements:
        return

    wheel = int(wheel_swept or measurements[0].get("wheel_swept", 0) or 0) or 1
    wheel_key = f"combination_wheel_{wheel}"

    # Extract arrays
    x = []
    lcp = []
    rcp = []
    for m in measurements:
        try:
            x.append(float(m.get(wheel_key)))
            lcp.append(float(m.get("left_contact")))
            rcp.append(float(m.get("right_contact")))
        except Exception:
            continue

    if not x:
        return

    width = []
    for m in measurements:
        try:
            width.append(float(m.get("contact_width", float(m.get("right_contact")) - float(m.get("left_contact")))))
        except Exception:
            width.append(0.0)

    data = sorted(zip(x, lcp, rcp, width), key=lambda t: t[0])
    x, lcp, rcp, width = map(np.array, zip(*data))
    # Unwrap circular contact points so plots don't jump across 0/99.
    dial_min = 0.0
    dial_max = 99.0
    try:
        lc = measurements[0].get("lock_config", {})
        if isinstance(lc, dict):
            dial_min = float(lc.get("dial_min", dial_min))
            dial_max = float(lc.get("dial_max", dial_max))
    except Exception:
        pass

    lcp_unwrapped, n_dial = _unwrap_series(list(lcp), dial_min=dial_min, dial_max=dial_max, anchor=float(lcp[0]))
    rcp_unwrapped, _ = _unwrap_series(list(rcp), dial_min=dial_min, dial_max=dial_max, anchor=float(rcp[0]))
    lcp = np.array(lcp_unwrapped, dtype=float)
    rcp = np.array(rcp_unwrapped, dtype=float)


    min_rcp_idx = int(np.argmin(rcp))
    min_width_idx = int(np.argmin(width))
    max_lcp_idx = int(np.argmax(lcp))

    def detect_gates(x_arr, width_arr):
        gates = []
        for i in range(1, len(width_arr) - 1):
            if width_arr[i] < width_arr[i - 1] and width_arr[i] < width_arr[i + 1]:
                depth = ((width_arr[i - 1] + width_arr[i + 1]) / 2.0) - width_arr[i]
                if depth > 0.3:
                    gates.append((x_arr[i], width_arr[i], depth))
        return gates

    gates = detect_gates(x, width)

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax_top.plot(x, rcp, marker="o")
    ax_top.scatter(x, rcp, marker="x")
    ax_top.scatter(x[min_rcp_idx], rcp[min_rcp_idx], color="red", s=120)
    ax_top.set_ylabel("Right Contact")
    ax_top.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_top.minorticks_on()
    ax_top.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _format_dial_tick(y, n_dial, dial_min)))

    ax_mid.plot(x, lcp, marker="o")
    ax_mid.scatter(x, lcp, marker="x")
    ax_mid.scatter(x[max_lcp_idx], lcp[max_lcp_idx], color="red", s=120)
    ax_mid.set_ylabel("Left Contact")
    ax_mid.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_mid.minorticks_on()
    ax_mid.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_mid.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _format_dial_tick(y, n_dial, dial_min)))

    # Set y-limits to a tight window around the data for readability (avoids tiny-looking changes).
    try:
        pad = max(1.0, 0.05 * float(max(rcp) - min(rcp)))
        ax_top.set_ylim(float(min(rcp)) - pad, float(max(rcp)) + pad)
    except Exception:
        pass
    try:
        pad = max(1.0, 0.05 * float(max(lcp) - min(lcp)))
        ax_mid.set_ylim(float(min(lcp)) - pad, float(max(lcp)) + pad)
    except Exception:
        pass

    ax_bot.plot(x, width, marker="o")
    ax_bot.scatter(x, width, marker="x")
    ax_bot.scatter(x[min_width_idx], width[min_width_idx], color="red", s=120)
    for gx, gw, strength in gates:
        ax_bot.scatter(gx, gw, color="gold", s=160, marker="*", zorder=5)

    ax_bot.set_ylabel("Contact Width")
    ax_bot.set_xlabel("Dial Position")
    ax_bot.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_bot.minorticks_on()
    ax_bot.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _format_dial_tick(y, n_dial, dial_min)))
    try:
        ax_bot.set_ylim(0, float(max(width)) * 1.1 if len(width) else 1)
    except Exception:
        pass

    # Adaptive x ticks
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = xmax - xmin
    if span <= 20:
        major_step = 1
    elif span <= 50:
        major_step = 2
    else:
        major_step = 5

    major_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, major_step)
    minor_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, major_step / 5.0)
    ax_bot.set_xticks(major_ticks)
    ax_bot.set_xticks(minor_ticks, minor=True)

    for label in ax_bot.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # Combination label from first row if present
    combo = []
    for i in (1, 2, 3):
        k = f"combination_wheel_{i}"
        if k in measurements[0]:
            combo.append(measurements[0].get(k))
    title_session = session_name or ""
    fig.suptitle(
        f"Session: {title_session} | Sweep: {int(sweep_id)}\n"
        f"Wheel swept: {wheel}   |   Combination: {combo}",
        fontsize=12
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_high_low_png(measurements: list, test_id: int, out_path: Path, session_name: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    data = [m for m in measurements if str(m.get("high_low_test", "")) == str(test_id)]
    if not data:
        return

    high = [m for m in data if str(m.get("hw_type", "")).strip().lower() == "high"]
    low = [m for m in data if str(m.get("hw_type", "")).strip().lower() == "low"]
    dial_min = 0.0
    dial_max = 99.0
    try:
        lc0 = data[0].get("lock_config", {})
        if isinstance(lc0, dict):
            dial_min = float(lc0.get("dial_min", dial_min))
            dial_max = float(lc0.get("dial_max", dial_max))
    except Exception:
        pass
    n_dial = float(dial_max - dial_min + 1)

    def combo_label(m):
        c1 = m.get("combination_wheel_1", "")
        c2 = m.get("combination_wheel_2", "")
        c3 = m.get("combination_wheel_3", "")
        return f"{c1}, {c2}, {c3}"

    def extract(meas):
        if not meas:
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        l_raw = [float(m["left_contact"]) for m in meas]
        r_raw = [float(m["right_contact"]) for m in meas]

        l_unwrapped, n_dial = _unwrap_series(l_raw, dial_min=dial_min, dial_max=dial_max, anchor=float(l_raw[0]) if l_raw else None)
        r_unwrapped, _ = _unwrap_series(r_raw, dial_min=dial_min, dial_max=dial_max, anchor=float(r_raw[0]) if r_raw else None)

        l = np.array(l_unwrapped, dtype=float)
        r = np.array(r_unwrapped, dtype=float)
        w = np.array([float(m.get("contact_width", float(m["right_contact"]) - float(m["left_contact"]))) for m in meas], dtype=float)
        x = np.arange(1, len(meas) + 1, dtype=int)
        labels = [combo_label(m) for m in meas]
        return x, l, r, w, labels

    def unique_extreme_index(values: np.ndarray, mode: str):
        if values is None or len(values) < 2:
            return None
        if mode == "max":
            vmax = np.max(values)
            idxs = np.where(values == vmax)[0]
            return int(idxs[0]) if len(idxs) == 1 else None
        if mode == "min":
            vmin = np.min(values)
            idxs = np.where(values == vmin)[0]
            return int(idxs[0]) if len(idxs) == 1 else None
        return None

    def draw(ax, x, y, ylabel):
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.plot(x, y, marker="o")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda yy, _: _format_dial_tick(yy, n_dial, dial_min)))

    def apply_combo_ticks(ax, x, labels):
        if len(x) == 0:
            return
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Combination Tried (W1, W2, W3)")

    def highlight_if_unique(ax, x, y, mode: str):
        idx = unique_extreme_index(y, mode)
        if idx is None:
            return
        ax.scatter([x[idx]], [y[idx]], color="red", s=140, zorder=5)

    xh, lh, rh, wh, lab_h = extract(high)
    xl, ll, rl, wl, lab_l = extract(low)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)

    draw(axes[0, 0], xh, rh, "RCP (+)")
    draw(axes[0, 1], xl, rl, "RCP (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[0, 0], xh, rh, mode="max")
    if len(xl) > 0:
        highlight_if_unique(axes[0, 1], xl, rl, mode="max")
    apply_combo_ticks(axes[0, 0], xh, lab_h)
    apply_combo_ticks(axes[0, 1], xl, lab_l)

    draw(axes[1, 0], xh, lh, "LCP (+)")
    draw(axes[1, 1], xl, ll, "LCP (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[1, 0], xh, lh, mode="min")
    if len(xl) > 0:
        highlight_if_unique(axes[1, 1], xl, ll, mode="min")
    apply_combo_ticks(axes[1, 0], xh, lab_h)
    apply_combo_ticks(axes[1, 1], xl, lab_l)

    draw(axes[2, 0], xh, wh, "Width (+)")
    draw(axes[2, 1], xl, wl, "Width (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[2, 0], xh, wh, mode="max")
    if len(xl) > 0:
        highlight_if_unique(axes[2, 1], xl, wl, mode="max")
    apply_combo_ticks(axes[2, 0], xh, lab_h)
    apply_combo_ticks(axes[2, 1], xl, lab_l)

    gate = data[0].get("hw_gate", "")
    inc = data[0].get("hw_increment", "")

    fig.suptitle(
        f"Session: {session_name}   High-Wheel Test {int(test_id)}\n"
        f"Gate = {gate}   Increment = ±{inc}",
        fontsize=12
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

def _maybe_save_graph(prompt: dict, data: dict, session_path: Path) -> None:
    """Save graphs at key points (scan complete / plot screens)."""
    pid = str(prompt.get("id", ""))
    frame = data["runtime"]["stack"][-1]
    ctx = frame.get("ctx", {}) or {}

    def _confirm_plot_path(out_path: Path) -> Optional[Path]:
        if not out_path.exists():
            return out_path
        print(f"[Plot exists] {out_path}")
        print("  1) Overwrite")
        print("  2) Save with new name")
        print("  3) Cancel")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            return out_path
        if choice == "2":
            new_name = input("Enter new filename (with .png): ").strip()
            if not new_name:
                ctx["_skip_plot_once"] = True
                return None
            return out_path.parent / new_name
        ctx["_skip_plot_once"] = True
        return None

    try:
        if ctx.pop("_skip_plot_once", False):
            return
        if pid in ("iso2.candidates", "iso2.plot", "iso2.post_refine.plot", "iso3.candidates", "iso3.plot", "iso3.post_refine.plot"):
            sweep_id = int(ctx.get("sweep_id"))
            wheel_swept = int(ctx.get("wheel_swept", 0) or 0)
            rows = ctx.get("rows", []) or ctx.get("scan_rows", []) or []
            out_path = _plot_output_path(session_path, "sweep", sweep_id)
            out_path = _confirm_plot_path(out_path)
            if out_path is None:
                return
            _plot_sweep_png(rows, wheel_swept, sweep_id, out_path, session_name=str(data.get('state',{}).get('session_name','')))
            print(f"[Plot saved] {out_path}")
            print("Press Enter to continue.")
            return

        if pid in ("iso2.finish", "iso3.finish", "plot_sweep.generate"):
            if pid in ("iso2.finish", "iso3.finish") and ctx.pop("_skip_finish_plot_once", False):
                return
            sweep_id = int(ctx.get("sweep_id"))
            wheel_swept = int(ctx.get("wheel_swept", 0) or 0)
            ms = [
                m for m in data["state"]["measurements"]
                if str(m.get("sweep", "")).replace(".", "", 1).isdigit() and int(float(m.get("sweep"))) == sweep_id
            ]
            out_path = _plot_output_path(session_path, "sweep", sweep_id)
            out_path = _confirm_plot_path(out_path)
            if out_path is None:
                return
            _plot_sweep_png(ms, wheel_swept, sweep_id, out_path, session_name=str(data.get('state',{}).get('session_name','')))
            print(f"[Plot saved] {out_path}")
            print("Press Enter to continue.")
            return

        if pid == "plot_high_low.generate":
            test_id = int(ctx.get("test_id"))
            out_path = _plot_output_path(session_path, "highlow", test_id)
            out_path = _confirm_plot_path(out_path)
            if out_path is None:
                return
            _plot_high_low_png(data["state"]["measurements"], test_id, out_path, session_name=str(data.get('state',{}).get('session_name','')))
            print(f"[Plot saved] {out_path}")
            print("Press Enter to continue.")
            return
    except Exception as e:
        print(f"[Plot error] {e}")
        return


import clas_core as core


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _is_recovery_file(p: Path) -> bool:
    return p.name.endswith(".json.recovery-data")


def _list_recovery_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.json.recovery-data"))


def _derive_base_session_name_from_recovery(recovery_path: Path) -> str:
    # session-name-<timestamp>.json.recovery-data  -> session-name.json
    name = recovery_path.name
    # Remove suffix
    if name.endswith(".json.recovery-data"):
        name = name[:-len(".json.recovery-data")]
    # Split last '-' chunk as timestamp (best-effort)
    if "-" in name:
        base = "-".join(name.split("-")[:-1])
    else:
        base = name
    return base + ".json"


def _make_recovery_path(session_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = session_path.stem
    return session_path.with_name(f"{base}-{ts}.json.recovery-data")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_session_interactive(folder: Path) -> tuple[Path, Optional[Path], Dict[str, Any]]:
    """
    Returns (session_path, recovery_path, session_data)
    - session_path: where manual saves go
    - recovery_path: active recovery mirror file path
    - session_data: loaded session dict (normalized + rebuilt)
    """
    folder.mkdir(parents=True, exist_ok=True)

    recovery_files = _list_recovery_files(folder)
    session_files = sorted([p for p in folder.glob("*.json") if p.is_file() and not _is_recovery_file(p)])

    print("\n=== CLAS (CLI) ===")
    print("Global commands anytime: u=undo, s=save, a=abort, q=quit\n")

    options: list[tuple[str, str, str]] = []
    # Add recovery options first
    for rf in recovery_files:
        base = _derive_base_session_name_from_recovery(rf)
        label = f"RECOVER from {rf.name} (base session: {base})"
        options.append(("recovery", str(rf), label))

    for sf in session_files:
        options.append(("session", str(sf), f"Open session {sf.name}"))

    options.append(("new", "", "Create a new session"))

    for i, (_, _, label) in enumerate(options, 1):
        print(f"{i}) {label}")

    while True:
        raw = input("\nChoose an option number: ").strip()
        if raw.lower() in ("q", "quit"):
            raise SystemExit(0)
        try:
            idx = int(raw)
        except Exception:
            print("Enter a valid option number.")
            continue
        if idx < 1 or idx > len(options):
            print("Enter a valid option number.")
            continue
        kind, path_s, _ = options[idx-1]

        if kind == "new":
            name = input("Enter new session name (no extension): ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            session_path = folder / f"{name}.json"
            if session_path.exists():
                print("That session already exists.")
                continue
            sess = core.new_session(name)
            sess.setdefault("meta", {})["show_splash"] = True
            sess = core.normalize_session(sess)
            sess = core.rebuild(sess)
            recovery_path = _make_recovery_path(session_path)
            return session_path, recovery_path, sess

        if kind == "session":
            session_path = Path(path_s)
            data = core.normalize_session(_load_json(session_path))
            data = core.rebuild(data)
            recovery_path = _make_recovery_path(session_path)
            return session_path, recovery_path, data

        if kind == "recovery":
            recovery_path = Path(path_s)
            data = core.normalize_session(_load_json(recovery_path))
            data = core.rebuild(data)
            # determine base session path
            base_name = _derive_base_session_name_from_recovery(recovery_path)
            session_path = folder / base_name
            # keep existing recovery timestamp filename (do not change timestamp)
            return session_path, recovery_path, data


def _render_prompt(prompt: Dict[str, Any], last_error: Optional[str]) -> None:
    if last_error:
        print(f"\n[Error] {last_error}\n")

    text = prompt.get("text", "")
    print("\n" + text)

    if prompt.get("kind") == "choice":
        choices = prompt.get("choices", []) or []
        for c in choices:
            if isinstance(c, dict):
                print(f"  {c.get('key')}) {c.get('label')}")

    help_text = prompt.get("help")
    if help_text:
        print(f"\n({help_text})")


def _show_first_time_splash() -> None:
    print("\nWelcome to CLAS.")
    print("Global commands are available at any time:")
    print("  (s)ave session to a JSON file")
    print("  (u)ndo the last command or value you entered")
    print("  (a)bort the current workflow")
    print("  (q)uit the program")
    print("\nTutorial and Learn more resources are available for help.")
    input("\nPress Enter to continue.")


def _read_user_input(prompt: Dict[str, Any]) -> str:
    # show default inline if present and kind isn't confirm/choice
    kind = prompt.get("kind")
    default = prompt.get("default", None)
    if kind in ("float","int","text","csv_floats","bool_yn") and default is not None:
        raw = input(f"> [{default}] ").rstrip("\n")
        if raw.strip() == "":
            return ""  # allow core to apply default
        return raw
    return input("> ").rstrip("\n")


def _interpret_global_commands(raw: str, prompt_kind: str) -> Optional[Dict[str, Any]]:
    """
    Returns an action dict for core, or None if it's normal input.
    Save and quit are handled in adapter, not passed to core.
    """
    r = raw.strip()

    # allow literal command letters for text prompts via escaping
    if prompt_kind == "text" and r.startswith("\\") and len(r) == 2 and r[1] in ("q","s","u","a"):
        return None  # treat as normal input; caller will strip backslash

    # commands (single letter or word)
    low = r.lower()
    if low in ("u", "undo"):
        return {"type":"command","name":"undo"}
    if low in ("a", "abort"):
        return {"type":"command","name":"abort"}
    # save/quit handled separately
    return None


def _is_save(raw: str) -> bool:
    return raw.strip() in ("s", "S") or raw.strip().lower() == "save"


def _is_quit(raw: str) -> bool:
    return raw.strip() in ("q", "Q") or raw.strip().lower() == "quit"


def run_cli(session_path: Optional[str] = None) -> None:
    folder = Path(".").resolve()

    if session_path:
        sp = Path(session_path).resolve()
        data = core.normalize_session(_load_json(sp))
        data = core.rebuild(data)
        session_path_p = sp
        recovery_path_p = _make_recovery_path(session_path_p)
    else:
        session_path_p, recovery_path_p, data = _select_session_interactive(folder)

    # Track whether the session has unsaved changes.
    # If this session file doesn't exist yet, treat it as dirty until first save.
    if session_path_p.exists():
        last_saved_cursor = int(data.get("meta", {}).get("last_saved_cursor", data.get("cursor", 0)) or 0)
    else:
        last_saved_cursor = -1

    # Immediately write initial recovery mirror
    _write_json_atomic(recovery_path_p, data)

    if data.get("meta", {}).pop("show_splash", False):
        _show_first_time_splash()
        _write_json_atomic(recovery_path_p, data)

    try:
        while True:
            prompt = core.get_prompt(data)
            frame = data["runtime"]["stack"][-1]
            last_error = frame["ctx"].get("_last_error")

            _render_prompt(prompt, last_error)
            _maybe_save_graph(prompt, data, session_path_p)

            raw = _read_user_input(prompt)

            # literal escape handling for text prompts (\q, \s, \u, \a)
            if prompt.get("kind") == "text" and raw.strip().startswith("\\") and len(raw.strip()) == 2:
                raw = raw.strip()[1:]

            # Write the pre-action recovery mirror so a crash mid-action can be recovered.
            _write_json_atomic(recovery_path_p, data)

            # quit (adapter-handled)
            if _is_quit(raw):
                dirty = (int(data.get("cursor", 0) or 0) != int(last_saved_cursor))
                if dirty:
                    ans = input("Save changes before quitting? (y/n): ").strip().lower()
                    if ans.startswith("y"):
                        data = core.normalize_session(data)
                        data.setdefault("meta", {})
                        data["meta"]["last_saved_cursor"] = int(data.get("cursor", 0) or 0)
                        last_saved_cursor = int(data["meta"]["last_saved_cursor"])
                        _write_json_atomic(session_path_p, data)
                        print(f"[Saved] {session_path_p.name}")
                _safe_unlink(recovery_path_p)
                print("Goodbye.")
                return

            # save (adapter-handled)
            if _is_save(raw):
                data = core.normalize_session(data)
                data.setdefault("meta", {})
                data["meta"]["last_saved_cursor"] = int(data.get("cursor", 0) or 0)
                last_saved_cursor = int(data["meta"]["last_saved_cursor"])
                _write_json_atomic(session_path_p, data)
                _write_json_atomic(recovery_path_p, data)
                print(f"[Saved] {session_path_p.name}")
                continue  # reprompt same prompt

            # undo/abort commands into core
            cmd_action = _interpret_global_commands(raw, prompt.get("kind", "text"))
            if cmd_action is not None:
                data = core.apply_action(data, cmd_action)
                _write_json_atomic(recovery_path_p, data)
                continue

            # normal input
            data = core.apply_action(data, {"type": "input", "text": raw})
            _write_json_atomic(recovery_path_p, data)

    except KeyboardInterrupt:
        print("\nInterrupted. Recovery file preserved for crash-style recovery.")
        print(f"Recovery: {recovery_path_p.name}")
        return
    except SystemExit:
        _safe_unlink(recovery_path_p)
        raise
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Recovery file preserved:")
        print(f"  {recovery_path_p.name}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="CLAS (CLI adapter) - refactored core/adapter architecture")
    ap.add_argument("--session", help="Path to session JSON file (optional). If omitted, interactive picker is used.")
    args = ap.parse_args()
    run_cli(args.session)


if __name__ == "__main__":
    main()
