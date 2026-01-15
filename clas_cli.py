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
    q = quit   s = save   u = undo   e = exit/back
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
    return session_path.parent / f"{base}-{kind}-{int(ident)}.png"


def _plot_sweep_png(measurements: list, wheel_swept: int, sweep_id: int, out_path: Path) -> None:
    # Lazy import so CLI still works in minimal environments.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_field = f"combination_wheel_{int(wheel_swept)}"
    if not any(x_field in m for m in measurements):
        for f in ("combination_wheel_3", "combination_wheel_2", "combination_wheel_1"):
            if any(f in m for m in measurements):
                x_field = f
                break

    pts = []
    for m in measurements:
        try:
            x = float(m.get(x_field))
            l = float(m.get("left_contact"))
            r = float(m.get("right_contact"))
            pts.append((x, l, r))
        except Exception:
            continue
    pts.sort(key=lambda t: t[0])
    if not pts:
        return

    xs = [p[0] for p in pts]
    lcp = [p[1] for p in pts]
    rcp = [p[2] for p in pts]

    fig = plt.figure()
    plt.plot(xs, lcp, marker="o", linestyle="-", label="LCP")
    plt.plot(xs, rcp, marker="o", linestyle="-", label="RCP")
    plt.title(f"Sweep {int(sweep_id)} (Wheel {int(wheel_swept)} swept)")
    plt.xlabel(x_field)
    plt.ylabel("Contact point")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_high_low_png(measurements: list, test_id: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for m in measurements:
        try:
            if int(float(m.get("high_low_test"))) != int(test_id):
                continue
            x = float(m.get("hw_gate"))
            l = float(m.get("left_contact"))
            r = float(m.get("right_contact"))
            pts.append((x, l, r))
        except Exception:
            continue
    pts.sort(key=lambda t: t[0])
    if not pts:
        return

    xs = [p[0] for p in pts]
    lcp = [p[1] for p in pts]
    rcp = [p[2] for p in pts]

    fig = plt.figure()
    plt.plot(xs, lcp, marker="o", linestyle="-", label="LCP")
    plt.plot(xs, rcp, marker="o", linestyle="-", label="RCP")
    plt.title(f"High Low Test {int(test_id)}")
    plt.xlabel("Gate position")
    plt.ylabel("Contact point")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _maybe_save_graph(prompt: dict, data: dict, session_path: Path) -> None:
    """Save graphs at key points (scan complete / plot screens)."""
    pid = str(prompt.get("id", ""))
    frame = data["runtime"]["stack"][-1]
    ctx = frame.get("ctx", {}) or {}

    try:
        if pid in ("iso2.candidates", "iso3.candidates"):
            sweep_id = int(ctx.get("sweep_id"))
            wheel_swept = int(ctx.get("wheel_swept", 0) or 0)
            rows = ctx.get("rows", []) or []
            out_path = _plot_output_path(session_path, "sweep", sweep_id)
            _plot_sweep_png(rows, wheel_swept, sweep_id, out_path)
            return

        if pid in ("iso2.finish", "iso3.finish", "plot_sweep.generate"):
            sweep_id = int(ctx.get("sweep_id"))
            wheel_swept = int(ctx.get("wheel_swept", 0) or 0)
            ms = [
                m for m in data["state"]["measurements"]
                if str(m.get("sweep", "")).replace(".", "", 1).isdigit() and int(float(m.get("sweep"))) == sweep_id
            ]
            out_path = _plot_output_path(session_path, "sweep", sweep_id)
            _plot_sweep_png(ms, wheel_swept, sweep_id, out_path)
            return

        if pid == "plot_high_low.generate":
            test_id = int(ctx.get("test_id"))
            out_path = _plot_output_path(session_path, "highlow", test_id)
            _plot_high_low_png(data["state"]["measurements"], test_id, out_path)
            return
    except Exception:
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
    print("Global commands anytime: u=undo, s=save, e=exit, q=quit\n")

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
    if prompt_kind == "text" and r.startswith("\\") and len(r) == 2 and r[1] in ("q","s","u","e"):
        return None  # treat as normal input; caller will strip backslash

    # commands (single letter or word)
    low = r.lower()
    if low in ("u", "undo"):
        return {"type":"command","name":"undo"}
    if low in ("e", "exit"):
        return {"type":"command","name":"exit"}
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

    try:
        while True:
            prompt = core.get_prompt(data)
            frame = data["runtime"]["stack"][-1]
            last_error = frame["ctx"].get("_last_error")

            _render_prompt(prompt, last_error)
            _maybe_save_graph(prompt, data, session_path_p)

            raw = _read_user_input(prompt)

            # literal escape handling for text prompts (\q, \s, \u, \e)
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

            # undo/exit commands into core
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
