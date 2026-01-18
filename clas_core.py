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
# Module: clas_core.py
# Purpose: Core engine (portable, UI-agnostic state machine).
#
# This file contains the “business logic” of CLAS and MUST remain portable.
# Any CLI/web specific behavior (printing, file paths, backups, web requests) belongs in an adapter.

"""
CLAS Core Engine

This module implements the UI-agnostic “prompt → action → prompt” engine used by CLAS.

Design goals:
- No direct I/O: no terminal printing, no interactive input, and no filesystem writes.
- Deterministic replay: the session records all user actions, and the engine can rebuild state
  by replaying history (enabling unlimited undo and crash-safe recovery when paired with an adapter).
- Adapter-friendly: a CLI adapter or a web adapter can drive the same workflows by calling:
    - get_prompt(session)  → dict describing what to show
    - apply_action(session, action) → updated session dict

Navigation guide (search for these headers / functions):
  - Session schema and normalization (new_session, normalize_session)
  - Prompt generation (get_prompt, _prompt_*)
  - Action application and replay (apply_action, rebuild, _apply_event)
  - Core utilities (dial math, distances, parsing helpers)
  - Workflows / tests (AWL/AWR, wheel isolation, enumeration, etc.)
"""


from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

Session = Dict[str, Any]
PromptSpec = Dict[str, Any]
Action = Dict[str, Any]

CLAS_CORE_VERSION = "0.2.5"
FLOAT_DISPLAY_PRECISION = 2

# -----------------------
# Lock config helpers
# -----------------------

def default_lock_config() -> Dict[str, Any]:
    return {
        "wheels": 3,
        "dial_min": 0.0,
        "dial_max": 99.0,
        "tolerance": 1.0,
        "turn_sequence": "LRL",   # "LRL" or "RLR"
        "flies": "fixed",         # "fixed" or "moveable"
        "make": "UNKNOWN",
        "fence_type": "UNKNOWN",  # FRICTION_FENCE / GRAVITY_LEVER / SPRING_LEVER / UNKNOWN
        "ul": "UNKNOWN",          # 2 / 2M / 1 / 1R / UNKNOWN
        "oval_wheels": "UNKNOWN", # YES / NO / UNKNOWN

        # optional low points / approximations
        "awl_low_point": None,
        "awr_low_point": None,
        "approx_lcp_location": None,
        "approx_rcp_location": None,

        # per-wheel data as strings "1", "2", "3", ...
        "wheel_data": {},
    }


def normalize_lock_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(default_lock_config() if not isinstance(cfg, dict) else cfg)

    # normalize basics
    try:
        cfg["wheels"] = int(cfg.get("wheels", 3) or 3)
    except Exception:
        cfg["wheels"] = 3

    for k in ("dial_min", "dial_max", "tolerance"):
        try:
            cfg[k] = float(cfg.get(k, default_lock_config()[k]))
        except Exception:
            cfg[k] = float(default_lock_config()[k])

    if cfg["tolerance"] <= 0:
        cfg["tolerance"] = 1.0

    ts = str(cfg.get("turn_sequence", "LRL") or "LRL").strip().upper()
    cfg["turn_sequence"] = ts if ts in ("LRL", "RLR") else "LRL"

    # flies: fixed/moveable
    fv = cfg.get("flies", default_lock_config()["flies"])
    if isinstance(fv, bool):
        cfg["flies"] = "moveable" if fv else "fixed"
    else:
        s = str(fv).strip().lower()
        if s in ("moveable", "movable", "move", "m"):
            cfg["flies"] = "moveable"
        elif s in ("fixed", "fix", "f"):
            cfg["flies"] = "fixed"
        else:
            cfg["flies"] = str(default_lock_config()["flies"])

    # strings / enums
    cfg["make"] = str(cfg.get("make", "UNKNOWN") or "UNKNOWN")

    ft = str(cfg.get("fence_type", "UNKNOWN") or "UNKNOWN").strip().upper()
    if ft not in ("FRICTION_FENCE", "GRAVITY_LEVER", "SPRING_LEVER", "UNKNOWN"):
        ft = "UNKNOWN"
    cfg["fence_type"] = ft

    ul = str(cfg.get("ul", "UNKNOWN") or "UNKNOWN").strip().upper()
    if ul not in ("2", "2M", "1", "1R", "UNKNOWN"):
        ul = "UNKNOWN"
    cfg["ul"] = ul

    ow = str(cfg.get("oval_wheels", "UNKNOWN") or "UNKNOWN").strip().upper()
    if ow not in ("YES", "NO", "UNKNOWN"):
        ow = "UNKNOWN"
    cfg["oval_wheels"] = ow

    # wheel_data buckets
    wd = cfg.get("wheel_data", {})
    if not isinstance(wd, dict):
        wd = {}
    for w in range(1, cfg["wheels"] + 1):
        key = str(w)
        wdata = wd.get(key, {}) or {}
        if not isinstance(wdata, dict):
            wdata = {}
        for lk in ("known_gates", "suspected_gates", "false_gates"):
            v = wdata.get(lk, [])
            if v is None:
                v = []
            if not isinstance(v, list):
                v = [v]
            # coerce to floats where possible
            out = []
            for item in v:
                try:
                    out.append(float(item))
                except Exception:
                    pass
            wdata[lk] = out
        wd[key] = wdata
    cfg["wheel_data"] = wd

    # optional floats may be None or float
    for k in ("awl_low_point", "awr_low_point", "approx_lcp_location", "approx_rcp_location"):
        v = cfg.get(k, None)
        if v is None or v == "":
            cfg[k] = None
        else:
            try:
                cfg[k] = float(v)
            except Exception:
                cfg[k] = None

    return cfg


def _format_wheel_data(lock_config: Dict[str, Any]) -> str:
    """Human-readable per-wheel gates for config display."""
    try:
        wheels = int(lock_config.get("wheels", 0) or 0)
    except Exception:
        wheels = 0
    wd = lock_config.get("wheel_data", {}) or {}
    lines = []
    for w in range(1, max(0, wheels) + 1):
        wdata = wd.get(str(w), {}) or {}
        kg = wdata.get("known_gates", []) or []
        sg = wdata.get("suspected_gates", []) or []
        fg = wdata.get("false_gates", []) or []
        lines.append(
            "    wheel {w}: known={kg} suspected={sg} false={fg}".format(
                w=w,
                kg=_fmt_float_list(kg),
                sg=_fmt_float_list(sg),
                fg=_fmt_float_list(fg),
            )
        )
    return "\n".join(lines) if lines else "    (none)"

def _fmt_float(val: Any, digits: Optional[int] = None) -> str:
    if digits is None:
        digits = FLOAT_DISPLAY_PRECISION
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)


def _fmt_float_list(vals: List[Any], digits: Optional[int] = None) -> str:
    return "[" + ", ".join(_fmt_float(v, digits) for v in vals) + "]"


def _fmt_float_tuple(vals: List[Any], digits: Optional[int] = None) -> str:
    return "(" + ", ".join(_fmt_float(v, digits) for v in vals) + ")"


# -----------------------
# Dial math helpers
# -----------------------

def wrap_dial(x: float, dial_min: float, dial_max: float) -> float:
    span = (dial_max - dial_min) + 1.0
    if span <= 0:
        return float(x)
    y = (x - dial_min) % span
    return dial_min + y


def circular_distance(a: float, b: float, dial_min: float, dial_max: float) -> float:
    span = (dial_max - dial_min) + 1.0
    if span <= 0:
        return 0.0
    # distance moving CW from b -> a (matching original style)
    d = (a - b) % span
    return float(d)


def _is_between_cw(start: float, end: float, x: float, dial_min: float, dial_max: float) -> bool:
    """True if x lies on the CW arc after start and before end."""
    dist_end = circular_distance(end, start, dial_min, dial_max)
    dist_x = circular_distance(x, start, dial_min, dial_max)
    return 0.0 < dist_x < dist_end


def _build_range_points(start: float, end: float, n_points: int, dial_min: float, dial_max: float) -> List[float]:
    try:
        n = int(n_points or 0)
    except Exception:
        n = 0
    if n <= 1:
        return [wrap_dial(float(start), dial_min, dial_max)]
    step = (float(end) - float(start)) / float(n - 1)
    return [wrap_dial(float(start + (i * step)), dial_min, dial_max) for i in range(n)]


def build_checkpoints(dial_min: float, dial_max: float, n_points: int = 10) -> List[float]:
    """
    Build evenly-spaced checkpoints for AWL/AWR scans.

    New spacing rule:
      step = dial_span / n_points
    where:
      dial_span = (dial_max - dial_min) + 1
    """

    try:
        n = int(n_points or 0)
    except Exception:
        n = 10

    if n < 1:
        n = 1
    if n > 36000:
        n = 36000

    if n == 1:
        return [float(dial_min)]

    dial_span = float((dial_max - dial_min) + 1.0)
    if dial_span <= 0:
        return [float(dial_min)]

    step = dial_span / float(n)
    return [wrap_dial(float(dial_min + (i * step)), dial_min, dial_max) for i in range(n)]


# -----------------------
# Session schema helpers
# -----------------------

def new_session(session_name: str = "session") -> Session:
    return {
        "version": CLAS_CORE_VERSION,
        "meta": {"session_name": session_name, "baseline_lock_config": default_lock_config()},
        "state": {
            "lock_config": normalize_lock_config(default_lock_config()),
            "measurements": [],
            "metadata": {"float_display_precision": FLOAT_DISPLAY_PRECISION},
        },
        # stack-based navigation; top of stack is active
        "runtime": {
            "stack": [{"screen": "main_menu", "ctx": {}}]
        },
        "history": [],
        "cursor": 0,
    }


def normalize_session(session: Session) -> Session:
    """
    Normalize/upgrade a loaded session. Also supports importing old-format sessions
    from earlier monolithic versions (top-level lock_config/measurements/session_name).
    """
    global FLOAT_DISPLAY_PRECISION
    if not isinstance(session, dict):
        return new_session("session")

    # old-format import
    if "state" not in session and "lock_config" in session and "measurements" in session:
        name = str(session.get("session_name", "session") or "session")
        sess = new_session(name)
        sess["state"]["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
        sess["state"]["measurements"] = list(session.get("measurements", []) or [])
        sess["state"]["metadata"] = dict(session.get("metadata", {}) or {})
        return sess

    sess = dict(session)
    if sess.get("version") != CLAS_CORE_VERSION:
        sess["version"] = CLAS_CORE_VERSION

    meta = sess.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    if not meta.get("session_name"):
        meta["session_name"] = "session"
    sess["meta"] = meta

    state = sess.get("state", {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("measurements", [])
    if not isinstance(state["measurements"], list):
        state["measurements"] = []
    state.setdefault("metadata", {})
    if not isinstance(state["metadata"], dict):
        state["metadata"] = {}
    try:
        precision = int(state["metadata"].get("float_display_precision", FLOAT_DISPLAY_PRECISION))
    except Exception:
        precision = FLOAT_DISPLAY_PRECISION
    precision = max(0, min(precision, 10))
    state["metadata"]["float_display_precision"] = precision
    FLOAT_DISPLAY_PRECISION = precision
    state["lock_config"] = normalize_lock_config(state.get("lock_config", {}))
    sess["state"] = state

    if "baseline_lock_config" not in meta:
        if not sess.get("history"):
            meta["baseline_lock_config"] = dict(state.get("lock_config", default_lock_config()))
        else:
            meta["baseline_lock_config"] = default_lock_config()

    rt = sess.get("runtime", {})
    if not isinstance(rt, dict):
        rt = {}
    stack = rt.get("stack", [])
    if not isinstance(stack, list) or not stack:
        stack = [{"screen": "main_menu", "ctx": {}}]
    # ensure frames are dicts
    norm_stack = []
    for fr in stack:
        if not isinstance(fr, dict):
            continue
        scr = fr.get("screen", "main_menu")
        ctx = fr.get("ctx", {})
        if not isinstance(ctx, dict):
            ctx = {}
        norm_stack.append({"screen": scr, "ctx": ctx})
    if not norm_stack:
        norm_stack = [{"screen": "main_menu", "ctx": {}}]
    sess["runtime"] = {"stack": norm_stack}

    hist = sess.get("history", [])
    if not isinstance(hist, list):
        hist = []
    sess["history"] = hist

    try:
        cursor = int(sess.get("cursor", len(hist)) or 0)
    except Exception:
        cursor = len(hist)
    cursor = max(0, min(cursor, len(hist)))
    sess["cursor"] = cursor

    return sess


# -----------------------
# Prompt parsing helpers
# -----------------------

def _parse_float(raw: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        return float(raw), None
    except Exception:
        return None, "Expected a number."


def _parse_int(raw: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        return int(raw), None
    except Exception:
        return None, "Expected an integer."


def _parse_bool_yn(raw: str) -> Tuple[Optional[bool], Optional[str]]:
    r = raw.strip().lower()
    if r in ("y", "yes", "1", "true", "t"):
        return True, None
    if r in ("n", "no", "0", "false", "f"):
        return False, None
    return None, "Expected yes/no."


def _parse_csv_floats(raw: str) -> Tuple[List[float], Optional[str]]:
    r = raw.strip()
    if r == "" or r == "-":
        return [], None
    parts = [p.strip() for p in r.split(",") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            return [], f"Invalid number in list: {p}"
    return out, None


# -----------------------
# Navigation helpers
# -----------------------

def _top(session: Session) -> Dict[str, Any]:
    return session["runtime"]["stack"][-1]


def _push(session: Session, screen: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    session["runtime"]["stack"].append({"screen": screen, "ctx": ctx or {}})


def _pop(session: Session) -> None:
    if len(session["runtime"]["stack"]) > 1:
        session["runtime"]["stack"].pop()


# -----------------------
# Public engine API
# -----------------------

def get_prompt(session: Session) -> PromptSpec:
    session = normalize_session(session)
    frame = _top(session)
    screen = frame["screen"]
    ctx = frame["ctx"]

    # universal hint (adapters can choose to show elsewhere)
    hint = "Global commands: s=save, u=undo, a=abort, q=quit"

    if screen == "main_menu":
        return {
            "id": "main.menu",
            "kind": "choice",
            "text": "Main Menu: choose an option",
            "choices": [
                {"key": "1", "label": "Configure lock"},
                {"key": "2", "label": "Tests"},
                {"key": "3", "label": "Analyze / Plot"},
                {"key": "4", "label": "Tutorial"},
                {"key": "5", "label": "Settings"},
                {"key": "6", "label": "About"},
                {"key": "7", "label": "Learn more"},
            ],
            "help": hint,
        }

    if screen == "settings_menu":
        return {
            "id": "settings.menu",
            "kind": "choice",
            "text": "Settings",
            "choices": [
                {"key": "1", "label": "Set float display precision"},
                {"key": "2", "label": "Clear saved history"},
                {"key": "3", "label": "Return"},
            ],
            "help": hint,
        }

    if screen == "settings_precision":
        cur = session["state"]["metadata"].get("float_display_precision", FLOAT_DISPLAY_PRECISION)
        return {
            "id": "settings.precision",
            "kind": "int",
            "text": "Float display precision (0-10)",
            "default": str(cur),
            "help": hint,
        }

    if screen == "settings_cleared":
        return {
            "id": "settings.cleared",
            "kind": "confirm",
            "text": "History cleared.\n\nPress Enter to return.",
            "help": hint,
        }
    if screen == "settings_clear_confirm":
        return {
            "id": "settings.clear_confirm",
            "kind": "bool_yn",
            "text": "Clear history?\nThis will remove undo access to prior states (undo will still work for future actions).\nConfirm (y/n)",
            "help": hint,
        }

    if screen == "about":
        return {
            "id": "about.back",
            "kind": "confirm",
            "text": f"""Combination Lock Analysis Suite (CLAS)

Version:
  {CLAS_CORE_VERSION}

License:
  GNU General Public License v3.0 (GPL-3.0-only)

Author:
  knowthebird

Description:
  CLAS is an open-source utility for recording, organizing,
  and visualizing mechanical combination lock measurements.
  It is intended for educational use, locksport practice, and
  locksmith training on locks you own or have permission to open.

Disclaimer:
  This software is provided without warranty of any kind.
  Users are responsible for ensuring lawful and ethical use.

Press Enter to return.""",
            "help": hint,
        }

    if screen == "resources":
        return {
            "id": "resources.back",
            "kind": "confirm",
            "text": """Resources

Books, Papers, and Tools:

  - Sophie Houlden's Safecracking Simulator
    https://sophieh.itch.io/sophies-safecracking-simulator

  - Safecracking for Everyone (2nd Edition)
    Jared Dygert
    https://drive.google.com/file/d/1xqfTAq-NY6-hXiPB0u44vdNjeMXbHJEz/view

  - Safe Lock Manipulation 101
    Jan-Willem Markus
    https://blackbag.toool.nl/wp-content/uploads/2024/02/Safe-manipulation-101-v2.pdf

  - Safecracking 101: A Beginner's Guide to Safe Manipulation and Drilling
    Thomas A. Mazzone & Thomas G. Seroogy
    https://dn720001.ca.archive.org/0/items/safecracking-101-a-beginners-guide-to-safe-manipulation-and-drilling/safecracking%20101%20-%20A%20Beginners%20Guide%20to%20Safe%20Manipulation%20and%20Drilling%20-%20Thomas%20Mazzone%20-%202013.pdf

(These resources are provided for educational reference only.)

Press Enter to return.""",
            "help": hint,
        }

    if screen == "analyze_menu":
        return {
            "id": "analyze.menu",
            "kind": "choice",
            "text": "Analyze Menu",
            "choices": [
                {"key": "1", "label": "Plot a sweep (save .png)"},
                {"key": "2", "label": "Plot High Low Test (save .png)"},
                {"key": "3", "label": "Return"},
            ],
            "help": hint,
        }

    if screen == "plot_sweep":
        return _prompt_plot_sweep(session, ctx)

    if screen == "plot_high_low":
        return _prompt_plot_high_low(session, ctx)


    if screen == "tutorial":
        return _prompt_tutorial(session, ctx)
    if screen == "tutorial_done":
        return _prompt_tutorial_done(session, ctx)

    if screen == "tests_menu":
        return {
            "id": "tests.menu",
            "kind": "choice",
            "text": "Tests Menu: choose a test",
            "choices": [
                {"key": "1", "label": "Find AWL low point (guided)"},
                {"key": "2", "label": "Find AWR low point (guided)"},
                {"key": "3", "label": "Isolate wheel 3 (3-wheel LRL)"},
                {"key": "4", "label": "Isolate wheel 2 (3-wheel LRL)"},
                {"key": "5", "label": "High Low Test (3-wheel LRL)"},
                {"key": "6", "label": "Candidate-combination search (all wheels)"},
                {"key": "7", "label": "Single-wheel sweep"},
                {"key": "8", "label": "Return"},
            ],
            "help": hint,
        }

    if screen == "configure_lock":
        return _prompt_configure_lock(session, ctx)

    if screen == "find_awl":
        return _prompt_find_awl(session, ctx)

    if screen == "find_awr":
        return _prompt_find_awr(session, ctx)

    if screen == "isolate_wheel_3":
        return _prompt_isolate_wheel_3(session, ctx)

    if screen == "isolate_wheel_2":
        return _prompt_isolate_wheel_2(session, ctx)

    if screen == "high_low_test":
        return _prompt_high_low_test(session, ctx)

    if screen in ("candidate_combo_all", "enum_all"):
        return _prompt_candidate_combo_all(session, ctx)

    if screen in ("single_wheel_sweep", "enum_wheel"):
        return _prompt_single_wheel_sweep(session, ctx)

    # fallback
    return {
        "id": "unknown",
        "kind": "confirm",
        "text": "Unknown screen. Press Enter to return to main menu.",
        "help": hint,
    }


def apply_action(session: Session, action: Action) -> Session:
    """
    Apply a single action. The adapter is responsible for interpreting global commands
    (q/s/u/a). For u and a, pass command actions into core.

    action formats:
      {"type":"input","text":"..."}
      {"type":"command","name":"abort"|"undo"}
    """
    session = normalize_session(session)

    at = action.get("type")
    if at == "command" and action.get("name") == "undo":
        if session["cursor"] <= 0:
            return session
        session["cursor"] -= 1
        return rebuild(session)

    # For any non-undo action: if we've undone, discard "future" history
    if session["cursor"] < len(session["history"]):
        session["history"] = session["history"][: session["cursor"]]

    prompt = get_prompt(session)
    prompt_id = prompt.get("id", "unknown")

    if at == "command":
        name = str(action.get("name", "")).strip().lower()
        if name == "abort":
            event = {"type": "command", "name": "abort", "prompt_id": prompt_id}
            session["history"].append(event)
            session["cursor"] += 1
            _apply_event(session, event)
            return session

        # unknown commands are ignored
        return session

    if at == "input":
        raw_text = str(action.get("text", ""))
        event = {"type": "input", "text": raw_text, "prompt_id": prompt_id}
        # Validate by attempting to apply; if invalid, do not record
        ok, err = _apply_input_to_current_prompt(session, prompt, raw_text)
        if not ok:
            # Store last error for adapter to display (no I/O here)
            _top(session)["ctx"]["_last_error"] = err or "Invalid input."
            return session
        if session.get("runtime", {}).pop("_suppress_history_once", False):
            _top(session)["ctx"].pop("_last_error", None)
            return session
        # record and advance
        session["history"].append(event)
        session["cursor"] += 1
        _top(session)["ctx"].pop("_last_error", None)
        return session

    return session


def rebuild(session: Session) -> Session:
    """
    Deterministically rebuild state/runtime from history[:cursor].
    """
    session = normalize_session(session)
    hist = list(session.get("history", []) or [])
    if not hist:
        session["cursor"] = 0
        return session
    cursor = int(session.get("cursor", 0) or 0)
    cursor = max(0, min(cursor, len(hist)))
    try:
        last_saved_cursor = int(session.get("meta", {}).get("last_saved_cursor", -1) or -1)
    except Exception:
        last_saved_cursor = -1
    if last_saved_cursor == cursor and cursor == len(hist):
        session["cursor"] = cursor
        return session

    name = str(session.get("meta", {}).get("session_name", "session") or "session")
    fresh = new_session(name)

    # Keep meta from existing session (other metadata, if any)
    fresh["meta"] = dict(session.get("meta", {}) or fresh["meta"])
    baseline = fresh["meta"].get("baseline_lock_config")
    if baseline is not None:
        fresh["state"]["lock_config"] = normalize_lock_config(baseline)

    # Replay with prompt-id guard to avoid misapplying history after prompt changes.
    applied = 0
    for i in range(cursor):
        ev = hist[i]
        if ev.get("type") == "input":
            prompt = get_prompt(fresh)
            expected = str(prompt.get("id", ""))
            recorded = str(ev.get("prompt_id", ""))
            if recorded and expected and recorded != expected:
                fresh.setdefault("runtime", {})["_rebuild_mismatch"] = {
                    "at": i,
                    "expected": expected,
                    "recorded": recorded,
                }
                break
        _apply_event(fresh, ev)
        applied = i + 1

    # Carry over history + cursor
    fresh["history"] = hist
    fresh["cursor"] = applied
    return normalize_session(fresh)


# -----------------------
# Event application
# -----------------------

def _apply_event(session: Session, ev: Dict[str, Any]) -> None:
    """
    Apply a recorded event to session by routing it through the current prompt.
    This is used both in live application and rebuild().
    """
    # IMPORTANT: normalize in-place so callers keep the same dict object.
    norm = normalize_session(session)
    session.clear()
    session.update(norm)

    if ev.get("type") == "command" and ev.get("name") in ("abort"):
        _pop(session)
        return

    if ev.get("type") == "input":
        prompt = get_prompt(session)
        raw = str(ev.get("text", ""))
        _apply_input_to_current_prompt(session, prompt, raw)
        return

    # Ignore unknown events


def _apply_input_to_current_prompt(session: Session, prompt: PromptSpec, raw_text: str) -> Tuple[bool, Optional[str]]:
    """
    Parse input according to prompt kind, then advance the state machine for that screen.
    Returns (ok, error_message).
    """
    kind = prompt.get("kind", "text")
    raw = raw_text

    # defaulting behavior
    default_provided = "default" in prompt and prompt.get("default", None) is not None
    if raw.strip() == "" and default_provided:
        raw = str(prompt.get("default"))

    parsed: Any = raw

    if kind == "confirm":
        # any input (incl empty) is accepted
        parsed = raw
    elif kind == "choice":
        key = raw.strip()
        if key == "" and str(prompt.get("id", "")) == "tutorial.show":
            key = "3"
        choices = prompt.get("choices", []) or []
        valid = {c["key"] for c in choices if isinstance(c, dict) and "key" in c}
        if key not in valid:
            return False, "Choose one of the listed options."
        # If a choice provides a "value", return that (allows numeric selection keys for non-numeric values)
        chosen = next((c for c in choices if isinstance(c, dict) and c.get("key") == key), None)
        parsed = chosen.get("value") if isinstance(chosen, dict) and "value" in chosen else key
    elif kind == "int":
        parsed, err = _parse_int(raw.strip())
        if err:
            return False, err
    elif kind == "float":
        parsed, err = _parse_float(raw.strip())
        if err:
            return False, err
    elif kind == "bool_yn":
        parsed, err = _parse_bool_yn(raw.strip())
        if err:
            return False, err
    elif kind == "csv_floats":
        if raw.strip() == "-" and str(prompt.get("id", "")).startswith("config.gates."):
            parsed, err = None, None
        else:
            parsed, err = _parse_csv_floats(raw)
        if err:
            return False, err
    elif kind == "text":
        parsed = raw

    # Route to screen handler
    frame = _top(session)
    screen = frame["screen"]
    ctx = frame["ctx"]

    if screen == "main_menu":
        return _apply_main_menu(session, ctx, parsed)

    if screen == "about":
        _pop(session)
        return True, None

    if screen == "tutorial":
        return _apply_tutorial(session, ctx, parsed, prompt)
    if screen == "tutorial_done":
        return _apply_tutorial_done(session, ctx, parsed, prompt)

    if screen == "settings_menu":
        return _apply_settings_menu(session, ctx, parsed)

    if screen == "settings_precision":
        return _apply_settings_precision(session, ctx, parsed)

    if screen == "settings_clear_confirm":
        return _apply_settings_clear_confirm(session, ctx, parsed)

    if screen == "settings_cleared":
        session["runtime"]["stack"] = [{"screen": "main_menu", "ctx": {}}]
        return True, None

    if screen == "tests_menu":
        return _apply_tests_menu(session, ctx, parsed)

    if screen == "configure_lock":
        return _apply_configure_lock(session, ctx, parsed, prompt)

    if screen == "find_awl":
        return _apply_find_awl(session, ctx, parsed, prompt)

    if screen == "find_awr":
        return _apply_find_awr(session, ctx, parsed, prompt)

    if screen == "isolate_wheel_3":
        return _apply_isolate_wheel_3(session, ctx, parsed, prompt)

    if screen == "isolate_wheel_2":
        return _apply_isolate_wheel_2(session, ctx, parsed, prompt)

    if screen == "high_low_test":
        return _apply_high_low_test(session, ctx, parsed, prompt)

    if screen in ("candidate_combo_all", "enum_all"):
        return _apply_candidate_combo_all(session, ctx, parsed, prompt)

    if screen in ("single_wheel_sweep", "enum_wheel"):
        return _apply_single_wheel_sweep(session, ctx, parsed, prompt)

    if screen == "analyze_menu":
        return _apply_analyze_menu(session, ctx, parsed)

    if screen == "plot_sweep":
        return _apply_plot_sweep(session, ctx, parsed, prompt)

    if screen == "plot_high_low":
        return _apply_plot_high_low(session, ctx, parsed, prompt)

    # unknown: bounce to main
    session["runtime"]["stack"] = [{"screen": "main_menu", "ctx": {}}]
    return True, None


# -----------------------
# Screen: main menu
# -----------------------

def _apply_main_menu(session: Session, ctx: Dict[str, Any], choice: str) -> Tuple[bool, Optional[str]]:
    if choice == "1":
        _push(session, "configure_lock", {})
        return True, None
    if choice == "2":
        _push(session, "tests_menu", {})
        return True, None
    if choice == "3":
        _push(session, "analyze_menu", {})
        return True, None
    if choice == "4":
        _push(session, "tutorial", {})
        return True, None
    if choice == "5":
        _push(session, "settings_menu", {})
        return True, None
    if choice == "6":
        _push(session, "about", {})
        return True, None
    if choice == "7":
        _push(session, "resources", {})
        return True, None

    return False, "Invalid choice."


# -----------------------
# Screen: settings menu
# -----------------------

def _apply_settings_menu(session: Session, ctx: Dict[str, Any], choice: str) -> Tuple[bool, Optional[str]]:
    if choice == "1":
        _push(session, "settings_precision", {})
        return True, None
    if choice == "2":
        _push(session, "settings_clear_confirm", {})
        return True, None
    if choice == "3":
        _pop(session)
        return True, None
    return False, "Invalid choice."


def _apply_settings_precision(session: Session, ctx: Dict[str, Any], value: int) -> Tuple[bool, Optional[str]]:
    try:
        precision = int(value)
    except Exception:
        return False, "Expected an integer."
    if precision < 0 or precision > 10:
        return False, "Precision must be between 0 and 10."
    session["state"]["metadata"]["float_display_precision"] = precision
    global FLOAT_DISPLAY_PRECISION
    FLOAT_DISPLAY_PRECISION = precision
    _pop(session)
    return True, None


def _apply_settings_clear_confirm(session: Session, ctx: Dict[str, Any], confirm: bool) -> Tuple[bool, Optional[str]]:
    if confirm:
        session["history"] = []
        session["cursor"] = 0
        session.setdefault("runtime", {})["_suppress_history_once"] = True
        session["runtime"]["stack"] = [{"screen": "main_menu", "ctx": {}}]
        session.setdefault("meta", {})["baseline_lock_config"] = dict(session["state"]["lock_config"])
    else:
        _pop(session)
    return True, None


# -----------------------
# Screen: tests menu
# -----------------------

def _apply_tests_menu(session: Session, ctx: Dict[str, Any], choice: str) -> Tuple[bool, Optional[str]]:
    if choice == "1":
        _push(session, "find_awl", {})
        return True, None
    if choice == "2":
        _push(session, "find_awr", {})
        return True, None
    if choice == "3":
        _push(session, "isolate_wheel_3", {})
        return True, None
    if choice == "4":
        _push(session, "isolate_wheel_2", {})
        return True, None
    if choice == "5":
        _push(session, "high_low_test", {})
        return True, None
    if choice == "6":
        _push(session, "candidate_combo_all", {})
        return True, None
    if choice == "7":
        _push(session, "single_wheel_sweep", {})
        return True, None
    if choice == "8":
        _pop(session)
        return True, None
    return False, "Invalid choice."


# -----------------------
# Screen: configure lock
# -----------------------

def _prompt_configure_lock(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    step = int(ctx.get("step", 0) or 0)
    editing = bool(ctx.get("editing", False))

    # keep a working copy in ctx
    if "work" not in ctx:
        ctx["work"] = dict(lc)

    work = ctx["work"]

    # Step 0: show summary and ask edit?
    if step == 0:
        rows = [
            ("Make", lc.get("make", "UNKNOWN")),
            ("UL Rating", lc.get("ul", "UNKNOWN")),
            ("Total Wheels", lc["wheels"]),
            ("Fence", lc.get("fence_type", "UNKNOWN")),
            ("Turn Sequence", lc["turn_sequence"]),
            ("Dial Range", f"{_fmt_float(lc['dial_min'])}..{_fmt_float(lc['dial_max'])}"),
            ("Flies", lc.get("flies", "fixed")),
            ("Tolerance", f"±{_fmt_float(lc['tolerance'])}"),
            ("Oval Wheels", lc.get("oval_wheels", "UNKNOWN")),
            ("AWL Low Point", _fmt_float(lc.get("awl_low_point"))),
            ("AWR Low Point", _fmt_float(lc.get("awr_low_point"))),
            ("approx_lcp_location", _fmt_float(lc.get("approx_lcp_location"))),
            ("approx_rcp_location", _fmt_float(lc.get("approx_rcp_location"))),
        ]
        label_width = max(len(label) for label, _ in rows)
        lines = [
            f"  {label:<{label_width}}: {value}"
            for label, value in rows
        ]
        summary = (
            "Current configuration:\n"
            + "\n".join(lines)
            + "\n"
            f"  wheel_data (gates):\n{_format_wheel_data(lc)}\n\n"
            "Edit configuration?\n"
        )
        return {
            "id": "config.edit",
            "kind": "choice",
            "text": summary,
            "choices": [{"key": "1", "label": "Yes"}, {"key": "2", "label": "No (Return)"}],
        }

    if not editing:
        # shouldn't happen; guard
        ctx["step"] = 0
        return {
            "id": "config.edit",
            "kind": "choice",
            "text": "Edit configuration?\n  1) Yes\n  2) No (return)\n",
            "choices": [{"key": "1", "label": "Yes"}, {"key": "2", "label": "No (Return)"}],
        }

    # Field steps (kept in same order as summary)
    if step == 1:
        return {"id":"config.make","kind":"text","text":"Make", "default": str(work.get("make", lc.get("make","UNKNOWN")))}
    if step == 2:
        cur = str(work.get("ul", lc.get("ul","UNKNOWN"))).strip().upper()
        opts = ["2","2M","1","1R","UNKNOWN"]
        try:
            default_key = str(opts.index(cur) + 1)
        except ValueError:
            default_key = "5"
        return {
            "id":"config.ul",
            "kind":"choice",
            "text":"UL rating",
            "choices":[
                {"key":"1","label":"2","value":"2"},
                {"key":"2","label":"2M","value":"2M"},
                {"key":"3","label":"1","value":"1"},
                {"key":"4","label":"1R","value":"1R"},
                {"key":"5","label":"UNKNOWN","value":"UNKNOWN"},
            ],
            "default": default_key,
        }
    if step == 3:
        return {"id":"config.wheels","kind":"int","text":"Number of wheels", "default": int(work.get("wheels", lc["wheels"]))}
    if step == 4:
        cur = str(work.get("fence_type", lc.get("fence_type","UNKNOWN"))).strip().upper()
        opts = ["FRICTION_FENCE","GRAVITY_LEVER","SPRING_LEVER","UNKNOWN"]
        try:
            default_key = str(opts.index(cur) + 1)
        except ValueError:
            default_key = "4"
        return {
            "id":"config.fence_type",
            "kind":"choice",
            "text":"Fence / lever type",
            "choices":[
                {"key":"1","label":"FRICTION_FENCE","value":"FRICTION_FENCE"},
                {"key":"2","label":"GRAVITY_LEVER","value":"GRAVITY_LEVER"},
                {"key":"3","label":"SPRING_LEVER","value":"SPRING_LEVER"},
                {"key":"4","label":"UNKNOWN","value":"UNKNOWN"},
            ],
            "default": default_key,
        }
    if step == 5:
        cur = str(work.get("turn_sequence", lc["turn_sequence"])).strip().upper()
        opts = ["LRL", "RLR"]
        try:
            default_key = str(opts.index(cur) + 1)
        except ValueError:
            default_key = "1"
        return {
            "id":"config.turn_sequence",
            "kind":"choice",
            "text":"Turn sequence",
            "choices":[
                {"key":"1","label":"LRL","value":"LRL"},
                {"key":"2","label":"RLR","value":"RLR"},
            ],
            "default": default_key,
        }
    if step == 6:
        return {"id":"config.dial_min","kind":"float","text":"Dial minimum value", "default": _fmt_float(work.get("dial_min", lc["dial_min"]))}
    if step == 7:
        return {"id":"config.dial_max","kind":"float","text":"Dial maximum value", "default": _fmt_float(work.get("dial_max", lc["dial_max"]))}
    if step == 8:
        cur = str(work.get("flies", lc.get("flies", "fixed"))).strip().lower()
        # normalize legacy boolean flies -> fixed/moveable
        if cur in ("true", "yes", "y", "1"):
            cur = "moveable"
        elif cur in ("false", "no", "n", "0"):
            cur = "fixed"

        default_key = "2" if cur in ("moveable", "movable") else "1"
        return {
            "id":"config.flies",
            "kind":"choice",
            "text":"Wheel flies",
            "choices":[
                {"key":"1","label":"FIXED","value":"fixed"},
                {"key":"2","label":"MOVEABLE","value":"moveable"},
            ],
            "default": default_key,
        }
    if step == 9:
        return {"id":"config.tolerance","kind":"float","text":"Lock tolerance (±)", "default": _fmt_float(work.get("tolerance", lc["tolerance"]))}
    if step == 10:
        cur = str(work.get("oval_wheels", lc.get("oval_wheels","UNKNOWN"))).strip().upper()
        opts = ["UNKNOWN","YES","NO"]
        try:
            default_key = str(opts.index(cur) + 1)
        except ValueError:
            default_key = "1"
        return {
            "id":"config.oval_wheels",
            "kind":"choice",
            "text":"Oval wheels",
            "choices":[
                {"key":"1","label":"UNKNOWN","value":"UNKNOWN"},
                {"key":"2","label":"YES","value":"YES"},
                {"key":"3","label":"NO","value":"NO"},
            ],
            "default": default_key,
        }
    if step == 11:
        cur = work.get("awl_low_point", lc.get("awl_low_point"))
        default = "" if cur is None else _fmt_float(cur)
        return {"id":"config.awl_low_point","kind":"text","text":"AWL low point (number; '-' to clear; Enter to keep)", "default": default}
    if step == 12:
        cur = work.get("awr_low_point", lc.get("awr_low_point"))
        default = "" if cur is None else _fmt_float(cur)
        return {"id":"config.awr_low_point","kind":"text","text":"AWR low point (number; '-' to clear; Enter to keep)", "default": default}
    if step == 13:
        cur = work.get("approx_lcp_location", lc.get("approx_lcp_location"))
        default = "" if cur is None else _fmt_float(cur)
        return {"id":"config.approx_lcp_location","kind":"text","text":"Approx LCP location (number; '-' to clear; Enter to keep)", "default": default}
    if step == 14:
        cur = work.get("approx_rcp_location", lc.get("approx_rcp_location"))
        default = "" if cur is None else _fmt_float(cur)
        return {"id":"config.approx_rcp_location","kind":"text","text":"Approx RCP location (number; '-' to clear; Enter to keep)", "default": default}


    # Gate editing loop: wheel index in ctx["gate_w"]
    if step == 17:
        w = int(ctx.get("gate_w", 1) or 1)
        wd = normalize_lock_config(work).get("wheel_data", {}).get(str(w), {})
        return {"id":"config.gates.known","kind":"csv_floats","text":f"Wheel {w} KNOWN gates (comma-separated; Enter=keep; '-'=clear)", "default": ",".join(_fmt_float(x) for x in wd.get("known_gates", []))}
    if step == 18:
        w = int(ctx.get("gate_w", 1) or 1)
        wd = normalize_lock_config(work).get("wheel_data", {}).get(str(w), {})
        return {"id":"config.gates.suspected","kind":"csv_floats","text":f"Wheel {w} SUSPECTED gates (comma-separated; Enter=keep; '-'=clear)", "default": ",".join(_fmt_float(x) for x in wd.get("suspected_gates", []))}
    if step == 19:
        w = int(ctx.get("gate_w", 1) or 1)
        wd = normalize_lock_config(work).get("wheel_data", {}).get(str(w), {})
        return {"id":"config.gates.false","kind":"csv_floats","text":f"Wheel {w} FALSE gates (comma-separated; Enter=keep; '-'=clear)", "default": ",".join(_fmt_float(x) for x in wd.get("false_gates", []))}

    # Safety fallback (should not happen)
    return {"id":"config.invalid","kind":"message","text":"[Error] Invalid configuration step."}


def _apply_configure_lock(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    step = int(ctx.get("step", 0) or 0)
    lc = session["state"]["lock_config"]

    if "work" not in ctx:
        ctx["work"] = dict(lc)
    work = ctx["work"]

    def _apply_optional_float(key: str) -> Tuple[bool, Optional[str]]:
        raw = str(parsed).strip()
        if raw == "":
            # keep as-is
            return True, None
        if raw in ("-", "none", "null"):
            work[key] = None
            return True, None
        try:
            work[key] = float(raw)
            return True, None
        except Exception:
            return False, "Expected a number (or '-' to clear)."

    if step == 0:
        if parsed == "2":
            _pop(session)
            return True, None
        ctx["editing"] = True
        ctx["step"] = 1
        return True, None

    # field steps
    if step == 1:
        work["make"] = str(parsed).strip() or "UNKNOWN"
        ctx["step"] = 2
        return True, None
    if step == 2:
        work["ul"] = str(parsed).strip().upper() or "UNKNOWN"
        ctx["step"] = 3
        return True, None
    if step == 3:
        work["wheels"] = int(parsed)
        # ensure wheel buckets exist immediately
        ctx["work"] = normalize_lock_config(work)
        ctx["step"] = 4
        return True, None
    if step == 4:
        work["fence_type"] = str(parsed).strip().upper() or "UNKNOWN"
        ctx["step"] = 5
        return True, None
    if step == 5:
        work["turn_sequence"] = str(parsed).upper()
        ctx["step"] = 6
        return True, None
    if step == 6:
        work["dial_min"] = float(parsed)
        ctx["step"] = 7
        return True, None
    if step == 7:
        work["dial_max"] = float(parsed)
        # validate dial range
        if float(work["dial_max"]) <= float(work["dial_min"]):
            return False, "Dial max must be greater than dial min."
        ctx["step"] = 8
        return True, None
    if step == 8:
        key = str(parsed).strip().upper()
        work["flies"] = "moveable" if key == "MOVEABLE" else "fixed"
        ctx["step"] = 9
        return True, None
    if step == 9:
        work["tolerance"] = float(parsed)
        if work["tolerance"] <= 0:
            work["tolerance"] = 1.0
        ctx["step"] = 10
        return True, None
    if step == 10:
        work["oval_wheels"] = str(parsed).strip().upper() or "UNKNOWN"
        ctx["step"] = 11
        return True, None
    if step == 11:
        ok, err = _apply_optional_float("awl_low_point")
        if not ok:
            return False, err
        ctx["step"] = 12
        return True, None
    if step == 12:
        ok, err = _apply_optional_float("awr_low_point")
        if not ok:
            return False, err
        ctx["step"] = 13
        return True, None
    if step == 13:
        ok, err = _apply_optional_float("approx_lcp_location")
        if not ok:
            return False, err
        ctx["step"] = 14
        return True, None
    if step == 14:
        ok, err = _apply_optional_float("approx_rcp_location")
        if not ok:
            return False, err
        ctx["gate_w"] = 1
        ctx["gate_kind"] = "known_gates"
        ctx["step"] = 17
        return True, None

        ctx["step"] = 16
        ctx["gate_w"] = 1
        return True, None


    if step == 17:
        w = int(ctx.get("gate_w", 1) or 1)
        wd = normalize_lock_config(work).get("wheel_data", {})
        cur = wd.get(str(w), {})
        if parsed is None:
            cur["known_gates"] = []
        elif parsed != []:
            cur["known_gates"] = parsed
        wd[str(w)] = cur
        work["wheel_data"] = wd
        ctx["step"] = 18
        return True, None

    if step == 18:
        w = int(ctx.get("gate_w", 1) or 1)
        wd = normalize_lock_config(work).get("wheel_data", {})
        cur = wd.get(str(w), {})
        if parsed is None:
            cur["suspected_gates"] = []
        elif parsed != []:
            cur["suspected_gates"] = parsed
        wd[str(w)] = cur
        work["wheel_data"] = wd
        ctx["step"] = 19
        return True, None

    if step == 19:
            # set false_gates
            w = int(ctx.get("gate_w", 1) or 1)
            wd = work.setdefault("wheel_data", {})
            wdata = wd.setdefault(str(w), {})
            if parsed is None:
                parsed_list = []
            else:
                parsed_list = parsed if isinstance(parsed, list) else ([] if parsed in ("",) else [parsed])
            wdata["false_gates"] = parsed_list
            wd[str(w)] = wdata
            work["wheel_data"] = wd

            # advance to next wheel or finish immediately (no intermediate prompt/step)
            try:
                wheels = int(work.get("wheels", lc.get("wheels", 0)) or 0)
            except Exception:
                wheels = int(lc.get("wheels", 0) or 0)

            next_w = w + 1
            if next_w > wheels:
                session["state"]["lock_config"] = normalize_lock_config(work)
                session.setdefault("runtime", {})["notify"] = "Configuration updated."
                _pop(session)
                return True, None

            ctx["gate_w"] = next_w
            ctx["gate_kind"] = "known_gates"
            ctx["step"] = 17
            return True, None


            ctx["gate_w"] = w
            ctx["gate_kind"] = "known_gates"
            ctx["step"] = 17
            return True, None
            ctx["gate_w"] = w
            ctx["gate_kind"] = "known_gates"
            ctx["step"] = 17
            return True, None



# -----------------------
# Screen: Find AWL
# -----------------------

def _prompt_find_awl(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    if int(lc.get("wheels", 0) or 0) != 3 or str(lc.get("turn_sequence","")).upper() != "LRL":
        return {"id":"awl.unsupported","kind":"confirm","text":"Find AWL is implemented for 3-wheel LRL locks. Press Enter to return."}

    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
    w3_gate, w3_gate_src = _w3_gate_info(lc)
    ctx["w3_gate"] = w3_gate
    ctx["w3_gate_src"] = w3_gate_src

    if ctx.get("n_points") is None:
        ctx["step"] = "ask_points"
        return {"id":"awl.points","kind":"int","text":"How many checkpoints to test? (1..36000)","default":10}
    checkpoints = ctx.get("checkpoints")
    if not isinstance(checkpoints, list):
        checkpoints = build_checkpoints(dial_min, dial_max, n_points=int(ctx.get('n_points',10) or 10))
        ctx["checkpoints"] = checkpoints
        ctx["idx"] = 0
        ctx["readings"] = []
        ctx["step"] = "intro"

    step = ctx.get("step","intro")

    if step == "manual":
        return {"id":"awl.manual","kind":"float","text":"Enter the value to save:"}

    if step == "intro":
        cps = "\n".join([f"  {i+1:>2}) {_fmt_float(c)}" for i,c in enumerate(checkpoints)])
        return {"id":"awl.intro","kind":"confirm","text":f"Find AWL Low Point\nCheckpoints:\n{cps}\n\nPress Enter to start."}

    idx = int(ctx.get("idx",0) or 0)
    if idx < len(checkpoints):
        cp = checkpoints[idx]
        if step == "to_checkpoint":
            if idx == 0:
                text = (
                    f"Checkpoint {idx+1}/{len(checkpoints)}:\n"
                    "Turn dial left (CCW) until you pass 0 two times, and stop on 0 when you reach it the third time.\n"
                    f"Continue turning left (CCW) to {_fmt_float(cp)}.\n"
                    "Press Enter when you are on the checkpoint."
                )
            else:
                text = (
                    f"Checkpoint {idx+1}/{len(checkpoints)}:\n"
                    f"Continue turning left (CCW) to {_fmt_float(cp)}.\n"
                    "Press Enter when you are on the checkpoint."
                )
            return {"id":"awl.to_cp","kind":"confirm","text": text}
        if step == "gate":
            if w3_gate is not None:
                return {"id":"awl.gate","kind":"confirm","text":f"Turn right (CW), passing {_fmt_float(cp)} one time, and stop on the value for gate 3 ({w3_gate_src}: {_fmt_float(w3_gate)}).\nPress Enter when you are on the gate."}
            ctx["step"] = "rcp"
            step = "rcp"
        if step == "rcp":
            return {"id":"awl.rcp","kind":"float","text":"Turn left (CCW) until you hit the RCP. Enter RCP"}
        if step == "lcp":
            return {"id":"awl.lcp","kind":"float","text":"Turn right (CW) until you hit the LCP. Enter LCP"}
        if step == "next":
            next_cp = checkpoints[idx + 1]
            if w3_gate is not None:
                text = f"Turn left (CCW), passing {_fmt_float(w3_gate)} one time, to {_fmt_float(next_cp)}.\nPress Enter when you are on the next checkpoint."
            else:
                text = f"Turn left (CCW) to {_fmt_float(next_cp)}.\nPress Enter when you are on the next checkpoint."
            return {"id":"awl.next","kind":"confirm","text": text}
        return {"id":"awl.rcp","kind":"float","text":"Turn left (CCW) until you hit the RCP. Enter RCP"}
    # done -> confirm save
    readings = ctx.get("readings", [])
    best = min(readings, key=lambda d: d.get("contact_width", float("inf"))) if readings else None
    if best is None:
        return {"id":"awl.done","kind":"confirm","text":"No readings collected. Press Enter to return."}
    lines = "\n".join([
        f"  {_fmt_float(r['checkpoint'])}: LCP={_fmt_float(r['lcp'])}, RCP={_fmt_float(r.get('rcp'))}, CW={_fmt_float(r.get('contact_width'))}"
        for r in readings
    ])
    return {
        "id":"awl.save",
        "kind":"choice",
        "text": f"RESULTS\n{lines}\n\nSmallest contact width at checkpoint {_fmt_float(best['checkpoint'])} (CW={_fmt_float(best.get('contact_width'))}).\nSave as AWL low point?\n",
        "choices":[{"key":"1","label":"Yes"},{"key":"2","label":"No"},{"key":"3","label":"Enter manually"}],
    }


def _apply_find_awl(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:

    if prompt.get("id") == "awl.manual":
        try:
            v = float(parsed)
        except Exception:
            return False, "Enter a numeric value."
        lc = dict(session["state"]["lock_config"])
        lc["awl_low_point"] = float(v)
        session["state"]["lock_config"] = normalize_lock_config(lc)
        session["dirty"] = True
        _pop(session)
        return True, None

    if prompt.get("id") == "awl.unsupported":
        _pop(session)
        return True, None


    if prompt.get("id") == "awl.points":
        try:
            n = int(parsed)
        except Exception:
            n = 10
        if n < 1 or n > 36000:
            return False, "Enter an integer from 1 to 36000."
        ctx["n_points"] = n
        ctx.pop("checkpoints", None)
        ctx["step"] = "intro"
        return True, None
    step = ctx.get("step","intro")
    if step == "intro":
        ctx["step"] = "to_checkpoint"
        return True, None

    if prompt.get("id") == "awl.to_cp":
        w3_gate, _ = _w3_gate_info(session["state"]["lock_config"])
        ctx["step"] = "gate" if w3_gate is not None else "rcp"
        return True, None

    if prompt.get("id") == "awl.gate":
        ctx["step"] = "rcp"
        return True, None

    if prompt.get("id") == "awl.rcp":
        try:
            ctx["rcp"] = float(parsed)
        except Exception:
            return False, "Enter a numeric value."
        ctx["step"] = "lcp"
        return True, None

    if prompt.get("id") == "awl.lcp":
        try:
            lcp_val = float(parsed)
        except Exception:
            return False, "Enter a numeric value."
        lc = session["state"]["lock_config"]
        dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
        checkpoints = ctx.get("checkpoints", build_checkpoints(dial_min, dial_max))
        idx = int(ctx.get("idx",0) or 0)
        cp = checkpoints[idx]
        readings = ctx.get("readings", [])
        rcp_val = float(ctx.get("rcp")) if ctx.get("rcp") is not None else None
        cw = circular_distance(rcp_val, lcp_val, dial_min, dial_max) if rcp_val is not None else None
        readings.append({"checkpoint": float(cp), "lcp": lcp_val, "rcp": rcp_val, "contact_width": cw})
        ctx["readings"] = readings
        ctx["rcp"] = None
        if idx + 1 < len(checkpoints):
            ctx["step"] = "next"
        else:
            ctx["idx"] = idx + 1
        return True, None

    if prompt.get("id") == "awl.next":
        ctx["idx"] = int(ctx.get("idx",0) or 0) + 1
        w3_gate, _ = _w3_gate_info(session["state"]["lock_config"])
        ctx["step"] = "gate" if w3_gate is not None else "rcp"
        return True, None

    if prompt.get("id") == "awl.save":

        if parsed == "3":
            ctx["step"] = "manual"
            return True, None

        if parsed == "1":
            readings = ctx.get("readings", [])
            best = min(readings, key=lambda d: d.get("contact_width", float("inf"))) if readings else None
            if best:
                lc = dict(session["state"]["lock_config"])
                lc["awl_low_point"] = float(best["checkpoint"])
                session["state"]["lock_config"] = normalize_lock_config(lc)
        _pop(session)
        return True, None

    _pop(session)
    return True, None


# -----------------------
# Screen: Find AWR
# -----------------------

def _prompt_find_awr(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    if int(lc.get("wheels", 0) or 0) != 3 or str(lc.get("turn_sequence","")).upper() != "LRL":
        return {"id":"awr.unsupported","kind":"confirm","text":"Find AWR is implemented for 3-wheel LRL locks. Press Enter to return."}

    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])

    if ctx.get("n_points") is None:
        ctx["step"] = "ask_points"
        return {"id":"awr.points","kind":"int","text":"How many checkpoints to test? (1..36000)","default":10}
    checkpoints = ctx.get("checkpoints")
    if not isinstance(checkpoints, list):
        checkpoints = list(reversed(build_checkpoints(dial_min, dial_max, n_points=int(ctx.get('n_points',10) or 10))))
        ctx["checkpoints"] = checkpoints
        ctx["idx"] = 0
        ctx["readings"] = []
        ctx["step"] = "intro"

    step = ctx.get("step","intro")

    if step == "manual":
        return {"id":"awr.manual","kind":"float","text":"Enter the value to save:"}

    if step == "intro":
        cps = "\n".join([f"  {i+1:>2}) {_fmt_float(c)}" for i,c in enumerate(checkpoints)])
        return {"id":"awr.intro","kind":"confirm","text":f"Find AWR Low Point\nCheckpoints:\n{cps}\n\nTurn dial right (CW) until you pass 0 three times, and stop on 0 when you reach it the fourth time.\nPress Enter to start."}

    idx = int(ctx.get("idx",0) or 0)
    if idx < len(checkpoints):
        cp = checkpoints[idx]
        return {"id":"awr.rcp","kind":"float","text":f"Checkpoint {idx+1}/{len(checkpoints)}:\nTurn dial right (CW) to {_fmt_float(cp)}\nThen turn dial left (CCW) until you reach the RCP. Enter RCP"}

    readings = ctx.get("readings", [])
    best = min(readings, key=lambda d: d["rcp"]) if readings else None
    if best is None:
        return {"id":"awr.done","kind":"confirm","text":"No readings collected. Press Enter to return."}
    lines = "\n".join([f"  {_fmt_float(r['checkpoint'])}: RCP={_fmt_float(r['rcp'])}" for r in readings])
    return {
        "id":"awr.save",
        "kind":"choice",
        "text": f"RESULTS\n{lines}\n\nSmallest RCP at checkpoint {_fmt_float(best['checkpoint'])} (RCP={_fmt_float(best['rcp'])}).\nSave as AWR low point?\n",
        "choices":[{"key":"1","label":"Yes"},{"key":"2","label":"No"},{"key":"3","label":"Enter manually"}],
    }


def _apply_find_awr(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:

    if prompt.get("id") == "awr.manual":
        try:
            v = float(parsed)
        except Exception:
            return False, "Enter a numeric value."
        lc = dict(session["state"]["lock_config"])
        lc["awr_low_point"] = float(v)
        session["state"]["lock_config"] = normalize_lock_config(lc)
        session["dirty"] = True
        _pop(session)
        return True, None

    if prompt.get("id") == "awr.unsupported":
        _pop(session)
        return True, None

    if prompt.get("id") == "awr.points":
        try:
            n = int(parsed)
        except Exception:
            n = 10
        if n < 1 or n > 36000:
            return False, "Enter an integer from 1 to 36000."
        ctx["n_points"] = n
        ctx.pop("checkpoints", None)
        ctx["step"] = "intro"
        return True, None

    step = ctx.get("step","intro")
    if step == "intro":
        ctx["step"] = "scan"
        return True, None

    if prompt.get("id") == "awr.rcp":
        lc = session["state"]["lock_config"]
        dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
        checkpoints = ctx.get("checkpoints", build_checkpoints(dial_min, dial_max))
        idx = int(ctx.get("idx",0) or 0)
        cp = checkpoints[idx]
        readings = ctx.get("readings", [])
        readings.append({"checkpoint": float(cp), "rcp": float(parsed)})
        ctx["readings"] = readings
        ctx["idx"] = idx + 1
        return True, None

    if prompt.get("id") == "awr.save":

        if parsed == "3":
            ctx["step"] = "manual"
            return True, None

        if parsed == "1":
            readings = ctx.get("readings", [])
            best = min(readings, key=lambda d: d["rcp"]) if readings else None
            if best:
                lc = dict(session["state"]["lock_config"])
                lc["awr_low_point"] = float(best["checkpoint"])
                session["state"]["lock_config"] = normalize_lock_config(lc)
        _pop(session)
        return True, None

    _pop(session)
    return True, None


# -----------------------
# Screen: Isolate wheel 3 (3-wheel LRL)
# -----------------------

def _prompt_isolate_wheel_3(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    if int(lc.get("wheels", 0) or 0) != 3 or str(lc.get("turn_sequence","")).upper() != "LRL":
        return {"id":"iso3.unsupported","kind":"confirm","text":"Isolate wheel 3 is implemented for 3-wheel LRL locks. Press Enter to return."}

    if ctx.get("phase") == "manual_awr":
        return {"id":"iso3.set_awr","kind":"float","text":"Enter AWR low point"}

    if lc.get("awr_low_point") is None:
        return {
            "id":"iso3.need_awr",
            "kind":"choice",
            "text":"AWR low point is not set.",
            "choices": [
                {"key":"1","label":"Find AWR low point (guided)"},
                {"key":"2","label":"Manually Enter Starting Low Point"},
                {"key":"3","label":"Return"},
            ],
        }

    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
    tol = float(lc["tolerance"]); scan_step = max(1e-9, tol * 2.0)
    awr = float(lc["awr_low_point"])

    # init context
    if "phase" not in ctx:
        # build scan points
        span = (dial_max - dial_min) + 1.0
        n_steps = int(span / max(scan_step, 1e-9))
        n_steps = max(1, n_steps)
        scan_points: List[float] = []
        cur = awr + scan_step
        seen = set()
        for _ in range(n_steps + 2):
            key = round(cur, 6)
            if key in seen and len(scan_points) > 0:
                break
            seen.add(key)
            scan_points.append(wrap_dial(cur, dial_min, dial_max))
            cur = wrap_dial(cur + scan_step, dial_min, dial_max)

        ctx.update({
            "phase": "intro",
            "scan_step": scan_step,
            "awr": awr,
            "passed_awr_low": False,
            "scan_points": scan_points,
            "i": 0,
            "scan_rows": [],
            "wheel_swept": 3,
            "sweep_id": _next_sweep_id(session),
            "iso_test_id": _next_iso_test_id(session, "isolate_wheel_3"),
        })

    phase = ctx["phase"]

    if phase == "intro":
        return {"id":"iso3.intro","kind":"confirm",
                "text":(
                    f"Isolate Wheel 3 (scan step={_fmt_float(scan_step)})\n"
                    f"Turn dial right (CW) until you pass 0 three times, and stop on 0 when you reach it the fourth time.\n"
                    f"Continue right (CW) to low point ({_fmt_float(awr)}).\n\n"
                    f"Press Enter to begin scan ({len(ctx['scan_points'])} points)."
                )}

    if phase == "scan_point":
        i = int(ctx.get("i",0) or 0)
        pts = ctx["scan_points"]
        if i >= len(pts):
            ctx["phase"] = "plot_offer"
            return _prompt_isolate_wheel_3(session, ctx)
        p = float(pts[i])
        passed_awr_low = bool(ctx.get("passed_awr_low", False))
        if i == 0 or passed_awr_low:
            lead = (
                f"Turn left (CCW), passing ({_fmt_float(awr)}) one time, stopping on {_fmt_float(p)}. "
                f"Turn right (CW) until you hit the LCP. Enter LCP."
            )
        else:
            lead = (
                f"Turn left (CCW), stopping on {_fmt_float(p)} the first time you reach it. "
                f"Turn right (CW) until you hit the LCP. Enter LCP."
            )
        return {"id":"iso3.scan.lcp","kind":"float",
                "text":(
                    f"Scan {i+1}/{len(pts)} @ Wheel 3 = {_fmt_float(p)}\n"
                    f"{lead}"
                )}

    if phase == "scan_rcp":
        i = int(ctx.get("i",0) or 0)
        p = float(ctx.get("current_p"))
        dir_label = "right (CW)" if ctx.get("rcp_dir") == "right" else "left (CCW)"
        return {"id":"iso3.scan.rcp","kind":"float",
                "text":f"Scan {i+1}/{len(ctx['scan_points'])} @ Wheel 3 = {_fmt_float(p)}\nTurn {dir_label} until you reach the RCP. Enter RCP"}

    if phase == "scan_between":
        p = float(ctx.get("current_p"))
        return {
            "id":"iso3.scan.between",
            "kind":"choice",
            "text": f"Is {_fmt_float(p)} after the LCP and before RCP?",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "plot_offer":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        return {
            "id":"iso3.plot_offer",
            "kind":"choice",
            "text": f"Plot sweep {sweep_id} now? (You can also plot later from the main menu.)",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "plot_now":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        ctx["rows"] = ctx.get("scan_rows", [])
        ctx["wheel_swept"] = 3
        return {
            "id":"iso3.plot",
            "kind":"confirm",
            "text": f"Generating plot for sweep {sweep_id} (saved next to the session file)...",
        }

    if phase == "refine_confirm":
        return {
            "id":"iso3.refine.confirm",
            "kind":"choice",
            "text":"Refine these points?",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No, finish test"},
            ],
        }

    if phase == "refine_range_start":
        candidates = ctx.get("candidates", []) or []
        default_start = _fmt_float(min(candidates)) if candidates else _fmt_float(dial_min)
        return {
            "id":"iso3.refine.range_start",
            "kind":"float",
            "text":"Refinement sweep range start",
            "default": default_start,
        }

    if phase == "refine_range_end":
        candidates = ctx.get("candidates", []) or []
        default_end = _fmt_float(max(candidates)) if candidates else _fmt_float(dial_max)
        return {
            "id":"iso3.refine.range_end",
            "kind":"float",
            "text":"Refinement sweep range end",
            "default": default_end,
        }

    if phase == "refine_range_points":
        return {
            "id":"iso3.refine.range_points",
            "kind":"int",
            "text":"How many refinement points?",
            "default": "10",
        }

    if phase == "candidates":
        cand_default = ",".join(_fmt_float(x) for x in (ctx.get("candidates") or []))
        return {"id":"iso3.candidates","kind":"csv_floats",
                "text":"Enter candidate gate positions for wheel 3 (comma-separated). Empty = No Possible Gates Observed",
                "default": cand_default}

    if phase == "refine_intro":
        refine_points = ctx.get("refine_points", [])
        rstart = ctx.get("refine_range_start")
        rend = ctx.get("refine_range_end")
        return {"id":"iso3.refine.intro","kind":"confirm",
                "text":(
                    f"Refinement: {len(refine_points)} points from {_fmt_float(rstart)} to {_fmt_float(rend)}.\n"
                    f"Turn dial right (CW) until you pass 0 three times, and stop on 0 when you reach it the fourth time.\n"
                    f"Continue right (CW) to low point ({_fmt_float(awr)}).\n\n"
                    f"Press Enter to begin refinement."
                )}

    if phase == "refine_point":
        j = int(ctx.get("j",0) or 0)
        rps = ctx.get("refine_points", [])
        if j >= len(rps):
            ctx["phase"] = "finish"
            return _prompt_isolate_wheel_3(session, ctx)
        p = float(rps[j])
        passed_awr_low = bool(ctx.get("passed_awr_low", False))
        if j == 0 or passed_awr_low:
            lead = (
                f"Turn left (CCW), passing ({_fmt_float(awr)}) one time, stopping on {_fmt_float(p)}. "
                f"Turn right (CW) until you hit the LCP. Enter LCP."
            )
        else:
            lead = (
                f"Turn left (CCW), stopping on {_fmt_float(p)} the first time you reach it. "
                f"Turn right (CW) until you hit the LCP. Enter LCP."
            )
        return {"id":"iso3.refine.lcp","kind":"float",
                "text":f"Refine {j+1}/{len(rps)} @ Wheel 3 = {_fmt_float(p)}\n{lead}"}

    if phase == "refine_between":
        p = float(ctx.get("current_p"))
        return {
            "id":"iso3.refine.between",
            "kind":"choice",
            "text": f"Is {_fmt_float(p)} after the LCP and before RCP?",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "refine_rcp":
        j = int(ctx.get("j",0) or 0)
        p = float(ctx.get("current_p"))
        dir_label = "right (CW)" if ctx.get("rcp_dir") == "right" else "left (CCW)"
        return {"id":"iso3.refine.rcp","kind":"float",
                "text":f"Refine {j+1}/{len(ctx.get('refine_points',[]))} @ Wheel 3 = {_fmt_float(p)}\nTurn {dir_label} until you reach the RCP. Enter RCP"}

    if phase == "finish":
        sweep_id = ctx["sweep_id"]
        return {"id":"iso3.finish","kind":"confirm",
                "text":(
                    f"Isolate Wheel 3 complete. Sweep saved as sweep={sweep_id}.\n"
                    "You can plot this sweep later from the main menu.\n"
                    "High Low Tests can help confirm whether suspected gates are on Wheel 3.\n"
                    "These tests can be long, consider saving your progress by typing s.\n\n"
                    f"Press Enter to return."
                )}

    if phase == "post_refine_candidates":
        cand_default = ",".join(_fmt_float(x) for x in (ctx.get("candidates") or []))
        return {"id":"iso3.post_refine.candidates","kind":"csv_floats",
                "text":"Update candidate gate positions for wheel 3 (comma-separated). Empty = No Possible Gates Observed",
                "default": cand_default}

    if phase == "post_refine_plot_offer":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        return {
            "id":"iso3.post_refine.plot_offer",
            "kind":"choice",
            "text": f"Plot sweep {sweep_id} now? (You can also plot later from the main menu.)",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "post_refine_plot":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        ctx["rows"] = [
            m for m in session["state"]["measurements"]
            if str(m.get("sweep", "")).replace(".", "", 1).isdigit()
            and int(float(m.get("sweep"))) == sweep_id
        ]
        ctx["wheel_swept"] = 3
        return {
            "id":"iso3.post_refine.plot",
            "kind":"confirm",
            "text": f"Generating plot for sweep {sweep_id} (saved next to the session file)...",
        }

    # default
    return {"id":"iso3.unknown","kind":"confirm","text":"Press Enter to return."}


def _apply_isolate_wheel_3(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = prompt.get("id","")
    if pid in ("iso3.unsupported","iso3.unknown"):
        _pop(session); return True, None
    if pid == "iso3.need_awr":
        choice = str(parsed)
        if choice == "1":
            ctx.clear()
            _push(session, "find_awr", {})
            return True, None
        if choice == "2":
            ctx["phase"] = "manual_awr"
            return True, None
        _pop(session)
        return True, None
    if pid == "iso3.set_awr":
        lc = session["state"]["lock_config"]
        lc["awr_low_point"] = float(parsed)
        ctx.clear()
        return True, None

    lc = session["state"]["lock_config"]
    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
    awr = float(lc.get("awr_low_point") or 0.0)
    sweep_id = int(ctx.get("sweep_id", _next_sweep_id(session)))
    iso_test_id = int(ctx.get("iso_test_id", _next_iso_test_id(session, "isolate_wheel_3")))

    if pid == "iso3.intro":
        ctx["phase"] = "scan_point"
        return True, None

    if pid == "iso3.scan.lcp":
        # store lcp and move to rcp prompt
        pts = ctx["scan_points"]
        i = int(ctx.get("i",0) or 0)
        p = float(pts[i])
        lcp = float(parsed)
        ctx["current_p"] = p
        ctx["current_lcp"] = lcp
        if _is_between_cw(p, awr, lcp, dial_min, dial_max):
            ctx["passed_awr_low"] = True
        elif _is_between_cw(awr, p, lcp, dial_min, dial_max):
            ctx["passed_awr_low"] = False
        ctx["phase"] = "scan_between"
        return True, None

    if pid == "iso3.scan.between":
        ctx["rcp_dir"] = "right" if str(parsed) == "1" else "left"
        ctx["phase"] = "scan_rcp"
        return True, None

    if pid == "iso3.scan.rcp":
        # record row to ctx scan_rows (not saved to measurements until finish)
        p = float(ctx.get("current_p"))
        lcp = float(ctx.get("current_lcp"))
        rcp = float(parsed)
        if ctx.get("rcp_dir") == "right":
            if _is_between_cw(p, awr, rcp, dial_min, dial_max):
                ctx["passed_awr_low"] = True
            elif _is_between_cw(awr, p, rcp, dial_min, dial_max):
                ctx["passed_awr_low"] = False
        row = {
            "id": _next_measurement_id(session, ctx),
            "sweep": sweep_id,
            "wheel_swept": 3,
            "combination_wheel_1": awr,
            "combination_wheel_2": awr,
            "combination_wheel_3": p,
            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": circular_distance(rcp, lcp, dial_min, dial_max),
            "iso_test": "isolate_wheel_3",
            "iso_test_id": iso_test_id,
            "iso_phase": "scan",
            "notes": "",
        }
        ctx["scan_rows"].append(row)
        ctx["i"] = int(ctx.get("i",0) or 0) + 1
        ctx["phase"] = "scan_point"
        return True, None

    if pid == "iso3.plot_offer":
        if str(parsed) == "1":
            ctx["rows"] = ctx.get("scan_rows", [])
            ctx["phase"] = "plot_now"
            return True, None
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "candidates"
        return True, None

    if pid == "iso3.plot":
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "candidates"
        return True, None

    if pid == "iso3.candidates":
        candidates: List[float] = list(parsed) if isinstance(parsed, list) else []
        # save scan rows now
        session["state"]["measurements"].extend(ctx.get("scan_rows", []))
        # if no candidates, finish
        if not candidates:
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["candidates"] = candidates
        lc = session["state"]["lock_config"]
        wd = lc.get("wheel_data", {}) or {}
        w3 = wd.get("3", {}) or {}
        w3["suspected_gates"] = [float(x) for x in candidates]
        wd["3"] = w3
        lc["wheel_data"] = wd
        session["state"]["lock_config"] = normalize_lock_config(lc)
        ctx["phase"] = "refine_confirm"
        return True, None

    if pid == "iso3.refine.confirm":
        if str(parsed) == "2":
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["phase"] = "refine_range_start"
        return True, None

    if pid == "iso3.refine.range_start":
        ctx["refine_range_start"] = float(parsed)
        ctx["phase"] = "refine_range_end"
        return True, None

    if pid == "iso3.refine.range_end":
        ctx["refine_range_end"] = float(parsed)
        ctx["phase"] = "refine_range_points"
        return True, None

    if pid == "iso3.refine.range_points":
        try:
            n_points = int(parsed)
        except Exception:
            n_points = 0
        if n_points < 1 or n_points > 36000:
            return False, "Enter an integer from 1 to 36000."
        rstart = float(ctx.get("refine_range_start", dial_min))
        rend = float(ctx.get("refine_range_end", dial_max))
        refine_points = _build_range_points(rstart, rend, n_points, dial_min, dial_max)
        # remove already measured points in this sweep
        measured = set(round(float(m.get("combination_wheel_3", -9999)), 6)
                       for m in session["state"]["measurements"]
                       if m.get("sweep") == sweep_id and m.get("wheel_swept") == 3)
        refine_list = [p for p in refine_points if round(float(p), 6) not in measured]
        ctx["refine_points"] = refine_list
        ctx["j"] = 0
        ctx["phase"] = "refine_intro" if refine_list else "finish"
        return True, None

    if pid == "iso3.refine.intro":
        ctx["phase"] = "refine_point"
        return True, None

    if pid == "iso3.refine.lcp":
        rps = ctx.get("refine_points", [])
        j = int(ctx.get("j",0) or 0)
        p = float(rps[j])
        ctx["current_p"] = p
        lcp = float(parsed)
        ctx["current_lcp"] = lcp
        if _is_between_cw(p, awr, lcp, dial_min, dial_max):
            ctx["passed_awr_low"] = True
        elif _is_between_cw(awr, p, lcp, dial_min, dial_max):
            ctx["passed_awr_low"] = False
        ctx["phase"] = "refine_between"
        return True, None

    if pid == "iso3.refine.between":
        ctx["rcp_dir"] = "right" if str(parsed) == "1" else "left"
        ctx["phase"] = "refine_rcp"
        return True, None

    if pid == "iso3.refine.rcp":
        rps = ctx.get("refine_points", [])
        j = int(ctx.get("j",0) or 0)
        p = float(ctx.get("current_p"))
        lcp = float(ctx.get("current_lcp"))
        rcp = float(parsed)
        if ctx.get("rcp_dir") == "right":
            if _is_between_cw(p, awr, rcp, dial_min, dial_max):
                ctx["passed_awr_low"] = True
            elif _is_between_cw(awr, p, rcp, dial_min, dial_max):
                ctx["passed_awr_low"] = False
        row = {
            "id": _next_measurement_id(session, ctx),
            "sweep": sweep_id,
            "wheel_swept": 3,
            "combination_wheel_1": awr,
            "combination_wheel_2": awr,
            "combination_wheel_3": p,
            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": circular_distance(rcp, lcp, dial_min, dial_max),
            "iso_test": "isolate_wheel_3",
            "iso_test_id": iso_test_id,
            "iso_phase": "refine",
            "notes": "",
        }
        session["state"]["measurements"].append(row)
        ctx["j"] = j + 1
        if ctx["j"] >= len(rps):
            ctx["phase"] = "post_refine_plot_offer"
        else:
            ctx["phase"] = "refine_point"
        return True, None

    if pid == "iso3.post_refine.plot_offer":
        if str(parsed) == "1":
            ctx["rows"] = ctx.get("scan_rows", [])
            ctx["phase"] = "post_refine_plot"
            return True, None
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "post_refine_candidates"
        return True, None

    if pid == "iso3.post_refine.plot":
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "post_refine_candidates"
        return True, None

    if pid == "iso3.post_refine.candidates":
        candidates: List[float] = list(parsed) if isinstance(parsed, list) else []
        if not candidates:
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["candidates"] = candidates
        lc = session["state"]["lock_config"]
        wd = lc.get("wheel_data", {}) or {}
        w3 = wd.get("3", {}) or {}
        w3["suspected_gates"] = [float(x) for x in candidates]
        wd["3"] = w3
        lc["wheel_data"] = wd
        session["state"]["lock_config"] = normalize_lock_config(lc)
        ctx["phase"] = "refine_confirm"
        return True, None

    if pid == "iso3.finish":
        _pop(session)
        return True, None

    return True, None


# -----------------------
# Screen: Isolate wheel 2 (3-wheel LRL)
# (Simplified from the procedural version; preserves core sweep semantics.)
# -----------------------

def _prompt_isolate_wheel_2(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    if int(lc.get("wheels", 0) or 0) != 3 or str(lc.get("turn_sequence","")).upper() != "LRL":
        return {"id":"iso2.unsupported","kind":"confirm","text":"Isolate wheel 2 is implemented for 3-wheel LRL locks. Press Enter to return."}

    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
    tol = float(lc["tolerance"]); step_size = max(1e-9, 2.0*tol)

    # init ctx if needed
    if "phase" not in ctx:
        # smart defaults: use known/suspected gates first, else AWL/AWR low points (priority per rules)
        ctx.update({
            "phase": "choose_stops",
            "wheel_1_stop": None,
            "wheel_3_stop": None,
            "sweep_id": _next_sweep_id(session),
            "iso_test_id": _next_iso_test_id(session, "isolate_wheel_2"),
            "n": 1,
            "oi": 0,
            "offsets": None,
            "visited": [],
            "rows": [],
            "candidates": None,
            "offset": None,
            "step_size": step_size,
            "tol": tol,
        })

    phase = ctx["phase"]

    if phase == "choose_stops":
        # Determine suggested stops (no prompts here; we prompt user to confirm/edit)
        w1, w1src = _suggest_stop_for_isolate_wheel_2(session, wheel_num=1)
        w3, w3src = _suggest_stop_for_isolate_wheel_2(session, wheel_num=3)
        ctx["wheel_1_stop_suggest"] = w1
        ctx["wheel_3_stop_suggest"] = w3
        ctx["wheel_1_stop_src"] = w1src
        ctx["wheel_3_stop_src"] = w3src
        ctx["phase"] = "confirm_w1"
        return _prompt_isolate_wheel_2(session, ctx)

    if phase == "confirm_w1":
        s = ctx.get("wheel_1_stop_suggest")
        src = ctx.get("wheel_1_stop_src","")
        if s is None:
            return {"id":"iso2.w1.manual","kind":"float","text":"Wheel 1 stop is required. Enter Wheel 1 stop"}
        return {"id":"iso2.w1.confirm","kind":"float","text":f"Wheel 1 stop suggested = {_fmt_float(s)} ({src}). Enter to accept or type new", "default": _fmt_float(s)}

    if phase == "confirm_w3":
        s = ctx.get("wheel_3_stop_suggest")
        src = ctx.get("wheel_3_stop_src","")
        # Special rule: even if missing, still suggest isolating wheel 3 first, but allow backup low point suggestion.
        note = ""
        if ctx.get("wheel_3_stop_src") in ("missing",):
            note = " (No known/suspected gate; consider isolating wheel 3 first.)"
        if s is None:
            return {"id":"iso2.w3.manual","kind":"float","text":f"Wheel 3 stop is required.{note} Enter Wheel 3 stop"}
        return {"id":"iso2.w3.confirm","kind":"float","text":f"Wheel 3 stop suggested = {_fmt_float(s)} ({src}).{note} Enter to accept or type new", "default": _fmt_float(s)}

    if phase == "intro":
        w1 = ctx["wheel_1_stop"]; w3 = ctx["wheel_3_stop"]
        return {"id":"iso2.intro","kind":"confirm",
                "text":(
                    f"Isolate Wheel 2\n"
                    f"Chosen stops: wheel1={_fmt_float(w1)}, wheel3={_fmt_float(w3)}\n"
                    f"Tolerance=±{_fmt_float(tol)} (step size={_fmt_float(step_size)})\n"
                    f"Dial range={_fmt_float(dial_min)}..{_fmt_float(dial_max)}\n\n"
                    f"Press Enter to start."
                )}

    if phase == "setup_left":
        w1 = ctx["wheel_1_stop"]
        return {"id":"iso2.step2","kind":"confirm",
                "text":f"Turn left (CCW) passing {_fmt_float(w1)} three times, continue until you hit {_fmt_float(w1)}, then stop. Press Enter."}

    if phase == "setup_right":
        w1 = ctx["wheel_1_stop"]
        n = int(ctx.get("n",1) or 1)
        offset = wrap_dial(w1 - (n * step_size), dial_min, dial_max)
        ctx["offset"] = offset
        return {"id":"iso2.step3","kind":"confirm",
                "text":f"Turn right (CW) passing {_fmt_float(w1)} two times, continue slowly until you hit {_fmt_float(offset)}, then stop. Press Enter."}

    if phase == "scan_lcp":
        offset = float(ctx["offset"]); w3 = float(ctx["wheel_3_stop"])
        prev_offset = ctx.get("prev_offset")
        wd = lc.get("wheel_data", {}) or {}
        w3_suspected = (wd.get("3", {}) or {}).get("suspected_gates", []) or []
        w3_suspected = [float(x) for x in w3_suspected if x is not None]
        if ctx.get("wheel_3_stop_override") and w3 is not None:
            w3_suspected = [float(w3)]
        elif not w3_suspected and w3 is not None:
            w3_suspected = [float(w3)]
        w3_suspected_label = ", ".join(_fmt_float(x) for x in w3_suspected)

        if not isinstance(ctx.get("offsets"), list):
            offsets = [offset]
            cur = offset
            max_cycles = int(((dial_max - dial_min + 1.0) / max(step_size, 1e-9)) + 10)
            for _ in range(max_cycles):
                cur = wrap_dial(cur - step_size, dial_min, dial_max)
                if round(cur, 6) == round(offsets[0], 6):
                    break
                offsets.append(cur)
            ctx["offsets"] = offsets
            ctx["oi"] = 0

        offsets = ctx.get("offsets", [])
        oi = int(ctx.get("oi", 0) or 0)
        if oi >= len(offsets):
            ctx["phase"] = "plot_offer"
            return _prompt_isolate_wheel_2(session, ctx)

        offset = float(offsets[oi])
        ctx["offset"] = offset
        cycle_label = f"{oi+1}/{len(offsets)}"
        if oi + 1 >= 2:
            if w3_suspected:
                intro = f"passing {w3_suspected_label} one time"
            else:
                intro = f"passing {_fmt_float(w3)} one time"
            continue_parts = []
            if prev_offset is not None:
                continue_parts.append(f"passing the last checkpoint {_fmt_float(prev_offset)} one time")
            detail_text = "continue turning"
            if continue_parts:
                detail_text = f"continue turning {' and '.join(continue_parts)},"
            text = (
                f"Cycle {cycle_label}\n"
                f"Turn right (CW) {intro}, then {detail_text} until you reach {_fmt_float(offset)}. Stop.\n"
                f"   NOTE: If, during testing, the target point {_fmt_float(offset)} has passed the Wheel 3 suspected gate "
                f"position ({w3_suspected_label}),\n   you must still complete one full revolution after passing "
                f"({w3_suspected_label}) before stopping at {_fmt_float(offset)}.\n"
                f"Turn left (CCW), pass {_fmt_float(offset)} once, then stop at {_fmt_float(w3)}.\n"
                f"Turn right (CW) until you hit the LCP. Enter LCP.\n"
            )
        else:
            text = (
                f"Cycle {cycle_label}\n"
                f"Turn left (CCW) passing {_fmt_float(offset)} one time, continue until you hit {_fmt_float(w3)}, then stop.\n"
                f"Turn right (CW) until you hit LCP. Enter LCP"
            )
        return {"id":"iso2.scan.lcp","kind":"float","text": text}

    if phase == "scan_rcp":
        offsets = ctx.get("offsets", []) or []
        oi = int(ctx.get("oi", 0) or 0)
        cycle_label = f"{oi+1}/{max(1, len(offsets))}"
        return {"id":"iso2.scan.rcp","kind":"float",
                "text":f"Cycle {cycle_label}\nTurn left (CCW) until you hit RCP. Enter RCP"}

    if phase == "plot_offer":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        if ctx.get("candidates") is None:
            rows = ctx.get("rows", [])
            scan_positions = [float(r["combination_wheel_2"]) for r in rows]
            lcps = [float(r["left_contact"]) for r in rows]
            rcps = [float(r["right_contact"]) for r in rows]
            candidates = set()
            for i in range(1, len(rows) - 1):
                if lcps[i] > lcps[i-1] and lcps[i] > lcps[i+1]:
                    candidates.add(scan_positions[i])
                if rcps[i] < rcps[i-1] and rcps[i] < rcps[i+1]:
                    candidates.add(scan_positions[i])
            ctx["candidates"] = sorted(candidates)
        return {
            "id":"iso2.plot_offer",
            "kind":"choice",
            "text": f"Plot sweep {sweep_id} now? (You can also plot later from the main menu.)",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "plot_now":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        ctx["rows"] = ctx.get("rows", [])
        ctx["wheel_swept"] = 2
        return {
            "id":"iso2.plot",
            "kind":"confirm",
            "text": f"Generating plot for sweep {sweep_id} (saved next to the session file)...",
        }

    if phase == "candidates":
        cand_default = ",".join(_fmt_float(x) for x in (ctx.get("candidates") or []))
        return {"id":"iso2.candidates","kind":"csv_floats",
                "text":"Enter candidate gate positions for wheel 2 (comma-separated). Empty = No Possible Gates Observed",
                "default": cand_default}

    if phase == "refine_confirm":
        return {
            "id":"iso2.refine.confirm",
            "kind":"choice",
            "text":"Refine these points?",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No, finish test"},
            ],
        }

    if phase == "refine_range_start":
        candidates = ctx.get("candidates", []) or []
        default_start = _fmt_float(min(candidates)) if candidates else _fmt_float(dial_min)
        return {
            "id":"iso2.refine.range_start",
            "kind":"float",
            "text":"Refinement sweep range start",
            "default": default_start,
        }

    if phase == "refine_range_end":
        candidates = ctx.get("candidates", []) or []
        default_end = _fmt_float(max(candidates)) if candidates else _fmt_float(dial_max)
        return {
            "id":"iso2.refine.range_end",
            "kind":"float",
            "text":"Refinement sweep range end",
            "default": default_end,
        }

    if phase == "refine_range_points":
        return {
            "id":"iso2.refine.range_points",
            "kind":"int",
            "text":"How many refinement points?",
            "default": "10",
        }

    if phase == "refine_intro":
        rps = ctx.get("refine_points", [])
        rstart = ctx.get("refine_range_start")
        rend = ctx.get("refine_range_end")
        return {"id":"iso2.refine.intro","kind":"confirm",
                "text":f"Refinement: {len(rps)} points from {_fmt_float(rstart)} to {_fmt_float(rend)}.\nPress Enter to start refinement."}

    if phase == "refine_lcp":
        rps = ctx.get("refine_points", [])
        i = int(ctx.get("ri",0) or 0)
        if i >= len(rps):
            ctx["phase"] = "post_refine_plot_offer"
            return _prompt_isolate_wheel_2(session, ctx)
        p = float(rps[i])
        w1 = float(ctx["wheel_1_stop"]); w3 = float(ctx["wheel_3_stop"])
        return {"id":"iso2.refine.lcp","kind":"float",
                "text":(
                    f"Refine {i+1}/{len(rps)} @ Wheel 2 = {_fmt_float(p)}\n"
                    f"Turn left (CCW) passing {_fmt_float(w1)} three times, stop on {_fmt_float(w1)}.\n"
                    f"Turn right (CW) passing {_fmt_float(w1)} two times, stop on {_fmt_float(p)}.\n"
                    f"Turn left (CCW) passing {_fmt_float(p)} one time, stop on {_fmt_float(w3)}.\n"
                    f"Turn right (CW) until you hit LCP. Enter LCP"
                )}

    if phase == "refine_rcp":
        i = int(ctx.get("ri",0) or 0)
        p = float(ctx.get("current_p"))
        return {"id":"iso2.refine.rcp","kind":"float",
                "text":f"Refine {i+1}/{len(ctx.get('refine_points',[]))} @ Wheel 2 = {_fmt_float(p)}\nTurn left (CCW) until you hit RCP. Enter RCP"}

    if phase == "post_refine_plot_offer":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        return {
            "id":"iso2.post_refine.plot_offer",
            "kind":"choice",
            "text": f"Plot sweep {sweep_id} now? (You can also plot later from the main menu.)",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "post_refine_plot":
        sweep_id = int(ctx.get("sweep_id", 0) or 0)
        ctx["rows"] = [
            m for m in session["state"]["measurements"]
            if str(m.get("sweep", "")).replace(".", "", 1).isdigit()
            and int(float(m.get("sweep"))) == sweep_id
        ]
        ctx["wheel_swept"] = 2
        return {
            "id":"iso2.post_refine.plot",
            "kind":"confirm",
            "text": f"Generating plot for sweep {sweep_id} (saved next to the session file)...",
        }

    if phase == "post_refine_candidates":
        cand_default = ",".join(_fmt_float(x) for x in (ctx.get("candidates") or []))
        return {"id":"iso2.post_refine.candidates","kind":"csv_floats",
                "text":"Update candidate gate positions for wheel 2 (comma-separated). Empty = No Possible Gates Observed",
                "default": cand_default}

    if phase == "finish":
        sweep_id = int(ctx.get("sweep_id"))
        return {"id":"iso2.finish","kind":"confirm",
                "text":(
                    f"Isolate Wheel 2 complete. Sweep saved as sweep={sweep_id}.\n"
                    "You can plot this sweep later from the main menu.\n"
                    "High Low Tests can help confirm whether suspected gates are on Wheel 2.\n"
                    "These tests can be long, consider saving your progress by typing s.\n\n"
                    "Press Enter to return."
                )}

    return {"id":"iso2.unknown","kind":"confirm","text":"Press Enter to return."}


def _apply_isolate_wheel_2(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = prompt.get("id","")
    if pid in ("iso2.unsupported","iso2.finish","iso2.unknown"):
        _pop(session)
        return True, None

    lc = session["state"]["lock_config"]
    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
    sweep_id = int(ctx.get("sweep_id", _next_sweep_id(session)))
    iso_test_id = int(ctx.get("iso_test_id", _next_iso_test_id(session, "isolate_wheel_2")))
    step_size = float(ctx.get("step_size"))

    if pid in ("iso2.w1.confirm","iso2.w1.manual"):
        ctx["wheel_1_stop"] = float(parsed)
        ctx["phase"] = "confirm_w3"
        return True, None

    if pid in ("iso2.w3.confirm","iso2.w3.manual"):
        w3_val = float(parsed)
        ctx["wheel_3_stop"] = w3_val
        if pid == "iso2.w3.manual":
            ctx["wheel_3_stop_override"] = True
            ctx["wheel_3_stop_src"] = "manual"
        else:
            suggested = ctx.get("wheel_3_stop_suggest")
            if suggested is not None and float(suggested) != w3_val:
                ctx["wheel_3_stop_override"] = True
        ctx["phase"] = "intro"
        return True, None

    if pid == "iso2.intro":
        ctx["phase"] = "setup_left"
        return True, None

    if pid == "iso2.step2":
        ctx["phase"] = "setup_right"
        return True, None

    if pid == "iso2.step3":
        ctx["n"] = int(ctx.get("n",1) or 1) + 1
        ctx["oi"] = 0
        ctx["offsets"] = None
        ctx["phase"] = "scan_lcp"
        return True, None

    if pid == "iso2.scan.lcp":
        ctx["current_lcp"] = float(parsed)
        ctx["phase"] = "scan_rcp"
        return True, None

    if pid == "iso2.scan.rcp":
        lcp = float(ctx.get("current_lcp"))
        rcp = float(parsed)
        w1 = float(ctx["wheel_1_stop"]); w3 = float(ctx["wheel_3_stop"])
        offset = float(ctx["offset"])
        rows = ctx.get("rows", [])
        rows.append({
            "id": _next_measurement_id(session, ctx),
            "sweep": sweep_id,
            "wheel_swept": 2,
            "combination_wheel_1": w1,
            "combination_wheel_2": offset,
            "combination_wheel_3": w3,
            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": circular_distance(rcp, lcp, dial_min, dial_max),
            "iso_test": "isolate_wheel_2",
            "iso_test_id": iso_test_id,
            "iso_phase": "scan",
            "notes": "",
        })
        ctx["rows"] = rows

        # advance offset
        offsets = ctx.get("offsets", []) or [offset]
        oi = int(ctx.get("oi", 0) or 0) + 1
        ctx["prev_offset"] = offset
        if oi >= len(offsets):
            ctx["phase"] = "plot_offer"
            return True, None
        ctx["oi"] = oi
        ctx["offset"] = float(offsets[oi])
        ctx["phase"] = "scan_lcp"
        return True, None

    if pid == "iso2.plot_offer":
        if str(parsed) == "1":
            ctx["rows"] = ctx.get("rows", [])
            ctx["wheel_swept"] = 2
            ctx["phase"] = "plot_now"
            return True, None
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "candidates"
        return True, None

    if pid == "iso2.plot":
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "candidates"
        return True, None

    if pid == "iso2.candidates":
        candidates: List[float] = list(parsed) if isinstance(parsed, list) else []
        # commit scan rows to measurements
        session["state"]["measurements"].extend(ctx.get("rows", []))
        if not candidates:
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["candidates"] = candidates
        wd = lc.get("wheel_data", {}) or {}
        w2 = wd.get("2", {}) or {}
        w2["suspected_gates"] = [float(x) for x in candidates]
        wd["2"] = w2
        lc["wheel_data"] = wd
        session["state"]["lock_config"] = normalize_lock_config(lc)
        ctx["phase"] = "refine_confirm"
        return True, None

    if pid == "iso2.refine.confirm":
        if str(parsed) == "2":
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["phase"] = "refine_range_start"
        return True, None

    if pid == "iso2.refine.range_start":
        ctx["refine_range_start"] = float(parsed)
        ctx["phase"] = "refine_range_end"
        return True, None

    if pid == "iso2.refine.range_end":
        ctx["refine_range_end"] = float(parsed)
        ctx["phase"] = "refine_range_points"
        return True, None

    if pid == "iso2.refine.range_points":
        try:
            n_points = int(parsed)
        except Exception:
            n_points = 0
        if n_points < 1 or n_points > 36000:
            return False, "Enter an integer from 1 to 36000."
        rstart = float(ctx.get("refine_range_start", dial_min))
        rend = float(ctx.get("refine_range_end", dial_max))
        refine_points = _build_range_points(rstart, rend, n_points, dial_min, dial_max)
        measured = set(round(float(m.get("combination_wheel_2", -9999)), 6)
                       for m in session["state"]["measurements"]
                       if m.get("sweep") == sweep_id and m.get("wheel_swept") == 2)
        refine_list = [p for p in refine_points if round(float(p), 6) not in measured]
        ctx["refine_points"] = refine_list
        ctx["ri"] = 0
        ctx["phase"] = "refine_intro" if refine_list else "finish"
        return True, None

    if pid == "iso2.refine.intro":
        ctx["phase"] = "refine_lcp"
        return True, None

    if pid == "iso2.refine.lcp":
        rps = ctx.get("refine_points", [])
        i = int(ctx.get("ri",0) or 0)
        p = float(rps[i])
        ctx["current_p"] = p
        ctx["current_lcp"] = float(parsed)
        ctx["phase"] = "refine_rcp"
        return True, None

    if pid == "iso2.refine.rcp":
        rps = ctx.get("refine_points", [])
        i = int(ctx.get("ri",0) or 0)
        p = float(ctx.get("current_p"))
        lcp = float(ctx.get("current_lcp"))
        rcp = float(parsed)
        w1 = float(ctx["wheel_1_stop"]); w3 = float(ctx["wheel_3_stop"])
        session["state"]["measurements"].append({
            "id": _next_measurement_id(session, ctx),
            "sweep": sweep_id,
            "wheel_swept": 2,
            "combination_wheel_1": w1,
            "combination_wheel_2": p,
            "combination_wheel_3": w3,
            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": circular_distance(rcp, lcp, dial_min, dial_max),
            "iso_test": "isolate_wheel_2",
            "iso_test_id": iso_test_id,
            "iso_phase": "refine",
            "notes": "",
        })
        ctx["ri"] = i + 1
        if ctx["ri"] >= len(rps):
            ctx["phase"] = "post_refine_plot_offer"
        else:
            ctx["phase"] = "refine_lcp"
        return True, None

    if pid == "iso2.post_refine.plot_offer":
        if str(parsed) == "1":
            ctx["rows"] = ctx.get("rows", [])
            ctx["wheel_swept"] = 2
            ctx["phase"] = "post_refine_plot"
            return True, None
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "post_refine_candidates"
        return True, None

    if pid == "iso2.post_refine.plot":
        ctx["_skip_plot_once"] = True
        ctx["phase"] = "post_refine_candidates"
        return True, None

    if pid == "iso2.post_refine.candidates":
        candidates: List[float] = list(parsed) if isinstance(parsed, list) else []
        if not candidates:
            ctx["_skip_finish_plot_once"] = True
            ctx["phase"] = "finish"
            return True, None
        ctx["candidates"] = candidates
        wd = lc.get("wheel_data", {}) or {}
        w2 = wd.get("2", {}) or {}
        w2["suspected_gates"] = [float(x) for x in candidates]
        wd["2"] = w2
        lc["wheel_data"] = wd
        session["state"]["lock_config"] = normalize_lock_config(lc)
        ctx["phase"] = "refine_confirm"
        return True, None

    if pid == "iso2.finish":
        _pop(session)
        return True, None

    return True, None


# -----------------------
# Screen: High Low Test (3-wheel LRL)
# -----------------------

def _prompt_high_low_test(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    if int(lc.get("wheels", 0) or 0) != 3 or str(lc.get("turn_sequence","")).upper() != "LRL":
        return {"id":"high_low.unsupported","kind":"confirm","text":"High Low Test is implemented for 3-wheel LRL locks. Press Enter to return."}

    dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])

    if "gate" not in ctx:
        suspected = (lc.get("wheel_data", {}) or {}).get("3", {}).get("suspected_gates", []) or []
        prompt = {"id":"high_low.gate","kind":"float","text":"Suspected true gate"}
        if suspected:
            prompt["default"] = _fmt_float(suspected[0])
        return prompt

    if "offset" not in ctx:
        return {"id":"high_low.offset","kind":"float","text":"Offset to test","default":"10"}

    if "combos" not in ctx:
        gate = float(ctx["gate"])
        offset = float(ctx["offset"])
        combos = [
            [wrap_dial(gate + offset, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max)],
            [wrap_dial(gate, dial_min, dial_max), wrap_dial(gate + offset, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max)],
            [wrap_dial(gate, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max), wrap_dial(gate + offset, dial_min, dial_max)],
        ]
        ctx["combos"] = combos
        ctx["idx"] = 0
        ctx["rows"] = []
        ctx["phase"] = "combo_lcp"
        ctx["mode"] = "high"
        ctx["test_id"] = _next_high_low_test_id(session)

    phase = str(ctx.get("phase", "combo_lcp"))
    idx = int(ctx.get("idx", 0) or 0)
    combos = ctx.get("combos", [])

    if phase == "combo_lcp":
        if idx >= len(combos):
            ctx["phase"] = "ask_low" if ctx.get("mode") == "high" else "result"
            return _prompt_high_low_test(session, ctx)
        c1, c2, c3 = [float(x) for x in combos[idx]]
        return {
            "id":"high_low.lcp",
            "kind":"float",
            "text":(
                f"High Low Test\nCombination {idx+1}/3: [{_fmt_float(c1)}, {_fmt_float(c2)}, {_fmt_float(c3)}]\n"
                f"Turn left (CCW) passing {_fmt_float(c1)} three times, and continue until you hit {_fmt_float(c1)}, then stop.\n"
                f"Turn right (CW) passing {_fmt_float(c1)} two times, and continue until you hit {_fmt_float(c2)}, then stop.\n"
                f"Turn left (CCW) passing {_fmt_float(c2)} one time, and continue until you hit {_fmt_float(c3)}, then stop.\n"
                "Turn right (CW) to the LCP. Enter LCP"
            ),
        }

    if phase == "combo_rcp":
        return {"id":"high_low.rcp","kind":"float","text":"Turn left (CCW) to the RCP. Enter RCP"}

    if phase == "ask_low":
        return {
            "id":"high_low.ask_low",
            "kind":"choice",
            "text":"Run Low Test (minus offset)?",
            "choices": [
                {"key":"1","label":"Yes"},
                {"key":"2","label":"No"},
            ],
        }

    if phase == "result":
        rows = ctx.get("rows", []) or []
        result_line = "Result inconclusive (tie)."
        if rows:
            max_by_case: Dict[int, float] = {}
            for r in rows:
                try:
                    case = int(r.get("hw_case", 0) or 0)
                except Exception:
                    continue
                if case < 1 or case > 3:
                    continue
                w = float(r.get("contact_width", 0.0))
                if case not in max_by_case or w > max_by_case[case]:
                    max_by_case[case] = w
            if max_by_case:
                max_w = max(max_by_case.values())
                winners = [c for c, w in max_by_case.items() if w == max_w]
                if len(winners) == 1:
                    case = winners[0]
                    result_line = f"Largest contact width in case {case}.\nIf contact width is much larger, than gate may be on Wheel {case}.\nIf not, other wheels may be overshadowing (possibly low point used earlier was a gate on another wheel.)"
        return {
            "id":"high_low.done",
            "kind":"confirm",
            "text":(
                "High Low Test complete.\n"
                f"{result_line}\n\n"
                "You can plot this test later from the main menu.\n\n"
                "Press Enter to return."
            ),
        }

    return {"id":"high_low.unknown","kind":"confirm","text":"Press Enter to return."}


def _apply_high_low_test(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = prompt.get("id","")
    if pid in ("high_low.unsupported","high_low.unknown","high_low.done"):
        _pop(session)
        return True, None

    if pid == "high_low.gate":
        ctx["gate"] = float(parsed)
        return True, None

    if pid == "high_low.offset":
        try:
            offset = float(parsed)
        except Exception:
            return False, "Enter a number."
        if offset <= 0:
            return False, "Offset must be positive."
        ctx["offset"] = offset
        return True, None

    if pid == "high_low.lcp":
        ctx["current_lcp"] = float(parsed)
        ctx["phase"] = "combo_rcp"
        return True, None

    if pid == "high_low.rcp":
        lc = session["state"]["lock_config"]
        dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
        combos = ctx.get("combos", [])
        idx = int(ctx.get("idx", 0) or 0)
        if idx >= len(combos):
            ctx["phase"] = "ask_low" if ctx.get("mode") == "high" else "result"
            return True, None
        c1, c2, c3 = [float(x) for x in combos[idx]]
        lcp = float(ctx.get("current_lcp"))
        rcp = float(parsed)
        row = {
            "id": _next_measurement_id(session, ctx),
            "combination_wheel_1": c1,
            "combination_wheel_2": c2,
            "combination_wheel_3": c3,
            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": circular_distance(rcp, lcp, dial_min, dial_max),
            "sweep": None,
            "high_low_test": int(ctx.get("test_id")),
            "hw_gate": float(ctx.get("gate")),
            "hw_offset": float(ctx.get("offset")),
            "hw_increment": float(ctx.get("offset")),
            "hw_case": idx + 1,
            "hw_type": str(ctx.get("mode", "high")),
            "lock_config": {"dial_min": dial_min, "dial_max": dial_max},
            "notes": "",
        }
        session["state"]["measurements"].append(row)
        ctx.setdefault("rows", []).append(row)
        ctx["idx"] = idx + 1
        ctx["phase"] = "combo_lcp"
        return True, None

    if pid == "high_low.ask_low":
        if str(parsed) == "1":
            lc = session["state"]["lock_config"]
            dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
            gate = float(ctx.get("gate"))
            offset = float(ctx.get("offset"))
            low_offset = -offset
            combos = [
                [wrap_dial(gate + low_offset, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max)],
                [wrap_dial(gate, dial_min, dial_max), wrap_dial(gate + low_offset, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max)],
                [wrap_dial(gate, dial_min, dial_max), wrap_dial(gate, dial_min, dial_max), wrap_dial(gate + low_offset, dial_min, dial_max)],
            ]
            ctx["combos"] = combos
            ctx["idx"] = 0
            ctx["mode"] = "low"
            ctx["phase"] = "combo_lcp"
            return True, None
        ctx["phase"] = "result"
        return True, None

    _pop(session)
    return True, None


def _first_num(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and val:
        try:
            return float(val[0])
        except Exception:
            return None
    return None


def _w3_gate_info(lc: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    wd = lc.get("wheel_data", {}) or {}
    w3 = wd.get("3", {}) or {}
    gate = _first_num(w3.get("known_gates"))
    if gate is not None:
        return gate, "known gate"
    gate = _first_num(w3.get("suspected_gates"))
    if gate is not None:
        return gate, "suspected gate"
    return None, None


def _suggest_stop_for_isolate_wheel_2(session: Session, wheel_num: int) -> Tuple[Optional[float], str]:
    """
    Stop suggestion priority:
      a) known gates (first item)
      b) suspected gates (first item)
      c) fallback to AWL/AWR low points using the LRL rules:
         - For wheel 1 or 3 and LRL: AWL else AWR
         - For wheel 2 and LRL: AWR else AWL
    Special rule handled by prompt messaging (wheel 3 missing gate still suggests isolating wheel 3 first).
    """
    lc = session["state"]["lock_config"]
    wd = lc.get("wheel_data", {}) or {}
    wdata = wd.get(str(wheel_num), {}) or {}

    known = _first_num(wdata.get("known_gates"))
    if known is not None:
        return known, "known gate"

    suspected = _first_num(wdata.get("suspected_gates"))
    if suspected is not None:
        return suspected, "suspected gate"

    # fallback
    turn_seq = str(lc.get("turn_sequence","")).strip().upper()
    awl = lc.get("awl_low_point")
    awr = lc.get("awr_low_point")

    if turn_seq == "LRL":
        if wheel_num in (1,3):
            if awl is not None:
                return float(awl), "AWL low point"
            if awr is not None:
                return float(awr), "AWR low point"
        else:  # wheel 2
            if awr is not None:
                return float(awr), "AWR low point"
            if awl is not None:
                return float(awl), "AWL low point"
    else:  # RLR
        # follow requested mapping (simplified; can be expanded)
        if wheel_num in (1,3):
            if awr is not None:
                return float(awr), "AWR low point"
            if awl is not None:
                return float(awl), "AWL low point"
        else:
            if awr is not None:
                return float(awr), "AWR low point"
            if awl is not None:
                return float(awl), "AWL low point"

    return None, "missing"


# -----------------------
# Candidate-combination search (simplified)
# TODO: Add optimized per-wheel enumeration paths to reduce unnecessary tries.
# -----------------------

def _effective_gate_list(wdata: Dict[str, Any]) -> List[float]:
    # TODO: Use false gates to optimize testing (skip known bad positions or use lower points to measure other wheels).
    gates = []
    for k in ("known_gates", "suspected_gates"):
        v = wdata.get(k, [])
        if isinstance(v, list):
            for x in v:
                try:
                    gates.append(float(x))
                except Exception:
                    pass
    # remove false
    false = set()
    for x in wdata.get("false_gates", []) or []:
        try:
            false.add(float(x))
        except Exception:
            pass
    gates = [g for g in gates if float(g) not in false]
    # unique preserving order
    seen = set()
    out = []
    for g in gates:
        k = round(float(g),6)
        if k in seen:
            continue
        seen.add(k)
        out.append(float(g))
    return out


def _prompt_candidate_combo_all(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    # TODO: Use optimized paths for iterating through points to reduce attempts.
    lc = session["state"]["lock_config"]
    wd = lc.get("wheel_data", {}) or {}
    wheels = int(lc.get("wheels",0) or 0)

    if "init" not in ctx:
        if not ctx.get("intro_ack"):
            ctx["shown_intro"] = True
            return {
                "id":"enumall.intro",
                "kind":"confirm",
                "text":(
                    "Candidate-combination search (All Wheels)\n"
                    "This test will guide testing all possible combinations of known/suspected gates.\n"
                    "It requires known/suspected gates to limit the number of enumerations/attempts, and\n"
                    "does not try all combinations possible for the given range and tolerance of the dials.\n"
                    "Press Enter to start."
                ),
            }
        candidates = {}
        for w in range(1, wheels+1):
            gates = _effective_gate_list(wd.get(str(w), {}) or {})
            if not gates:
                return {"id":"enumall.no_gates","kind":"confirm","text":f"Wheel {w} has no known/suspected gates. Press Enter to return."}
            candidates[w] = list(gates)
        combos = list(product(*[candidates[w] for w in range(1,wheels+1)]))
        ctx.update({"init":True, "candidates": candidates, "combos": combos, "i": 0, "attempts": []})

    combos = ctx["combos"]
    i = int(ctx.get("i",0) or 0)
    if i >= len(combos):
        return {"id":"enumall.done","kind":"confirm","text":"Candidate-combination search complete. Press Enter to return."}

    combo = combos[i]
        # result prompt: avoid reserved q/s/u/a
    return {
        "id":"enumall.result",
        "kind":"choice",
        "text":(
            f"Try combination: {_fmt_float_tuple(list(combo))}\n"
            f"Enter result:\n"
            f"  0) Closed (record attempt)\n"
            f"  1) Opened (SUCCESS and stop)\n"
            f"  2) Error/unknown (record attempt)\n"
            f"  3) Skip (do not record)\n"
        ),
        "choices":[{"key":"0","label":"Closed"},{"key":"1","label":"Opened"},{"key":"2","label":"Error"},{"key":"3","label":"Skip"}],
    }


def _apply_candidate_combo_all(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = prompt.get("id","")
    if pid == "enumall.intro":
        ctx["intro_ack"] = True
        return True, None
    if pid in ("enumall.no_gates","enumall.done"):
        _pop(session)
        return True, None

    if pid == "enumall.result":
        combos = ctx["combos"]
        i = int(ctx.get("i",0) or 0)
        combo = combos[i]
        res = str(parsed)

        if res == "1":
            # success
            session["state"]["metadata"]["found_combination"] = list(combo)
            lc = dict(session["state"]["lock_config"])
            wd = lc.get("wheel_data", {}) or {}
            wheels = int(lc.get("wheels", 0) or 0)
            for i in range(1, wheels + 1):
                wdata = wd.get(str(i), {}) or {}
                known = wdata.get("known_gates", []) or []
                suspected = wdata.get("suspected_gates", []) or []
                val = float(combo[i - 1])
                if val not in known:
                    known.append(val)
                suspected = [x for x in suspected if float(x) != val]
                wdata["known_gates"] = known
                wdata["suspected_gates"] = suspected
                wd[str(i)] = wdata
            lc["wheel_data"] = wd
            session["state"]["lock_config"] = normalize_lock_config(lc)
            ctx["i"] = len(combos)  # stop
            return True, None

        if res in ("0","2"):
            # record attempt
            att = {"combo": list(combo), "status": "closed" if res=="0" else "error"}
            ctx["attempts"].append(att)
            session["state"]["metadata"].setdefault("enumeration_attempts", []).append(att)

        # skip or recorded -> move next
        ctx["i"] = i + 1
        return True, None

    _pop(session)
    return True, None


def _prompt_single_wheel_sweep(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    # TODO: Use optimized paths for iterating through points to reduce attempts.
    lc = session["state"]["lock_config"]
    wheels = int(lc.get("wheels",0) or 0)

    if "step" not in ctx:
        ctx["step"] = "choose_wheel"

    if ctx["step"] == "choose_wheel":
        choices = [{"key": str(w), "label": f"Wheel {w}"} for w in range(1, wheels+1)]
        wd = lc.get("wheel_data", {}) or {}
        default_key = None
        for w in range(1, wheels + 1):
            wdata = wd.get(str(w), {}) or {}
            if not (wdata.get("known_gates") or []) and not (wdata.get("suspected_gates") or []):
                default_key = str(w)
                break
        suggest = f" (suggest Wheel {default_key})" if default_key else ""
        return {
            "id":"enumw.wheel",
            "kind":"choice",
            "text":f"Choose a wheel to enumerate{suggest}",
            "choices": choices,
            "default": default_key,
        }

    # after choosing wheel, prompt fixed combination for other wheels (simple)
    if ctx["step"] == "enter_fixed":
        w = int(ctx["wheel"])
        others = [i for i in range(1,wheels+1) if i != w]
        if "fixed" not in ctx:
            ctx["fixed"] = {}
            ctx["fixed_idx"] = 0
        idx = int(ctx.get("fixed_idx",0) or 0)
        if idx >= len(others):
            ctx["step"] = "run"
            return _prompt_single_wheel_sweep(session, ctx)
        ow = others[idx]
        wdata = (lc.get("wheel_data", {}) or {}).get(str(ow), {}) or {}
        known = _first_num(wdata.get("known_gates"))
        suspected = _first_num(wdata.get("suspected_gates"))
        default = known if known is not None else suspected
        prompt = {"id":"enumw.fixed", "kind":"float", "text":f"Enter fixed stop for wheel {ow} (used for all tries)"}
        if default is not None:
            prompt["default"] = _fmt_float(default)
        return prompt

    if ctx["step"] == "run":
        # build candidates for chosen wheel
        lc = session["state"]["lock_config"]
        w = int(ctx["wheel"])
        if "gates" not in ctx:
            dial_min = float(lc["dial_min"]); dial_max = float(lc["dial_max"])
            tol = float(lc.get("tolerance", 1.0) or 1.0)
            step = max(1e-9, 2.0 * tol)
            span = float(dial_max - dial_min)
            steps = int(span / step) + 1
            gates = [wrap_dial(dial_min + (i * step), dial_min, dial_max) for i in range(max(1, steps))]
            ctx["gates"] = gates
            ctx["gi"] = 0
        gi = int(ctx.get("gi",0) or 0)
        if gi >= len(ctx["gates"]):
            return {"id":"enumw.done","kind":"confirm","text":"Single-wheel sweep complete. Press Enter to return."}
        g = float(ctx["gates"][gi])

        # build combo for display
        wheels = int(lc.get("wheels",0) or 0)
        combo = []
        for i in range(1, wheels+1):
            if i == w:
                combo.append(g)
            else:
                combo.append(float(ctx["fixed"][str(i)]))
        return {
            "id":"enumw.result",
            "kind":"choice",
            "text":(
                f"Refine {gi+1}/{len(ctx['gates'])} @ Wheel {w} = {_fmt_float(g)}\n"
                f"Try combination (wheel {w} varied): {_fmt_float_tuple(combo)}\n\n"
                f"Turn left (CCW) passing {_fmt_float(combo[0])} three times, stop on {_fmt_float(combo[0])}.\n"
                f"Turn right (CW) passing {_fmt_float(combo[0])} two times, stop on {_fmt_float(combo[1])}.\n"
                f"Turn left (CCW) passing {_fmt_float(combo[1])} one time, stop on {_fmt_float(combo[2])}.\n"
                "Turn right (CW) to test position.\n\n"
                f"Enter result:\n"
                f"  0) Closed (record attempt)\n"
                f"  1) Opened (SUCCESS and stop)\n"
                f"  2) Error/unknown (record attempt)\n"
                f"  3) Skip (do not record)\n"
            ),
            "choices":[{"key":"0","label":"Closed"},{"key":"1","label":"Opened"},{"key":"2","label":"Error"},{"key":"3","label":"Skip"}],
        }

    return {"id":"enumw.unknown","kind":"confirm","text":"Press Enter to return."}


def _apply_single_wheel_sweep(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = prompt.get("id","")
    if pid in ("enumw.no_gates","enumw.done","enumw.unknown"):
        _pop(session)
        return True, None

    if pid == "enumw.wheel":
        ctx["wheel"] = int(parsed)
        ctx["step"] = "enter_fixed"
        return True, None

    if pid == "enumw.fixed":
        lc = session["state"]["lock_config"]
        wheels = int(lc.get("wheels",0) or 0)
        w = int(ctx["wheel"])
        others = [i for i in range(1,wheels+1) if i != w]
        idx = int(ctx.get("fixed_idx",0) or 0)
        ow = others[idx]
        ctx["fixed"][str(ow)] = float(parsed)
        ctx["fixed_idx"] = idx + 1
        return True, None

    if pid == "enumw.result":
        lc = session["state"]["lock_config"]
        wheels = int(lc.get("wheels",0) or 0)
        w = int(ctx["wheel"])
        gi = int(ctx.get("gi",0) or 0)
        g = float(ctx["gates"][gi])
        combo = []
        for i in range(1, wheels+1):
            if i == w:
                combo.append(g)
            else:
                combo.append(float(ctx["fixed"][str(i)]))
        res = str(parsed)
        if res == "1":
            session["state"]["metadata"]["found_combination"] = list(combo)
            wd = lc.get("wheel_data", {}) or {}
            for i in range(1, wheels + 1):
                wdata = wd.get(str(i), {}) or {}
                known = wdata.get("known_gates", []) or []
                suspected = wdata.get("suspected_gates", []) or []
                val = float(combo[i - 1])
                if val not in known:
                    known.append(val)
                suspected = [x for x in suspected if float(x) != val]
                wdata["known_gates"] = known
                wdata["suspected_gates"] = suspected
                wd[str(i)] = wdata
            lc["wheel_data"] = wd
            session["state"]["lock_config"] = normalize_lock_config(lc)
            ctx["gi"] = len(ctx["gates"])  # stop
            return True, None
        if res in ("0","2"):
            att = {"combo": list(combo), "status": "closed" if res=="0" else "error"}
            session["state"]["metadata"].setdefault("enumeration_attempts", []).append(att)
        ctx["gi"] = gi + 1
        return True, None

    # if we just finished fixed entries:
    if ctx.get("step") == "run":
        return True, None

    if ctx.get("step") == "enter_fixed":
        # transition handled by prompt logic
        return True, None

    return True, None


# -----------------------
# Deterministic ID helpers
# -----------------------

# -----------------------
# Screen: Analyze / Plot
# -----------------------

def _prompt_plot_sweep(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    phase = str(ctx.get("phase", "choose") or "choose")
    sweeps = sorted({
        int(float(m.get("sweep")))
        for m in session["state"]["measurements"]
        if m.get("sweep") is not None and str(m.get("sweep")).replace(".", "", 1).isdigit()
    })
    ctx["sweeps"] = sweeps

    if not sweeps:
        return {
            "id": "plot_sweep.none",
            "kind": "confirm",
            "text": "No sweep data available.\nPress Enter to return.",
        }

    if phase == "generate":
        sid = int(ctx.get("sweep_id"))
        # infer wheel_swept for plot output
        wheel = None
        for m in session["state"]["measurements"]:
            if str(m.get("sweep", "")).replace(".", "", 1).isdigit() and int(float(m.get("sweep"))) == sid:
                try:
                    wheel = int(m.get("wheel_swept", 0) or 0)
                except Exception:
                    wheel = 0
                break
        ctx["wheel_swept"] = int(wheel or 0)
        return {
            "id": "plot_sweep.generate",
            "kind": "confirm",
            "text": f"Generating plot for sweep {sid} (saved next to the session file)...",
        }

    choices: List[Dict[str, str]] = []
    for i, sid in enumerate(sweeps, 1):
        choices.append({"key": str(i), "label": f"Sweep {sid}"})
    choices.append({"key": str(len(choices) + 1), "label": "Return"})
    return {
        "id": "plot_sweep.choose",
        "kind": "choice",
        "text": "Select a sweep to plot:",
        "choices": choices,
    }


def _prompt_plot_high_low(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    phase = str(ctx.get("phase", "choose") or "choose")
    ids = sorted({
        int(float(m.get("high_low_test")))
        for m in session["state"]["measurements"]
        if m.get("high_low_test") is not None and str(m.get("high_low_test")).replace(".", "", 1).isdigit()
    })
    ctx["high_low_ids"] = ids

    if not ids:
        return {
            "id": "plot_high_low.none",
            "kind": "confirm",
            "text": "No High Low Tests recorded in this session.\nPress Enter to return.",
        }

    if phase == "generate":
        tid = int(ctx.get("test_id"))
        return {
            "id": "plot_high_low.generate",
            "kind": "confirm",
            "text": f"Generating plot for High Low Test {tid} (saved next to the session file)...",
        }

    choices: List[Dict[str, str]] = []
    for i, tid in enumerate(ids, 1):
        choices.append({"key": str(i), "label": f"Test ID {tid}"})
    choices.append({"key": str(len(choices) + 1), "label": "Return"})
    return {
        "id": "plot_high_low.choose",
        "kind": "choice",
        "text": "Select a High Low Test to plot:",
        "choices": choices,
    }


def _apply_analyze_menu(session: Session, ctx: Dict[str, Any], choice: str) -> Tuple[bool, Optional[str]]:
    if choice == "1":
        _push(session, "plot_sweep", {})
        return True, None
    if choice == "2":
        _push(session, "plot_high_low", {})
        return True, None
    if choice == "3":
        _pop(session)
        return True, None
    return False, "Invalid option"


def _apply_plot_sweep(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = str(prompt.get("id", ""))
    if pid == "plot_sweep.none":
        _pop(session)
        return True, None

    if pid == "plot_sweep.generate":
        _pop(session)  # back to analyze menu
        return True, None

    if pid == "plot_sweep.choose":
        sweeps = ctx.get("sweeps", []) or []
        try:
            idx = int(str(parsed))
        except Exception:
            return False, "Invalid option"
        if idx == len(sweeps) + 1:
            _pop(session)
            return True, None
        if idx < 1 or idx > len(sweeps):
            return False, "Invalid option"
        ctx["sweep_id"] = int(sweeps[idx - 1])
        ctx["phase"] = "generate"
        return True, None

    _pop(session)
    return True, None


def _apply_plot_high_low(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = str(prompt.get("id", ""))
    if pid == "plot_high_low.none":
        _pop(session)
        return True, None

    if pid == "plot_high_low.generate":
        _pop(session)
        return True, None

    if pid == "plot_high_low.choose":
        ids = ctx.get("high_low_ids", []) or []
        try:
            idx = int(str(parsed))
        except Exception:
            return False, "Invalid option"
        if idx == len(ids) + 1:
            _pop(session)
            return True, None
        if idx < 1 or idx > len(ids):
            return False, "Invalid option"
        ctx["test_id"] = int(ids[idx - 1])
        ctx["phase"] = "generate"
        return True, None

    _pop(session)
    return True, None


# -----------------------
# Screen: Tutorial (guided suggestions)
# -----------------------

def _tutorial_effective_candidates(lc: Dict[str, Any], wheel: int) -> List[float]:
    wd = (lc.get("wheel_data", {}) or {}).get(str(wheel), {}) or {}
    known = wd.get("known_gates") or []
    suspected = wd.get("suspected_gates") or []
    if known:
        return [float(x) for x in known]
    if suspected:
        return [float(x) for x in suspected]
    return []


def _tutorial_wheels_with_known_or_suspected(lc: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for w in (1, 2, 3):
        wd = (lc.get("wheel_data", {}) or {}).get(str(w), {}) or {}
        if (wd.get("known_gates") or []) or (wd.get("suspected_gates") or []):
            out.append(w)
    return out


def _tutorial_decide(lc: Dict[str, Any]) -> Tuple[str, Optional[int], str]:
    """Return (action_kind, detail, human_label)."""
    wks = _tutorial_wheels_with_known_or_suspected(lc)

    if len(wks) == 3:
        kind, detail = ("ENUM_ALL", None)
    elif len(wks) == 2:
        missing = [w for w in (1, 2, 3) if w not in wks][0]
        kind, detail = ("ENUM_WHEEL", missing)
    elif len(wks) == 1:
        kind, detail = (("ISOLATE_WHEEL_2", None) if wks[0] == 3 else ("ISOLATE_WHEEL_3", None))
    else:
        kind, detail = ("ISOLATE_WHEEL_3", None)

    # prerequisites
    if kind == "ISOLATE_WHEEL_3" and lc.get("awr_low_point") is None:
        return ("FIND_AWR", None, "Find the All Wheels Right (AWR) low point")
    if kind == "ISOLATE_WHEEL_2" and lc.get("awl_low_point") is None:
        return ("FIND_AWL", None, "Find the All Wheels Left (AWL) low point")

    if kind == "ISOLATE_WHEEL_3":
        return ("ISOLATE_WHEEL_3", None, "Isolate Wheel 3")
    if kind == "ISOLATE_WHEEL_2":
        return ("ISOLATE_WHEEL_2", None, "Isolate Wheel 2")
    if kind == "ENUM_ALL":
        return ("ENUM_ALL", None, "Candidate-combination search using current candidates")

    return ("ENUM_WHEEL", detail, f"Single-wheel sweep for Wheel {detail}")


def _tutorial_plan_actions(lc: Dict[str, Any]) -> List[Tuple[str, Optional[int], str]]:
    actions: List[Tuple[str, Optional[int], str]] = []
    wks = _tutorial_wheels_with_known_or_suspected(lc)
    if lc.get("awr_low_point") is None:
        actions.append(("FIND_AWR", None, "Find the All Wheels Right (AWR) low point"))
    # After AWR, isolate wheel 3 only if we don't already have a known/suspected gate.
    if 3 not in wks:
        actions.append(("ISOLATE_WHEEL_3", None, "Isolate Wheel 3"))
    actions.append(("HIGH_LOW", None, "High Low Test (confirm Wheel 3 gate)"))
    if lc.get("awl_low_point") is None:
        actions.append(("FIND_AWL", None, "Find the All Wheels Left (AWL) low point"))
    if 2 not in wks:
        actions.append(("ISOLATE_WHEEL_2", None, "Isolate Wheel 2"))
    else:
        actions.append(("HIGH_LOW", None, "High Low Test (confirm Wheel 2 gate)"))
    if len(wks) == 2:
        missing = [w for w in (1, 2, 3) if w not in wks][0]
        actions.append(("ENUM_WHEEL", missing, f"Single-wheel sweep for Wheel {missing}"))
    elif len(wks) >= 2:
        actions.append(("ENUM_ALL", None, "Candidate-combination search using current candidates"))
    return actions


def _prompt_tutorial(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    lc = session["state"]["lock_config"]
    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        return {
            "id": "tutorial.unsupported",
            "kind": "confirm",
            "text": "Tutorial is only implemented for 3-wheel LRL locks at this time.\nPress Enter to return.",
        }

    if ctx.get("last_action") in ("ENUM_ALL", "ENUM_WHEEL") and not ctx.get("_celebrated", False):
        all_known = True
        for w in range(1, wheels + 1):
            wd = (lc.get("wheel_data", {}) or {}).get(str(w), {}) or {}
            if not (wd.get("known_gates") or []):
                all_known = False
                break
        if all_known:
            ctx["_celebrated"] = True
            _push(session, "tutorial_done", {})
            return _prompt_tutorial_done(session, _top(session)["ctx"])

    step = int(ctx.get("step", 1) or 1)
    actions = _tutorial_plan_actions(lc)
    total_steps = len(actions) if actions else step
    if ctx.pop("advance_step", False):
        step = min(step + 1, total_steps) if actions else step + 1
        if ctx.get("last_action") == "FIND_AWR" and lc.get("awr_low_point") is not None:
            for i, (kind, _, _) in enumerate(actions, 1):
                if kind == "ISOLATE_WHEEL_3":
                    step = i
                    break
        if ctx.get("last_action") == "ISOLATE_WHEEL_3":
            for i, (kind, _, _) in enumerate(actions, 1):
                if kind == "HIGH_LOW":
                    step = i
                    break
    if actions:
        step = min(step, total_steps)
    ctx["step"] = step

    lines: List[str] = []
    lines.append(f"Step {step} of {total_steps} — Current lock knowledge:")
    lines.append("Lock configuration:")
    lines.append(f"  Wheels: {lc.get('wheels')}")
    lines.append(f"  Dial range: {_fmt_float(lc.get('dial_min'))}..{_fmt_float(lc.get('dial_max'))}")
    lines.append(f"  Tolerance: ±{_fmt_float(lc.get('tolerance'))}")
    lines.append(f"  Turn sequence: {lc.get('turn_sequence')}")
    lines.append(f"  Flies: {lc.get('flies')}")
    lines.append(f"  Make: {lc.get('make')}")
    lines.append(f"  Fence type: {lc.get('fence_type')}")
    lines.append(f"  UL: {lc.get('ul')}")
    lines.append(f"  Oval wheels: {lc.get('oval_wheels')}")
    lines.append(f"  AWL low point: {_fmt_float(lc.get('awl_low_point'))}")
    lines.append(f"  AWR low point: {_fmt_float(lc.get('awr_low_point'))}")
    lines.append(f"  Approx LCP: {_fmt_float(lc.get('approx_lcp_location'))}")
    lines.append(f"  Approx RCP: {_fmt_float(lc.get('approx_rcp_location'))}")
    lines.append("Gate knowledge:")
    for w in (1, 2, 3):
        wd = (lc.get("wheel_data", {}) or {}).get(str(w), {}) or {}
        known = wd.get("known_gates", []) or []
        suspected = wd.get("suspected_gates", []) or []
        using = _tutorial_effective_candidates(lc, w)
        lines.append(
            "  Wheel {w}: known={known} | suspected={suspected} | using={using}".format(
                w=w,
                known=_fmt_float_list(known),
                suspected=_fmt_float_list(suspected),
                using=_fmt_float_list(using),
            )
        )

    if actions:
        action_kind, detail, label = actions[max(0, step - 1)]
    else:
        action_kind, detail, label = _tutorial_decide(lc)
    ctx["action_kind"] = action_kind
    ctx["action_detail"] = detail
    ctx["action_label"] = label

    text = (
        "Tutorial\n\n"
        + "\n".join(lines)
        + "\n\nSuggested action:\n  → "
        + label
    )
    return {
        "id": "tutorial.show",
        "kind": "choice",
        "text": text,
        "choices": [
            {"key": "1", "label": "Try suggested action"},
            {"key": "2", "label": "Skip this suggestion"},
            {"key": "3", "label": "Return (Exit Tutorial)"},
        ],
    }


def _apply_tutorial(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    pid = str(prompt.get("id", ""))
    if pid == "tutorial.unsupported":
        _pop(session)
        return True, None

    if pid == "tutorial.show":
        if str(parsed) == "3":
            _pop(session)
            return True, None
        if str(parsed) == "2":
            ctx["advance_step"] = True
            return True, None
        kind = str(ctx.get("action_kind", ""))
        detail = ctx.get("action_detail", None)
        ctx["last_action"] = kind
        if kind == "FIND_AWR":
            _push(session, "find_awr", {})
        elif kind == "FIND_AWL":
            _push(session, "find_awl", {})
        elif kind == "ISOLATE_WHEEL_3":
            _push(session, "isolate_wheel_3", {})
        elif kind == "ISOLATE_WHEEL_2":
            _push(session, "isolate_wheel_2", {})
        elif kind == "HIGH_LOW":
            _push(session, "high_low_test", {})
        elif kind == "ENUM_ALL":
            _push(session, "candidate_combo_all", {})
        elif kind == "ENUM_WHEEL":
            _push(session, "single_wheel_sweep", {"wheel": int(detail or 1)})
        ctx["advance_step"] = True
        return True, None

    _pop(session)
    return True, None


def _prompt_tutorial_done(session: Session, ctx: Dict[str, Any]) -> PromptSpec:
    return {
        "id": "tutorial.done",
        "kind": "confirm",
        "text": "Congratulations, you got an open!\nPress Enter to return (Exit Tutorial).",
    }


def _apply_tutorial_done(session: Session, ctx: Dict[str, Any], parsed: Any, prompt: PromptSpec) -> Tuple[bool, Optional[str]]:
    session["runtime"]["stack"] = [{"screen": "main_menu", "ctx": {}}]
    return True, None



def _next_measurement_id(session: Session, ctx: Optional[Dict[str, Any]] = None) -> int:
    # Deterministic: id = max existing + 1
    meas = session["state"]["measurements"]
    mmax = 0
    for m in meas:
        try:
            mmax = max(mmax, int(m.get("id", 0) or 0))
        except Exception:
            pass
    # also consider rows being built in ctx (not yet committed)
    if ctx:
        for key in ("rows","scan_rows"):
            for m in ctx.get(key, []) or []:
                try:
                    mmax = max(mmax, int(m.get("id",0) or 0))
                except Exception:
                    pass
    return mmax + 1


def _next_sweep_id(session: Session) -> int:
    meas = session["state"]["measurements"]
    smax = 0
    for m in meas:
        s = m.get("sweep")
        try:
            smax = max(smax, int(float(s)))
        except Exception:
            pass
    return smax + 1


def _next_iso_test_id(session: Session, test_name: str) -> int:
    meas = session["state"]["measurements"]
    imax = 0
    for m in meas:
        if m.get("iso_test") == test_name:
            try:
                imax = max(imax, int(m.get("iso_test_id",0) or 0))
            except Exception:
                pass
    return imax + 1


def _next_high_low_test_id(session: Session) -> int:
    meas = session["state"]["measurements"]
    hmax = 0
    for m in meas:
        if m.get("high_low_test") is not None:
            try:
                hmax = max(hmax, int(m.get("high_low_test", 0) or 0))
            except Exception:
                pass
    return hmax + 1
