#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Combination Lock Analysis Suite (CLAS)
#
# An open-source command-line utility for recording, visualizing, and
# analyzing mechanical combination lock measurements for educational and
# locksport purposes.
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

"""
CLAS (Combination Lock Analysis Suite)

Single-file CLI for recording, visualizing, and analyzing mechanical combination-lock measurements.

Navigation guide (search for these headers):
  - Session I/O
  - Input helpers
  - Menus
  - Plotting
  - Measurement workflows
  - Domain workflows
"""

from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from itertools import product


# =============================================================================
# Constants & Types
# =============================================================================

SAVE_EXT = ".json"
CLAS_VERSION = "0.1.0"  # update as needed

Number = float
Session = Dict[str, Any]
LockConfig = Dict[str, Any]


# =============================================================================
# Small console helpers
# =============================================================================

def print_banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def print_section(title: str) -> None:
    print("\n--- " + title + " ---")

def dedent_print(s: str) -> None:
    print(textwrap.dedent(s).strip())


# =============================================================================
# Lock config defaults / normalization (KEEP: safe + general)
# =============================================================================

def default_lock_config() -> LockConfig:
    return {
        "wheels": 3,
        "dial_min": 0,
        "dial_max": 99,
        "tolerance": 1,
        "turn_sequence": "LRL",
        "flies": "fixed",
        "make": "UNKNOWN",
        "fence_type": "UNKNOWN",   # FRICTION_FENCE / GRAVITY_LEVER / SPRING_LEVER / UNKNOWN
        "ul": "UNKNOWN",           # 2 / 2M / 1 / 1R / UNKNOWN
        "oval_wheels": "UNKNOWN",  # YES / NO / UNKNOWN
        "notes": "",
        "awr_low_point": None,
        "awl_low_point": None,
        "approx_lcp_location": None,
        "approx_rcp_location": None,
        "wheel_data": {},
    }

def normalize_lock_config(lock_config: dict) -> LockConfig:
    """
    Ensure lock_config contains all expected fields and per-wheel entries.
    Keeps backward compatibility when schema changes.
    """
    if not isinstance(lock_config, dict):
        lock_config = {}

    defaults = default_lock_config()

    for k, v in defaults.items():
        if k not in lock_config:
            lock_config[k] = v

    if not isinstance(lock_config.get("wheel_data"), dict):
        lock_config["wheel_data"] = {}

    try:
        wheels = int(lock_config.get("wheels") or defaults["wheels"])
    except Exception:
        wheels = defaults["wheels"]
        lock_config["wheels"] = wheels

    for w in range(1, wheels + 1):
        wk = str(w)
        if wk not in lock_config["wheel_data"] or not isinstance(lock_config["wheel_data"][wk], dict):
            lock_config["wheel_data"][wk] = {}

        wd = lock_config["wheel_data"][wk]
        wd.setdefault("known_gates", [])
        wd.setdefault("suspected_gates", [])
        wd.setdefault("false_gates", [])

        for key in ("known_gates", "suspected_gates", "false_gates"):
            if not isinstance(wd.get(key), list):
                wd[key] = []

    return lock_config


# =============================================================================
# Session management
# =============================================================================

def create_new_session() -> Session:
    while True:
        name = input("Enter a name for the new session: ").strip()
        if not name:
            print("Name cannot be empty.")
            continue
        filename = name + SAVE_EXT
        if os.path.exists(filename):
            print("That session already exists. Choose another name.")
            continue
        break

    return {
        "session_name": name,
        "measurements": [],
        "metadata": {},
        "lock_config": normalize_lock_config(default_lock_config()),
    }

def list_sessions() -> List[str]:
    return sorted(f[:-len(SAVE_EXT)] for f in os.listdir(".") if f.endswith(SAVE_EXT))

def load_session(name: str) -> Session:
    with open(name + SAVE_EXT, "r", encoding="utf-8") as f:
        session = json.load(f)

    if "lock_config" not in session:
        session["lock_config"] = {}
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    session.setdefault("measurements", [])
    session.setdefault("metadata", {})
    session.setdefault("session_name", name)

    return session

def save_session(session: Session) -> None:
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    filename = session["session_name"] + SAVE_EXT
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    print(f"Session saved to {filename}")

def has_unsaved_changes(session: Session) -> bool:
    filename = session["session_name"] + SAVE_EXT
    if not os.path.exists(filename):
        return True
    try:
        with open(filename, "r", encoding="utf-8") as f:
            disk = json.load(f)
        return disk != session
    except Exception:
        return True

def next_measurement_id(session: Session) -> int:
    """Centralized ID generator (avoids repeated len()+1 patterns)."""
    meas = session.get("measurements", [])
    if not meas:
        return 1
    # Defensive: handle non-int IDs
    ids = [m.get("id") for m in meas if isinstance(m.get("id"), int)]
    return (max(ids) + 1) if ids else (len(meas) + 1)


# =============================================================================
# Input handling (centralized)
# =============================================================================

def safe_input(prompt: str, session: Session) -> str:
    value = input(prompt).strip()
    if value.upper() == "Q":
        print("\nQuit requested.")
        if has_unsaved_changes(session):
            while True:
                save = input("Unsaved changes detected. Save before quitting? (y/n): ").lower().strip()
                if save in ("y", "n"):
                    break
                print("Please enter 'y' or 'n'.")
            if save == "y":
                save_session(session)
        print("Goodbye.")
        sys.exit(0)
    return value

def ask_yes_no(prompt: str, session: Session) -> bool:
    while True:
        ans = safe_input(prompt, session).strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("Please enter 'y' or 'n'.")

def prompt_int(prompt: str, session: Session, *, default: Optional[int] = None, min_value: Optional[int] = None) -> int:
    while True:
        raw = safe_input(prompt, session)
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Invalid integer.")
            continue
        if min_value is not None and val < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return val

def prompt_float(prompt: str, session: Session, *, default: Optional[float] = None) -> float:
    while True:
        raw = safe_input(prompt, session)
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Invalid number.")

def prompt_optional_float(prompt: str, session: Session, *, current: Optional[float]) -> Optional[float]:
    raw = safe_input(f"{prompt} (current={current}, blank=keep, '-'=clear): ", session).strip()
    if raw == "":
        return current
    if raw == "-":
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number; keeping current.")
        return current

def prompt_choice(prompt: str, session: Session, choices: Sequence[str], *, current: str) -> str:
    """
    Drop-in replacement for prompt_choice().

    Supports:
      - Enter: keep current
      - Number selection: 1..N
      - Typing the option text (case-insensitive)

    Example input:
      2
      gravity_lever
      2m
      UNKNOWN
    """
    # Normalize choices to strings (keep original order)
    choices = [str(c) for c in choices]

    # Normalize current to match style used in choices (usually uppercase)
    current_norm = str(current).strip()

    # Display menu
    print(f"\n{prompt} (current={current_norm})")
    for i, opt in enumerate(choices, 1):
        marker = " *" if opt.upper() == current_norm.upper() else ""
        print(f"  {i}) {opt}{marker}")

    raw = safe_input(f"Select 1-{len(choices)} or type value (Enter = keep): ", session).strip()

    # Enter keeps current
    if raw == "":
        return current_norm

    # Number selection
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
        print("Invalid selection; keeping current.")
        return current_norm

    # Text selection (case-insensitive)
    raw_up = raw.upper()
    for opt in choices:
        if opt.upper() == raw_up:
            return opt

    print("Invalid choice; keeping current.")
    return current_norm

def prompt_csv_floats(prompt: str, session: Session, *, allow_clear: bool = True, current: Optional[List[float]] = None) -> List[float]:
    """
    CSV list parser for gate lists, etc.
    - blank -> keep current (or empty list if no current)
    - '-' -> clear (if allow_clear)
    """
    cur = list(current) if current is not None else []
    raw = safe_input(prompt, session).strip()
    if raw == "":
        return cur
    if allow_clear and raw == "-":
        return []
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        print("Invalid list; keeping current.")
        return cur


# =============================================================================
# UI: splash / session select
# =============================================================================

def show_splash_screen(session: Session) -> None:
    banner = """
====================================================
   Combination Lock Analysis Suite (CLAS)
====================================================
"""
    message = """
This application is intended ONLY for lawful and ethical purposes,
such as education, locksport, and locksmith training.

By using this program, you acknowledge you are solely responsible
for its use and you have permission to work on the device.
"""
    print(banner)
    print(textwrap.dedent(message).strip())
    print("\nDo you agree to use this tool only for lawful and ethical purposes?\n")

    while True:
        choice = safe_input("Type YES to continue, or NO to exit (Q to quit): ", session).upper()
        if choice == "YES":
            return
        if choice == "NO":
            print("\nExiting. Use responsibly.\n")
            sys.exit(0)
        print("Please type YES or NO.")

def select_session() -> Session:
    while True:
        print("\n1) Start new session")
        print("2) Load existing session")

        choice = input("Select option (Q to quit): ").strip()
        if choice.upper() == "Q":
            print("Goodbye.")
            sys.exit(0)

        if choice == "1":
            print("\nStarting new session.")
            return create_new_session()

        if choice == "2":
            sessions = list_sessions()
            if not sessions:
                print("\nNo saved sessions found.")
                continue

            print("\nAvailable sessions:")
            for i, name in enumerate(sessions, 1):
                print(f" {i}) {name}")

            while True:
                sel = input("Select session number (Q to quit): ").strip()
                if sel.upper() == "Q":
                    print("Goodbye.")
                    sys.exit(0)
                if sel.isdigit() and 1 <= int(sel) <= len(sessions):
                    name = sessions[int(sel) - 1]
                    print(f"\nLoaded session: {name}")
                    return load_session(name)
                print("Invalid selection.")

        print("Invalid choice.")


# =============================================================================
# Configuration UI (refactored for readability; logic preserved)
# =============================================================================

def configure_lock(session: Session) -> None:
    print_section("Lock Configuration")

    lock = normalize_lock_config(session.get("lock_config", {}))
    session["lock_config"] = lock

    print("\nCurrent configuration:")
    print(f"  Number of wheels: {lock['wheels']}")
    print(f"  Dial min value: {lock['dial_min']}")
    print(f"  Dial max value: {lock['dial_max']}")
    print(f"  Lock tolerance: ±{lock['tolerance']}")
    print(f"  Turn sequence: {lock['turn_sequence']}")
    print(f"  Flies: {lock['flies']}")
    print(f"  Make: {lock.get('make', 'UNKNOWN')}")
    print(f"  Fence type: {lock.get('fence_type', 'UNKNOWN')}")
    print(f"  UL rating: {lock.get('ul', 'UNKNOWN')}")
    print(f"  Oval wheels: {lock.get('oval_wheels', 'UNKNOWN')}")
    print(f"  AWR low point: {lock.get('awr_low_point')}")
    print(f"  AWL low point: {lock.get('awl_low_point')}")
    print(f"  Approx LCP location: {lock.get('approx_lcp_location')}")
    print(f"  Approx RCP location: {lock.get('approx_rcp_location')}")
    print(f"  Notes: {lock['notes']}")

    print("\nWheel gate data:")
    for w in range(1, int(lock["wheels"]) + 1):
        wd = lock["wheel_data"].get(str(w), {})
        print(
            f"  Wheel {w}: "
            f"known={wd.get('known_gates', [])} | "
            f"suspected={wd.get('suspected_gates', [])} | "
            f"false={wd.get('false_gates', [])}"
        )

    if not ask_yes_no("\nDo you want to change the configuration? (y/n): ", session):
        return

    wheels = prompt_int(f"Number of wheels (current={lock['wheels']}): ", session, default=int(lock["wheels"]), min_value=1)
    lock["wheels"] = wheels
    lock = normalize_lock_config(lock)  # ensure wheel buckets exist after wheel count changes

    lock["dial_min"] = prompt_float(f"Dial minimum value (current={lock['dial_min']}): ", session, default=float(lock["dial_min"]))

    while True:
        lock["dial_max"] = prompt_float(f"Dial maximum value (current={lock['dial_max']}): ", session, default=float(lock["dial_max"]))
        if lock["dial_max"] > lock["dial_min"]:
            break
        print("Dial maximum must be greater than dial minimum.")

    lock["tolerance"] = prompt_float(f"Lock tolerance (current={lock['tolerance']}): ", session, default=float(lock["tolerance"]))
    lock["turn_sequence"] = prompt_choice("Turn sequence", session, ["LRL", "RLR"], current=str(lock["turn_sequence"]).upper())
    flies = prompt_choice("Wheel flies", session, ["FIXED", "MOVEABLE"], current=str(lock["flies"]).upper())
    lock["flies"] = flies.lower()
    make = safe_input(f"Lock make (current={lock.get('make', 'UNKNOWN')}): ", session)
    if make != "":
        lock["make"] = make

    lock["fence_type"] = prompt_choice(
        "Fence type",
        session,
        ["FRICTION_FENCE", "GRAVITY_LEVER", "SPRING_LEVER", "UNKNOWN"],
        current=str(lock.get("fence_type", "UNKNOWN")).upper(),
    )

    lock["ul"] = prompt_choice(
        "UL rating",
        session,
        ["2", "2M", "1", "1R", "UNKNOWN"],
        current=str(lock.get("ul", "UNKNOWN")).upper(),
    )

    lock["oval_wheels"] = prompt_choice(
        "Oval wheels",
        session,
        ["NO", "YES", "UNKNOWN"],
        current=str(lock.get("oval_wheels", "UNKNOWN")).upper(),
    )

    lock["awr_low_point"] = prompt_optional_float("All Wheels Right (AWR) low point", session, current=lock.get("awr_low_point"))
    lock["awl_low_point"] = prompt_optional_float("All Wheels Left (AWL) low point", session, current=lock.get("awl_low_point"))
    lock["approx_lcp_location"] = prompt_optional_float("Approximate Left Contact Point (LCP) location", session, current=lock.get("approx_lcp_location"))
    lock["approx_rcp_location"] = prompt_optional_float("Approximate Right Contact Point (RCP) location", session, current=lock.get("approx_rcp_location"))

    print("\n--- Per-wheel gate data ---")
    for w in range(1, wheels + 1):
        wd = lock["wheel_data"].get(str(w), {})
        wd["known_gates"] = prompt_csv_floats(
            f"Wheel {w} KNOWN gates (current={wd.get('known_gates', [])}, blank=keep, '-'=clear): ",
            session,
            current=wd.get("known_gates", []),
        )
        wd["suspected_gates"] = prompt_csv_floats(
            f"Wheel {w} SUSPECTED gates (current={wd.get('suspected_gates', [])}, blank=keep, '-'=clear): ",
            session,
            current=wd.get("suspected_gates", []),
        )
        wd["false_gates"] = prompt_csv_floats(
            f"Wheel {w} FALSE gates (current={wd.get('false_gates', [])}, blank=keep, '-'=clear): ",
            session,
            current=wd.get("false_gates", []),
        )
        lock["wheel_data"][str(w)] = wd

    notes = safe_input(f"Lock notes (current={lock['notes']}): ", session)
    if notes != "":
        lock["notes"] = notes

    session["lock_config"] = lock
    print("\nConfiguration updated.")


# =============================================================================
# Measurement utilities (small, general)
# =============================================================================

def _prompt_for_current_combination(session: Session) -> List[float]:
    lock = session["lock_config"]
    wheels = int(lock["wheels"])
    while True:
        combo_input = safe_input(
            f"Enter current combination as comma-separated numbers (length={wheels}): ",
            session,
        )
        parts = combo_input.replace(" ", "").split(",")
        if len(parts) != wheels:
            print(f"Please enter exactly {wheels} numbers.")
            continue
        try:
            return [float(p) for p in parts]
        except ValueError:
            print("All numbers must be numeric.")

def _combination_to_dict(comb: Sequence[float]) -> Dict[str, float]:
    return {f"combination_wheel_{i+1}": float(comb[i]) for i in range(len(comb))}


# =============================================================================
# Plot selectors (refactored, including a common “ID parsing” approach)
# =============================================================================

def _measurement_int_field_values(session: Session, field: str) -> List[int]:
    vals: List[int] = []
    for m in session.get("measurements", []):
        v = m.get(field)
        if isinstance(v, int):
            vals.append(v)
        elif isinstance(v, float) and v.is_integer():
            vals.append(int(v))
        elif isinstance(v, str) and v.strip().replace(".", "", 1).isdigit():
            f = float(v)
            if f.is_integer():
                vals.append(int(f))
    return sorted(set(vals))

def plot_sweep_selector(session: Session) -> None:
    sweeps = _measurement_int_field_values(session, "sweep")
    if not sweeps:
        print("No sweep data available.")
        return

    print("\nAvailable sweeps:")
    for s in sweeps:
        print(f"  Sweep {s}")

    while True:
        val = safe_input("Enter sweep number to plot (Q to cancel): ", session)
        if val.upper() == "Q":
            return
        try:
            sweep_id = int(float(val))
        except ValueError:
            print("Enter a valid sweep number.")
            continue
        if sweep_id not in sweeps:
            print("That sweep does not exist.")
            continue

        plot_sweep(session, sweep_id)
        return

def plot_hig_low_test_selector(session: Session) -> None:
    test_ids = _measurement_int_field_values(session, "high_low_test")
    if not test_ids:
        print("No High Low Tests recorded in this session.")
        return

    print("\nAvailable High Low Tests:")
    for tid in test_ids:
        print(f"  Test ID {tid}")

    while True:
        val = safe_input("Enter High Low Test ID to plot (Q to cancel): ", session)
        if val.upper() == "Q":
            return
        try:
            test_id = int(float(val))
        except ValueError:
            print("Enter a valid numeric Test ID.")
            continue
        if test_id not in test_ids:
            print("That Test ID does not exist.")
            continue

        plot_high_low_test(session, test_id)
        return


# =============================================================================
# Analyze menu (refactored to dispatch)
# =============================================================================

def analyze_session(session: Session) -> None:
    actions: Dict[str, Callable[[Session], None]] = {
        "1": lambda s: plot_sweep_selector(s),
        "2": lambda s: plot_hig_low_test_selector(s),
        "3": lambda s: None,
    }

    while True:
        print_section("Analyze Session")
        print("1) Plot a sweep")
        print("2) Plot High Low Test")
        print("3) Return to main menu")

        choice = safe_input("Select option: ", session).strip()
        if choice == "3":
            return
        action = actions.get(choice)
        if not action:
            print("Invalid selection.")
            continue
        action(session)


# =============================================================================
# Export (safe/general)
# =============================================================================

def export_to_csv(session: Session) -> None:
    measurements = session.get("measurements", [])
    if not measurements:
        print("No measurements to export.")
        return

    all_keys = set()
    for m in measurements:
        all_keys.update(m.keys())

    fieldnames = ["id"] + sorted(k for k in all_keys if k != "id")
    default_filename = session["session_name"] + ".csv"

    filename = safe_input(f"Enter CSV filename (default = {default_filename}): ", session).strip()
    if filename == "":
        filename = default_filename
    if not filename.endswith(".csv"):
        filename += ".csv"

    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for m in measurements:
                row = {k: m.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print(f"Measurements exported to {filename}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")


# =============================================================================
# Help menu (refactored)
# =============================================================================

def show_about(session: Session) -> None:
    print_section("About")
    dedent_print(f"""
    Combination Lock Analysis Suite (CLAS)

    Version:
      {CLAS_VERSION}

    License:
      GNU General Public License v3.0 (GPL-3.0-only)

    Author:
      knowthebird

    Description:
      CLAS is an open-source command-line utility for recording,
      organizing, and visualizing mechanical combination lock measurements.
      It is intended for educational use, locksport practice,
      and locksmith training on locks you own or have permission to open.

    Disclaimer:
      This software is provided without warranty of any kind.
      Users are responsible for ensuring lawful and ethical use.
    """)
    safe_input("\nPress Enter to return to Help menu: ", session)

def show_resources(session: Session) -> None:
    print_section("Resources")
    dedent_print("""
    Books, Papers, and Tools:

      • Sophie Houlden’s Safecracking Simulator
        https://sophieh.itch.io/sophies-safecracking-simulator

      • Safecracking for Everyone (2nd Edition)
        Jared Dygert
        https://drive.google.com/file/d/1xqfTAq-NY6-hXiPB0u44vdNjeMXbHJEz/view

      • Safe Lock Manipulation 101
        Jan-Willem Markus
        https://blackbag.toool.nl/wp-content/uploads/2024/02/Safe-manipulation-101-v2.pdf

      • Safecracking 101: A Beginner’s Guide to Safe Manipulation and Drilling
        Thomas A. Mazzone & Thomas G. Seroogy
        https://dn720001.ca.archive.org/0/items/safecracking-101-a-beginners-guide-to-safe-manipulation-and-drilling/Safecracking%20101%20-%20A%20Beginners%20Guide%20to%20Safe%20Manipulation%20and%20Drilling%20-%20Thomas%20Mazzone%20-%202013.pdf

    (These resources are provided for educational reference only.)
    """)
    safe_input("\nPress Enter to return to Help menu: ", session)

def help_menu(session: Session) -> None:
    while True:
        print_section("Help")
        print("1) Tutorial")
        print("2) About")
        print("3) Resources")
        print("4) Return to main menu")

        choice = safe_input("Select option: ", session).strip()
        if choice == "1":
            show_tutorial(session)
        elif choice == "2":
            show_about(session)
        elif choice == "3":
            show_resources(session)
        elif choice == "4":
            return
        else:
            print("Invalid selection.")


# =============================================================================
# Test menu (dispatch-based)
# =============================================================================

def test_menu(session: Session) -> None:
    """
    IMPORTANT:
    - Leave your existing domain functions unchanged.
    - This menu just dispatches to them more cleanly.
    """
    actions: Dict[str, Callable[[Session], None]] = {
        "1": measure_contact_points_single,
        "2": measure_contact_points_sweep,
        "3": isolate_wheel_3,
        "4": isolate_wheel_2,
        "5": run_high_low_test,
        "6": find_awr_low_point,
        "7": find_awl_low_point,
        "8": exhaustive_enumeration_all,
        "9": _exhaustive_enumeration_wheel_prompt,
        "0": lambda s: None,
    }

    while True:
        print_section("Test Menu")
        print("1) Measure contact points for single combination")
        print("2) Measure contact points for sweep")
        print("3) Isolate Wheel 3")
        print("4) Isolate Wheel 2")
        print("5) High - Low Test")
        print("6) Find AWR low point")
        print("7) Find AWL low point")
        print("8) Exhaustive Enumeration (All Wheels)")
        print("9) Exhaustive Enumeration (One Wheel)")
        print("0) Return to main menu")

        choice = safe_input("Select option: ", session).strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if not action:
            print("Invalid selection.")
            continue
        action(session)

def _exhaustive_enumeration_wheel_prompt(session: Session) -> None:
    val = safe_input("Which wheel to use for exhaustive enumeration? (1-3): ", session).strip()
    if val.isdigit() and 1 <= int(val) <= 3:
        exhaustive_enumeration_wheel(session, int(val))
    else:
        print("Invalid wheel number.")


# =============================================================================
# Main menu (dispatch-based)
# =============================================================================

def main() -> None:
    temp = {"session_name": "__temp__", "measurements": [], "metadata": {}, "lock_config": {}}
    show_splash_screen(temp)

    session = select_session()
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    print(f"\nSession ready: {session['session_name']}")

    main_actions: Dict[str, Callable[[Session], None]] = {
        "h": help_menu,
        "c": configure_lock,
        "t": test_menu,
        "a": analyze_session,
        "e": export_to_csv,
        "s": save_session,
        "q": lambda s: safe_input("Press Q to quit: ", s),  # triggers safe_input quit handling
    }

    while True:
        print("\nMain Menu:")
        print("H) Help")
        print("C) Configure")
        print("T) Test")
        print("A) Analyze")
        print("E) Export")
        print("S) Save")
        print("Q) Quit")

        cmd = safe_input("Enter command: ", session).lower().strip()
        action = main_actions.get(cmd)
        if not action:
            print("Unknown command. Please select H, C, T, A, E, S, or Q.")
            continue
        action(session)


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_sweep(session: Session, sweep_id: int) -> None:
    measurements = [m for m in session["measurements"] if m.get("sweep") == sweep_id]

    if not measurements:
        print("No data for this sweep.")
        return

    wheel = measurements[0]["wheel_swept"]
    wheel_key = f"combination_wheel_{wheel}"

    x = [m[wheel_key] for m in measurements]
    lcp = [m["left_contact"] for m in measurements]
    rcp = [m["right_contact"] for m in measurements]
    width = [r - l for r, l in zip(rcp, lcp)]

    data = sorted(zip(x, lcp, rcp, width))
    x, lcp, rcp, width = map(np.array, zip(*data))

    min_rcp_idx = np.argmin(rcp)
    min_width_idx = np.argmin(width)
    max_lcp_idx = np.argmax(lcp)

    def detect_gates(x, width):
        gates = []
        for i in range(1, len(width) - 1):
            if width[i] < width[i-1] and width[i] < width[i+1]:
                depth = ((width[i-1] + width[i+1]) / 2) - width[i]
                if depth > 0.3:
                    gates.append((x[i], width[i], depth))
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

    ax_mid.plot(x, lcp, marker="o")
    ax_mid.scatter(x, lcp, marker="x")
    ax_mid.scatter(x[max_lcp_idx], lcp[max_lcp_idx], color="red", s=120)
    ax_mid.set_ylabel("Left Contact")
    ax_mid.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_mid.minorticks_on()
    ax_mid.grid(True, which="minor", linestyle=":", alpha=0.3)

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

    # ---- Adaptive X-axis ticks for readability ----
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = xmax - xmin

    # Choose a reasonable major tick spacing
    if span <= 20:
        major_step = 1
    elif span <= 50:
        major_step = 2
    else:
        major_step = 5

    major_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, major_step)
    minor_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, major_step / 5)

    ax_bot.set_xticks(major_ticks)
    ax_bot.set_xticks(minor_ticks, minor=True)

    # Rotate labels slightly so they don’t collide
    for label in ax_bot.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


    wheels = session["lock_config"]["wheels"]
    combo = [measurements[0][f"combination_wheel_{i+1}"] for i in range(wheels)]

    fig.suptitle(
        f"Session: {session['session_name']} | Sweep: {sweep_id}\n"
        f"Wheel swept: {wheel}   |   Combination: {combo}",
        fontsize=12
    )

    plt.tight_layout()
    filename = f"{session['session_name']}_sweep_{sweep_id}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


def plot_high_low_test(session: Session, test_id: int) -> None:

    data = [m for m in session["measurements"]
            if str(m.get("high_low_test", "")) == str(test_id)]

    if not data:
        print("No data for that test.")
        return

    high = [m for m in data if str(m.get("hw_type", "")).strip().lower() == "high"]
    low  = [m for m in data if str(m.get("hw_type", "")).strip().lower() == "low"]

    def combo_label(m):
        c1 = m.get("combination_wheel_1", "")
        c2 = m.get("combination_wheel_2", "")
        c3 = m.get("combination_wheel_3", "")
        return f"{c1}, {c2}, {c3}"

    def extract(meas):
        if not meas:
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        l = np.array([m["left_contact"] for m in meas], dtype=float)
        r = np.array([m["right_contact"] for m in meas], dtype=float)
        w = np.array([m["contact_width"] for m in meas], dtype=float)
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

    # RCP: highlight unique max only
    draw(axes[0, 0], xh, rh, "RCP (+)")
    draw(axes[0, 1], xl, rl, "RCP (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[0, 0], xh, rh, mode="max")
    if len(xl) > 0:
        highlight_if_unique(axes[0, 1], xl, rl, mode="max")
    apply_combo_ticks(axes[0, 0], xh, lab_h)
    apply_combo_ticks(axes[0, 1], xl, lab_l)

    # LCP: highlight unique min only
    draw(axes[1, 0], xh, lh, "LCP (+)")
    draw(axes[1, 1], xl, ll, "LCP (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[1, 0], xh, lh, mode="min")
    if len(xl) > 0:
        highlight_if_unique(axes[1, 1], xl, ll, mode="min")
    apply_combo_ticks(axes[1, 0], xh, lab_h)
    apply_combo_ticks(axes[1, 1], xl, lab_l)

    # Width: highlight unique max only
    draw(axes[2, 0], xh, wh, "Width (+)")
    draw(axes[2, 1], xl, wl, "Width (−)")
    if len(xh) > 0:
        highlight_if_unique(axes[2, 0], xh, wh, mode="max")
    if len(xl) > 0:
        highlight_if_unique(axes[2, 1], xl, wl, mode="max")
    apply_combo_ticks(axes[2, 0], xh, lab_h)
    apply_combo_ticks(axes[2, 1], xl, lab_l)

    gate = data[0].get("hw_gate", "")
    inc  = data[0].get("hw_increment", "")

    fig.suptitle(
        f"Session: {session['session_name']}   High-Wheel Test {test_id}\n"
        f"Gate = {gate}   Increment = ±{inc}",
        fontsize=12
    )

    plt.tight_layout()
    filename = f"{session['session_name']}_high_low_test_{test_id}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")


# =============================================================================
# Measurement Workflows
# =============================================================================
def measure_contact_points_single(session: Session) -> None:
    print("\n--- Measure Contact Points (Single Combination) ---")
    combination_values = _prompt_for_current_combination(session)

    left = _get_valid_number("Enter left contact", 0.0, session)
    right = _get_valid_number("Enter right contact", 0.0, session)
    width = right - left
    notes = safe_input("Notes (optional): ", session)

    measurement = {
        "id": next_measurement_id(session),
        **_combination_to_dict(combination_values),
        "left_contact": left,
        "right_contact": right,
        "contact_width": width,
        "sweep": None,
        "notes": notes
    }
    session["measurements"].append(measurement)
    print("Single measurement added.")

def measure_contact_points_sweep(session: Session) -> None:
    print("\n--- Measure Contact Points (Sweep) ---")
    combination_values = _prompt_for_current_combination(session)

    existing_sweeps = [m["sweep"] for m in session["measurements"] if m.get("sweep") is not None]
    sweep_id = max(existing_sweeps, default=0) + 1
    print(f"Sweep number: {sweep_id}")
    print("During the sweep, you can type Q at any prompt to quit or save your session.")

    while True:
        wheel_num = safe_input(f"Which wheel are you sweeping? (1-{session['lock_config']['wheels']}): ", session)
        if wheel_num.isdigit() and 1 <= int(wheel_num) <= session["lock_config"]["wheels"]:
            wheel_num = int(wheel_num)
            break
        print("Invalid wheel number.")

    dial_min = session["lock_config"]["dial_min"]
    dial_max = session["lock_config"]["dial_max"]
    range_size = dial_max - dial_min + 1

    start_dial = combination_values[wheel_num - 1]
    print(f"Default sweep start = {start_dial} (current wheel value)")

    tolerance = session["lock_config"].get("tolerance", 1)
    default_increment = tolerance * 2
    print(f"\nDefault sweep increment = {default_increment} for lock tolerance = ±{tolerance}")
    increment = _get_valid_number("Increment to move each time", default_increment, session)
    if increment <= 0:
        print("Increment must be positive. Using default.")
        increment = default_increment

    num_steps = int(round(range_size / increment))
    if num_steps == 0:
        num_steps = 1

    dial = start_dial
    step_count = 0

    for _ in range(num_steps):
        print(f"\nWheel {wheel_num} | Dial position: {dial} (type Q at any prompt to quit)")

        while True:
            try:
                left = float(safe_input("Enter left contact: ", session))
                break
            except ValueError:
                print("Invalid number. Enter a valid float.")

        while True:
            try:
                right = float(safe_input("Enter right contact: ", session))
                break
            except ValueError:
                print("Invalid number. Enter a valid float.")

        width = right - left
        notes = safe_input("Notes (optional): ", session)

        current_combination = combination_values.copy()
        current_combination[wheel_num - 1] = dial

        measurement = {
            "id": next_measurement_id(session),
            **_combination_to_dict(current_combination),
            "wheel_swept": wheel_num,
            "left_contact": left,
            "right_contact": right,
            "contact_width": width,
            "sweep": sweep_id,
            "notes": notes
        }

        session["measurements"].append(measurement)
        step_count += 1
        dial = (dial + increment - dial_min) % range_size + dial_min

    print(f"Sweep {sweep_id} complete. {step_count} entries added.")

    if ask_yes_no("Do you want to plot this sweep? (y/n): ", session):
        plot_sweep(session, sweep_id)


# =============================================================================
# Domain Workflows
# =============================================================================

def _get_valid_number(prompt, default, session):
    while True:
        val = safe_input(f"{prompt} (default = {default}): ", session)
        if val == "":
            return default
        try:
            return float(val)
        except ValueError:
            print("Invalid number. Try again.")

def _effective_gate_list(wheel_data: dict):
    """Known gates override suspected gates."""
    known = wheel_data.get("known_gates", []) or []
    suspected = wheel_data.get("suspected_gates", []) or []
    return known if known else suspected


def _record_exhaustive_enumeration_attempt(session, combo, test_type, result):
    m = {
        "id": next_measurement_id(session),
        "combination_wheel_1": combo[0],
        "combination_wheel_2": combo[1],
        "combination_wheel_3": combo[2],
        "left_contact": None,
        "right_contact": None,
        "contact_width": None,
        "test_type": test_type,
        "test_result": result,
        "notes": ""
    }
    session["measurements"].append(m)


def _combination_already_tested(session, combo):
    for m in session["measurements"]:
        if m.get("test_type", "").startswith("exhaustive_enumeration"):
            if (
                m.get("combination_wheel_1") == combo[0]
                and m.get("combination_wheel_2") == combo[1]
                and m.get("combination_wheel_3") == combo[2]
            ):
                return True
    return False


def _handle_exhaustive_enumeration_success(session, combo, test_type):
    _record_exhaustive_enumeration_attempt(session, combo, test_type, "verified")

    lc = session["lock_config"]
    wd = lc["wheel_data"]

    for i, val in enumerate(combo, start=1):
        w = str(i)
        wd[w].setdefault("known_gates", [])
        if val not in wd[w]["known_gates"]:
            wd[w]["known_gates"].append(val)
        if val in wd[w].get("suspected_gates", []):
            wd[w]["suspected_gates"].remove(val)

    print("\nCOMBINATION VERIFIED")
    print("Known gates updated from successful combination.")

def _edit_gate_estimates(initial: list, session, wheel_num: int):
    """
    Allow user to review and edit gate estimates before saving.
    Returns final list (may be unchanged or empty).
    """
    current = list(initial)

    print(f"\nCurrent candidate gate estimates for Wheel {wheel_num}: {current}")
    print("Enter a revised comma-separated list to replace them,")
    print("press Enter to keep them as-is, or '-' to clear.\n")

    raw = safe_input("Revised gate list: ", session).strip()
    if raw == "":
        return current
    if raw == "-":
        return []

    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        print("Invalid input. Keeping original values.")
        return current


def _count_remaining_enumerations(session, combos):
    remaining = 0
    for combo in combos:
        if not _combination_already_tested(session, combo):
            remaining += 1
    return remaining

def _wrap_dial(value: float, dial_min: float, dial_max: float) -> float:
    """Wrap a dial value into [dial_min, dial_max] inclusive, assuming a circular dial."""
    range_size = (dial_max - dial_min) + 1
    return ((value - dial_min) % range_size) + dial_min


def _build_checkpoints(dial_min: float, dial_max: float, *, count: int = 10) -> list[float]:
    start = float(dial_min)
    dial_min = float(dial_min)
    dial_max = float(dial_max)

    decades: list[float] = []
    top_decade = int(dial_max // 10) * 10
    p = top_decade
    while len(decades) < (count - 1) and p > dial_min:
        if float(p) != start:
            decades.append(float(p))
        p -= 10

    checkpoints: list[float] = [start] + decades

    if len(checkpoints) < count:
        needed = count - len(checkpoints)
        span = dial_max - dial_min
        step = span / (needed + 1) if span != 0 else 1.0
        for i in range(1, needed + 1):
            candidate = dial_max - i * step
            if abs(candidate - start) > 1e-9:
                checkpoints.append(float(candidate))

    return checkpoints[:count]


def run_high_low_test(session: Session) -> None:
    print("\n--- High Low Test ---")

    existing = [m["high_low_test"] for m in session["measurements"]
                if m.get("high_low_test") is not None]
    test_id = max(existing, default=0) + 1

    while True:
        try:
            gate = float(safe_input("Enter suspected true gate: ", session))
            break
        except ValueError:
            print("Invalid number.")

    default_inc = 10
    while True:
        val = safe_input(f"Enter increment (default {default_inc}): ", session)
        if val == "":
            increment = default_inc
            break
        try:
            increment = float(val)
            if increment > 0:
                break
        except ValueError:
            pass
        print("Increment must be a positive number.")

    wheels = session["lock_config"]["wheels"]
    if wheels != 3:
        print("High Low Test currently supports 3-wheel locks only.")
        return

    turn = session["lock_config"]["turn_sequence"]

    def run_test(hw_type, offset):
        print(f"\n--- {hw_type.upper()} TEST  (offset = {offset:+}) ---")

        if turn == "LRL":
            tests = [
                [gate + offset, gate, gate],
                [gate, gate + offset, gate],
                [gate, gate, gate + offset]
            ]
        else:
            tests = [
                [gate + offset, gate, gate],
                [gate, gate + offset, gate],
                [gate, gate, gate + offset]
            ]

        for i, combo in enumerate(tests, 1):
            print(f"\nCombination {i}: {combo}")
            print(f"Turn left (CCW) passing {combo[0]} three times, and continue turning until you hit {combo[0]}, then stop.")
            print(f"Turn right (CW) passing {combo[0]} two times, and continue turning until you hit {combo[1]}, then stop.")
            print(f"Turn left (CCW) passing {combo[1]} one time, and continue turning until you hit {combo[2]}, then stop.")
            while True:
                try:
                    l = float(safe_input("  Left contact: ", session))
                    r = float(safe_input("  Right contact: ", session))
                    break
                except ValueError:
                    print("Invalid number.")

            m = {
                "id": next_measurement_id(session),
                "combination_wheel_1": combo[0],
                "combination_wheel_2": combo[1],
                "combination_wheel_3": combo[2],
                "left_contact": l,
                "right_contact": r,
                "contact_width": r - l,
                "sweep": None,
                "high_low_test": test_id,
                "hw_gate": gate,
                "hw_increment": increment,
                "hw_type": hw_type,
                "notes": ""
            }
            session["measurements"].append(m)

    run_test("high", increment)

    if ask_yes_no("\nPerform Low Test (−increment)? (y/n): ", session):
        run_test("low", -increment)

    if ask_yes_no("\nAnalyze results? (y/n): ", session):
        plot_high_low_test(session, test_id)

    print(f"\nHigh Low Test {test_id} recorded.")


def find_awr_low_point(session: Session) -> None:
    """
    Guided workflow (3-wheel LRL) to identify the All Wheels Right (AWR) wheel stack low point.

    Process (conceptually):
      - Reset: Turn right (CW) 3 full turns to start position
      - At each checkpoint: Turn left (CCW) until you hit RCP; record RCP
      - The smallest recorded RCP is saved as lock_config['awr_low_point']

    Notes:
      - Uses 10 checkpoints. By default: [dial_min] + nine descending decade points (e.g., 90..10 on a 0-99 dial).
      - User can type 'E' at any input prompt in this workflow to exit back to the Test menu.
      - 'Q' always quits the program (handled by safe_input).
    """
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]

    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        print("\n--- Find AWR Low Point ---")
        print("This guided AWR workflow is only implemented for 3-wheel LRL locks at this time.")
        print("You can still enter AWR manually in Configure (C).")
        return

    dial_min = float(lc.get("dial_min", 0))
    dial_max = float(lc.get("dial_max", 99))

    print("\n--- Find AWR Low Point (Guided) ---")
    print("You will take 10 RCP readings at specific dial checkpoints.")
    print("Type 'E' to exit this workflow and return to the Test menu.\n")

    checkpoints = _build_checkpoints(dial_min, dial_max)

    # Display checkpoints so the user knows what to expect
    print("Checkpoints (right (CW) to value, then left (CCW) to RCP):")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i:>2}) {cp}")

    readings = []  # list of dicts: {"checkpoint": cp, "rcp": val}

    # Step 0 instructions
    start_pos = checkpoints[0]
    print("\nRESET / SETUP")
    print(f"1) Turn right (CW) at least 3 full turns, ending exactly on {start_pos}.")
    _ = safe_input("Press Enter when ready (or Q to quit): ", session)

    for idx, cp in enumerate(checkpoints):
        print("\n" + "-" * 60)
        if idx == 0:
            print(f"Checkpoint {idx+1}/{len(checkpoints)}:")
            print("2) Now turn left (CCW) until you hit the Right Contact Point (RCP).")
        else:
            print(f"Checkpoint {idx+1}/{len(checkpoints)}:")
            print(f"1) Turn right (CW) to {cp}.")
            print("2) Now turn left (CCW) until you hit the Right Contact Point (RCP).")

        while True:
            raw = safe_input("Enter RCP value (or E to exit): ", session).strip()
            if raw.lower() == "e":
                print("\nExiting AWR workflow (no changes saved).")
                return
            try:
                rcp_val = float(raw)
                readings.append({"checkpoint": cp, "rcp": rcp_val})
                break
            except ValueError:
                print("Please enter a valid number for RCP, or 'E' to exit.")

    # Determine checkpoint with smallest RCP
    min_item = min(readings, key=lambda d: d["rcp"])
    awr_low_point = float(min_item["checkpoint"])

    print("\nRESULTS")
    print("RCP readings:")
    for r in readings:
        print(f"  checkpoint {r['checkpoint']}: RCP = {r['rcp']}")

    print(
        f"\nLowest RCP observed at checkpoint {min_item['checkpoint']} "
        f"(RCP = {min_item['rcp']})"
    )

    if ask_yes_no("Save this checkpoint as the AWR low point in configuration? (y/n): ", session):
        lc["awr_low_point"] = awr_low_point
        session["lock_config"] = lc
        print(f"Saved: AWR low point = {awr_low_point}")
    else:
        print("Not saved.")

def find_awl_low_point(session: Session) -> None:
    """
    Guided workflow (3-wheel LRL) to identify the All Wheels Left (AWL) wheel stack low point.

    User provides LCP readings at 10 checkpoints.
    We save the CHECKPOINT (dial position) corresponding to the HIGHEST LCP as lock_config['awl_low_point'].

    Type 'E' to exit back to the Test menu.
    'Q' always quits program (safe_input).
    """
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]

    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        print("\n--- Find AWL Low Point ---")
        print("This guided AWL workflow is only implemented for 3-wheel LRL locks at this time.")
        print("You can still enter AWL manually in Configure (C).")
        return

    dial_min = float(lc.get("dial_min", 0))
    dial_max = float(lc.get("dial_max", 99))

    print("\n--- Find AWL Low Point (Guided) ---")
    print("You will take 10 LCP readings at specific dial checkpoints.")
    print("Type 'E' to exit this workflow and return to the Test menu.\n")

    checkpoints = _build_checkpoints(dial_min, dial_max)

    print("Checkpoints (navigate to value, then record LCP):")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i:>2}) {cp}")

    readings = []  # {"checkpoint": cp, "lcp": val}

    for idx, cp in enumerate(checkpoints, 1):
        print("\n" + "-" * 60)
        print(f"Checkpoint {idx}/{len(checkpoints)}: {cp}")
        while True:
            raw = safe_input("Enter LCP value (or E to exit): ", session).strip()
            if raw.lower() == "e":
                print("\nExiting AWL workflow (no changes saved).")
                return
            try:
                lcp_val = float(raw)
                readings.append({"checkpoint": cp, "lcp": lcp_val})
                break
            except ValueError:
                print("Please enter a valid number for LCP, or 'E' to exit.")

    # AWL low point = checkpoint where LCP is HIGHEST
    best = max(readings, key=lambda d: d["lcp"])
    awl_low_point = float(best["checkpoint"])

    print("\nRESULTS")
    print("LCP readings:")
    for r in readings:
        print(f"  checkpoint {r['checkpoint']}: LCP = {r['lcp']}")

    print(f"\nHighest LCP observed at checkpoint {best['checkpoint']} (LCP = {best['lcp']})")

    if ask_yes_no("Save this checkpoint as the AWL low point in configuration? (y/n): ", session):
        lc["awl_low_point"] = awl_low_point
        session["lock_config"] = lc
        print(f"Saved: AWL low point = {awl_low_point}")
    else:
        print("Not saved.")

def isolate_wheel_3(session: Session) -> None:
    """
    Isolate Wheel 3 workflow (data-entry + plotting via existing plot_sweep()).

    Corrections applied:
      - Scan increment uses +/- tolerance => step = tolerance * 2
      - Refinement points (gate-1, gate, gate+1) are added to the SAME sweep as the scan
    """
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]

    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        print("\n--- Isolate Wheel 3 ---")
        print("This isolate-wheel-3 workflow is only implemented for 3-wheel LRL locks at this time.")
        return

    if lc.get("awr_low_point") is None:
        print("\n--- Isolate Wheel 3 ---")
        print("AWR low point is not set.")
        if ask_yes_no("Run 'Find AWR low point' now? (y/n): ", session):
            find_awr_low_point(session)
            session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
            lc = session["lock_config"]
        if lc.get("awr_low_point") is None:
            print("AWR low point still not set. Aborting isolate wheel 3.")
            return

    dial_min = float(lc.get("dial_min", 0))
    dial_max = float(lc.get("dial_max", 99))
    tol = float(lc.get("tolerance", 1))
    if tol <= 0:
        tol = 1.0

    # Scan step should reflect +/- tolerance
    scan_step = tol * 2

    awr = float(lc["awr_low_point"])

    # New iso_test_id (separate from sweep ids)
    existing_ids = [
        int(m.get("iso_test_id"))
        for m in session.get("measurements", [])
        if m.get("iso_test") == "isolate_wheel_3" and str(m.get("iso_test_id", "")).isdigit()
    ]
    iso_test_id = (max(existing_ids) + 1) if existing_ids else 1

    # New sweep id (single sweep for scan + refine)
    existing_sweeps = []
    for m in session.get("measurements", []):
        s = m.get("sweep")
        if isinstance(s, (int, float)):
            existing_sweeps.append(int(s))
        elif isinstance(s, str) and s.replace(".", "", 1).isdigit():
            existing_sweeps.append(int(float(s)))
    sweep_id = (max(existing_sweeps) + 1) if existing_sweeps else 1

    print("\n--- Isolate Wheel 3 (Data Collection) ---")
    print("This workflow will ask you to enter observed LCP/RCP measurements at generated dial positions.")
    print("Type 'E' at a prompt to exit back to the Test menu.")
    print(f"\nUsing: AWR low point = {awr}, tolerance = ±{tol}, scan step = {scan_step}, dial range = {dial_min}..{dial_max}")

    # -------- Phase 1: Scan (full revolution in steps of scan_step) --------
    range_size = (dial_max - dial_min) + 1
    n_steps = int(round(range_size / scan_step))
    if n_steps < 1:
        n_steps = 1

    scan_points = []
    cur = awr+scan_step
    seen = set()
    for _ in range(n_steps + 2):
        key = round(cur, 6)
        if key in seen and len(scan_points) > 0:
            break
        seen.add(key)
        scan_points.append(cur)
        cur = _wrap_dial(cur + scan_step, dial_min, dial_max)

    print(f"\nSCAN PHASE: {len(scan_points)} points (step = {scan_step})")
    # print("At each point, enter the LCP and RCP you observed.\n")

    print("\nStart at 0, turn right (CW), 3 full turns, ending on 0 again.")
    print(f"Continue to turn right (CW), ending on the AWR Wheel Stack Low Point ({awr}).")
    

    scan_rows = []
    for i, p in enumerate(scan_points, 1):
        print(f"\nPoint {i}/{len(scan_points)}")
        print(f"Start turning left (CCW), continue until you reach ({p}).")
        print(f"Turn right (CW) until you hit LCP.")
        raw = safe_input("  Enter LCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting isolate wheel 3 (scan not saved).")
            return
        try:
            lcp = float(raw)
        except ValueError:
            print("  Invalid LCP. Try this point again.")
            continue
        print(f"Turn left (CCW) until you hit RCP.")
        raw = safe_input("  Enter RCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting isolate wheel 3 (scan not saved).")
            return
        try:
            rcp = float(raw)
        except ValueError:
            print("  Invalid RCP. Try this point again.")
            continue

        scan_rows.append({
            "id": next_measurement_id(session),

            # Save as ONE sweep so we can reuse plot_sweep()
            "sweep": sweep_id,
            "wheel_swept": 3,
            "combination_wheel_1": awr,
            "combination_wheel_2": awr,
            "combination_wheel_3": p,

            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": rcp - lcp,

            # Isolation metadata
            "iso_test": "isolate_wheel_3",
            "iso_test_id": iso_test_id,
            "iso_phase": "scan",

            "notes": ""
        })

    session["measurements"].extend(scan_rows)

    
    print("\nGenerating plot.")
    # Plot scan (so user can decide candidates)
    plot_sweep(session, sweep_id)

    # Ask for candidate gate locations
    print("\nEnter candidate gate locations.")
    print("Use comma-separated dial positions (example: 12, 12.5, 13).")
    raw = safe_input("Candidate gate locations (leave blank if none found): ", session).strip()
    candidates = []
    if raw:
        try:
            candidates = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
        except ValueError:
            print("Invalid list; skipping refinement for now.")
            candidates = []

    if not candidates:
        print("\nNo candidates entered. Scan saved; refinement skipped.")
        return

    # -------- Phase 2: Refine around candidates (±1 regardless of tolerance) --------
    refine_points = set()
    for g in candidates:
        refine_points.add(_wrap_dial(g - 1.0, dial_min, dial_max))
        refine_points.add(_wrap_dial(g, dial_min, dial_max))
        refine_points.add(_wrap_dial(g + 1.0, dial_min, dial_max))

    # Remove points already measured in scan to avoid duplicates (by dial position)
    measured_positions = set(
        round(float(m.get("combination_wheel_3", -9999)), 6)
        for m in session["measurements"]
        if m.get("sweep") == sweep_id and m.get("wheel_swept") == 3
    )
    refine_points = sorted(p for p in refine_points if round(float(p), 6) not in measured_positions)

    if not refine_points:
        print("\nAll refine points already exist in the sweep. Skipping refine data entry.")
    else:
        print(f"\nREFINE PHASE: {len(refine_points)} NEW points (gate-1, gate, gate+1)")

        print("\nStart at 0, turn right (CW), 3 full turns, ending on 0 again.")
        print(f"Continue to turn right (CW), ending on the AWR Wheel Stack Low Point ({awr}).")

        refine_rows = []
        for i, p in enumerate(refine_points, 1):
            print(f"\nRefine {i}/{len(refine_points)}")
            print(f"Start turning left (CCW), continue until you reach ({p}).")
            print(f"Turn right (CW) until you hit LCP.")

            raw = safe_input("  Enter LCP (or E to exit): ", session).strip()
            if raw.lower() == "e":
                print("Exiting refine early (already-entered refine points will be saved).")
                break
            try:
                lcp = float(raw)
            except ValueError:
                print("  Invalid LCP. Try this point again.")
                continue
            print(f"Turn left (CCW) until you hit RCP.")
            raw = safe_input("  Enter RCP (or E to exit): ", session).strip()
            if raw.lower() == "e":
                print("Exiting refine early (already-entered refine points will be saved).")
                break
            try:
                rcp = float(raw)
            except ValueError:
                print("  Invalid RCP. Try this point again.")
                continue

            refine_rows.append({
                "id": next_measurement_id(session),

                # SAME sweep as scan
                "sweep": sweep_id,
                "wheel_swept": 3,
                "combination_wheel_1": awr,
                "combination_wheel_2": awr,
                "combination_wheel_3": p,

                "left_contact": lcp,
                "right_contact": rcp,
                "contact_width": rcp - lcp,

                "iso_test": "isolate_wheel_3",
                "iso_test_id": iso_test_id,
                "iso_phase": "refine",

                "notes": ""
            })

        session["measurements"].extend(refine_rows)

    # Plot again (combined scan + refine in the SAME sweep)
    plot_sweep(session, sweep_id)

    # Ask whether to update suspected gates for wheel 3
    final_candidates = _edit_gate_estimates(candidates, session, wheel_num=3)

    if not final_candidates:
        print("No candidate gates selected. Wheel 3 configuration not changed.")
    elif ask_yes_no("\nUpdate these as Wheel 3 suspected gates? Warning, may want to perform a High - Low Test to confirm these gates belong to wheel 3 first. (y/n): ", session):
        session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
        lc = session["lock_config"]
        wd = lc["wheel_data"].get("3", {})
        wd["suspected_gates"] = final_candidates
        lc["wheel_data"]["3"] = wd
        session["lock_config"] = lc
        print(f"Wheel 3 suspected gates updated: {final_candidates}")
    else:
        print("Wheel 3 suspected gates not changed.")

def isolate_wheel_2(session: Session) -> None:
    """
    Updated isolate wheel 2 workflow (guided data-entry).
    """

    # Ensure lock_config exists and normalized
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]

    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        print("\n--- Isolate Wheel 2 ---")
        print("This isolate-wheel-2 workflow is only implemented for 3-wheel LRL locks at this time.")
        return

    dial_min = float(lc.get("dial_min", 0))
    dial_max = float(lc.get("dial_max", 99))
    tol = float(lc.get("tolerance", 1))
    if tol <= 0:
        tol = 1.0

    # IMPORTANT: step size is 2x tolerance because tolerance is +/- tolerance
    step_size = 2 * tol

    # Determine next iso_test_id
    existing_ids = [
        int(m.get("iso_test_id"))
        for m in session.get("measurements", [])
        if m.get("iso_test") == "isolate_wheel_2" and str(m.get("iso_test_id", "")).isdigit()
    ]
    iso_test_id = (max(existing_ids) + 1) if existing_ids else 1

    # Determine next sweep id
    existing_sweeps = []
    for m in session.get("measurements", []):
        s = m.get("sweep")
        if isinstance(s, (int, float)):
            existing_sweeps.append(int(s))
        elif isinstance(s, str) and s.replace(".", "", 1).isdigit():
            existing_sweeps.append(int(float(s)))
    sweep_id = (max(existing_sweeps) + 1) if existing_sweeps else 1

    print("\n--- Isolate Wheel 2 (Updated Logic) ---")
    print("This workflow follows the incremental-offset isolation sequence.")
    print("Type 'E' at any prompt to exit back to the Test menu.\n")

    # ============================================================
    # Pick / confirm wheel_1_stop and wheel_3_stop (smart defaults)
    # ============================================================
    wd = lc.get("wheel_data", {}) or {}

    def _first_num(val):
        """Return first numeric item from list-like or scalar, else None."""
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

    def _suggest_stop_value(wheel_num: int) -> tuple[float | None, str]:
        """
        Suggest stop value with priority:
          known gate -> suspected gate -> None
        """
        wdata = wd.get(str(wheel_num), {}) or {}

        known = _first_num(wdata.get("known_gates"))
        if known is not None:
            return known, "known gate"

        suspected = _first_num(wdata.get("suspected_gates"))
        if suspected is not None:
            return suspected, "suspected gate"

        return None, "missing"

    # Wheel 1 suggestion: known -> suspected -> AWL low point -> missing
    wheel_1_stop, w1_src = _suggest_stop_value(1)
    if wheel_1_stop is None:
        awl_low = lc.get("awl_low_point", None)
        if awl_low is not None:
            wheel_1_stop = float(awl_low)
            w1_src = "AWL low point"

    # Wheel 3 suggestion: known -> suspected -> missing
    wheel_3_stop, w3_src = _suggest_stop_value(3)

    # Missing prerequisites -> suggest + allow exit OR manual continue
    missing_msgs = []
    if wheel_3_stop is None:
        missing_msgs.append(
            "Wheel 3 stop is missing (no known/suspected gate). Suggestion: isolate Wheel 3 first."
        )
    if wheel_1_stop is None:
        missing_msgs.append(
            "Wheel 1 stop is missing (no known/suspected gate and no AWL low point). Suggestion: run an All Wheels Left (AWL) test first."
        )

    if missing_msgs:
        print("\n--- Before Starting Isolate Wheel 2 ---")
        for msg in missing_msgs:
            print(msg)

        print("\nYou can:")
        print("  1) Exit and do the suggested prerequisite test")
        print("  2) Continue anyway and manually enter missing stop value(s)")

        choice = safe_input("Choose 1 or 2: ", session).strip()
        if choice == "1":
            print("Exiting isolate wheel 2.")
            return

    def _prompt_float(prompt: str, default: float | None = None) -> float | None:
        """
        Prompt user for a float.
          - If default is provided: Enter accepts default.
          - 'E' exits (returns None).
        """
        if default is None:
            raw = safe_input(prompt, session).strip()
            if raw.lower() == "e":
                return None
            try:
                return float(raw)
            except ValueError:
                return None
        else:
            raw = safe_input(f"{prompt} [{default}]: ", session).strip()
            if raw.lower() == "e":
                return None
            if raw == "":
                return float(default)
            try:
                return float(raw)
            except ValueError:
                return float(default)

    # If still missing required values, force manual entry
    if wheel_1_stop is None:
        print("\nWheel 1 stop is required to proceed.")
        wheel_1_stop = _prompt_float("Enter Wheel 1 stop value (or E to exit): ")
        if wheel_1_stop is None:
            print("Exiting isolate wheel 2.")
            return
        w1_src = "manual entry"

    if wheel_3_stop is None:
        print("\nWheel 3 stop is required to proceed.")
        wheel_3_stop = _prompt_float("Enter Wheel 3 stop value (or E to exit): ")
        if wheel_3_stop is None:
            print("Exiting isolate wheel 2.")
            return
        w3_src = "manual entry"

    # Show the final suggested combination and allow edits
    print("\n--- Suggested Combination for Wheel 2 Isolation ---")
    print(f"Wheel 1 stop = {wheel_1_stop}   ({w1_src})")
    print(f"Wheel 3 stop = {wheel_3_stop}   ({w3_src})")
    print("\nPress Enter to accept, or type a new value.\n")

    new_w1 = _prompt_float("Wheel 1 stop", default=wheel_1_stop)
    if new_w1 is None:
        print("Exiting isolate wheel 2.")
        return
    wheel_1_stop = new_w1

    new_w3 = _prompt_float("Wheel 3 stop", default=wheel_3_stop)
    if new_w3 is None:
        print("Exiting isolate wheel 2.")
        return
    wheel_3_stop = new_w3

    print("\nFinal chosen stops:")
    print(f"  Wheel 1 stop: {wheel_1_stop}")
    print(f"  Wheel 3 stop: {wheel_3_stop}\n")

    print(f"Tolerance = ±{tol}  (step size = {step_size})")
    print(f"Dial range = {dial_min}..{dial_max}\n")

    # =========================
    # Step 2 (initial setup)
    # =========================
    n = 1
    visited_offsets = set()
    rows = []

    print("STEP 2:")
    print(
        f"Turn left (CCW) passing {wheel_1_stop} three times, and continue turning until you hit {wheel_1_stop}, then stop."
    )
    safe_input("Press Enter when ready to continue: ", session)

    # =========================
    # Step 3 (ONCE)
    # =========================
    offset = _wrap_dial(wheel_1_stop - (n * step_size), dial_min, dial_max)
    print(
        f"Turn right (CW) passing {wheel_1_stop} two times, and continue slowly until you hit {offset}, then stop."
    )
    safe_input("Press Enter when ready to continue: ", session)
    n += 1  # n++

    # =========================
    # Steps 4–9 repeat loop
    # =========================
    max_cycles = int((dial_max - dial_min + 1) / max(step_size, 1)) + 10

    for cycle in range(1, max_cycles + 1):
        # Step 10: full revolution detection (offset repeats)
        key = round(float(offset), 6)
        if key in visited_offsets and cycle >= 3:
            print("\nFull revolution detected (offset repeated). Stopping scan phase.\n")
            break
        visited_offsets.add(key)

        print("\n" + "-" * 60)
        print(f"Cycle {cycle}")
        # print(f"Current offset target = {offset}   (n={n})")

        # Step 4
        print(
            f"Turn left (CCW) passing {offset} one time, and continue turning until you hit {wheel_3_stop}, then stop."
        )

        # Step 5 + 6
        print("\nTurn right (CW) until you hit LCP.")
        raw = safe_input("  Enter LCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting isolate wheel 2 (keeping collected measurements).")
            break
        try:
            lcp = float(raw)
        except ValueError:
            print("Invalid LCP; skipping this cycle.")
            lcp = None

        # Step 7 + 8
        print("\nTurn left (CCW) until you hit RCP.")
        raw = safe_input("  Enter RCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting isolate wheel 2 (keeping collected measurements).")
            break
        try:
            rcp = float(raw)
        except ValueError:
            print("Invalid RCP; skipping this cycle.")
            rcp = None

        # Record only if both exist
        if lcp is not None and rcp is not None:
            rows.append({
                "id": next_measurement_id(session) + len(rows),

                "sweep": sweep_id,
                "wheel_swept": 2,

                "combination_wheel_1": wheel_1_stop,
                "combination_wheel_2": offset,
                "combination_wheel_3": wheel_3_stop,

                "left_contact": lcp,
                "right_contact": rcp,
                "contact_width": rcp - lcp,

                "iso_test": "isolate_wheel_2",
                "iso_test_id": iso_test_id,
                "iso_phase": "scan",

                "notes": ""
            })

        # Step 9
        next_offset = _wrap_dial(wheel_1_stop - (n * step_size), dial_min, dial_max)
        print(
            f"\nTurn right (CW) until you hit {wheel_3_stop}, then continue to {next_offset}."
        )
        safe_input("Press Enter when ready to continue: ", session)
        n += 1  # n++

        # Prepare for next cycle
        offset = next_offset

    if not rows:
        print("No usable measurements collected.")
        return

    session["measurements"].extend(rows)

    print("\nScan complete. Generating plot...")
    plot_sweep(session, sweep_id)

    # =========================
    # Step 11: Candidate gates
    # =========================
    scan_positions = [float(r["combination_wheel_2"]) for r in rows]
    lcps = [float(r["left_contact"]) for r in rows]
    rcps = [float(r["right_contact"]) for r in rows]

    candidates = set()
    for i in range(1, len(rows) - 1):
        if lcps[i] > lcps[i - 1] and lcps[i] > lcps[i + 1]:
            candidates.add(scan_positions[i])
        if rcps[i] < rcps[i - 1] and rcps[i] < rcps[i + 1]:
            candidates.add(scan_positions[i])

    candidates = sorted(candidates)

    print("\n--- Candidate Gates (auto-detected) ---")
    if not candidates:
        print("No clear candidate gates detected from LCP/RCP shape.")
        print("You may still manually inspect the plot for gate-like behavior.")
        return

    print("Possible gates at Wheel 2 positions:")
    for c in candidates:
        print(f"  {c}")

    # =========================
    # Step 12: refine ±1
    # =========================
    print("\n--- Refinement: test ±1 around each candidate ---")
    refine_points = set()
    for c in candidates:
        refine_points.add(_wrap_dial(c - 1.0, dial_min, dial_max))
        refine_points.add(_wrap_dial(c, dial_min, dial_max))
        refine_points.add(_wrap_dial(c + 1.0, dial_min, dial_max))

    already = set(round(float(p), 6) for p in scan_positions)
    refine_points = sorted(p for p in refine_points if round(float(p), 6) not in already)

    if not refine_points:
        print("No new refinement points needed.")
        return

    print(f"Refining {len(refine_points)} points: {refine_points}\n")

    refine_rows = []
    for i, p in enumerate(refine_points, 1):
        print("\n" + "-" * 60)
        print(f"Refine point {i}/{len(refine_points)} @ Wheel 2 = {p}")

        print(
            f"Turn left (CCW) passing {wheel_1_stop} three times, and continue turning until you hit {wheel_1_stop}, then stop."
        )
        print(
            f"Turn right (CW) passing {wheel_1_stop} two times, and continue slowly until you hit {p}, then stop."
        )
        print(
            f"Turn left (CCW) passing {p} one time, and continue turning until you hit {wheel_3_stop}, then stop."
        )

        print("Turn right (CW) until you hit LCP.")
        raw = safe_input("  Enter LCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting refinement early (keeping already-entered refinement points).")
            break
        try:
            lcp = float(raw)
        except ValueError:
            print("Invalid LCP; skipping.")
            continue

        print("Turn left (CCW) until you hit RCP.")
        raw = safe_input("  Enter RCP (or E to exit): ", session).strip()
        if raw.lower() == "e":
            print("Exiting refinement early (keeping already-entered refinement points).")
            break
        try:
            rcp = float(raw)
        except ValueError:
            print("Invalid RCP; skipping.")
            continue

        refine_rows.append({
            "id": next_measurement_id(session) + len(refine_rows),

            "sweep": sweep_id,
            "wheel_swept": 2,

            "combination_wheel_1": wheel_1_stop,
            "combination_wheel_2": p,
            "combination_wheel_3": wheel_3_stop,

            "left_contact": lcp,
            "right_contact": rcp,
            "contact_width": rcp - lcp,

            "iso_test": "isolate_wheel_2",
            "iso_test_id": iso_test_id,
            "iso_phase": "refine",

            "notes": ""
        })

    if refine_rows:
        session["measurements"].extend(refine_rows)
        print("\nRefinement saved. Plotting combined scan + refine...")
        plot_sweep(session, sweep_id)

    # Optional: store candidates into suspected gates for wheel 2
    final_candidates = _edit_gate_estimates(candidates, session, wheel_num=2)

    if not final_candidates:
        print("No candidate gates selected. Wheel 2 configuration not changed.")
        return

    if ask_yes_no("\nUpdate these as Wheel 2 suspected gates? (y/n): ", session):
        session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
        lc = session["lock_config"]

        wd2 = lc["wheel_data"].get("2", {}) or {}
        wd2["suspected_gates"] = final_candidates
        lc["wheel_data"]["2"] = wd2

        session["lock_config"] = lc
        print(f"Wheel 2 suspected gates updated: {final_candidates}")
    else:
        print("Wheel 2 suspected gates not changed.")

def exhaustive_enumeration_all(session: Session) -> None:
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]
    wd = lc["wheel_data"]

    # 1️⃣ Build candidates FIRST
    candidates = {}
    for w in range(1, lc["wheels"] + 1):
        gates = _effective_gate_list(wd[str(w)])
        if not gates:
            print(f"Wheel {w} has no known or suspected gates. Aborting.")
            return
        candidates[w] = list(gates)

    # 2️⃣ Build full combination list AFTER candidates exist
    all_combos = list(product(
        candidates[1],
        candidates[2],
        candidates[3]
    ))

    print("\n--- Exhaustive Enumeration (All Wheels) ---")
    print("For each combination:")
    print("  T = Tested (no success)")
    print("  S = Success (combination verified)")
    print("  Enter = skip")
    print("  E = exit\n")

    def recurse(current, idx):
        if idx > lc["wheels"]:
            combo = [current[i] for i in range(1, lc["wheels"] + 1)]

            if _combination_already_tested(session, combo):
                return

            remaining = _count_remaining_enumerations(session, all_combos)
            print(f"Remaining combinations: {remaining}")
            print(f"\nTry combination: {combo}")

            resp = safe_input("Result [T/S/E/Enter]: ", session).lower()

            if resp == "e":
                raise StopIteration
            elif resp == "s":
                _handle_exhaustive_enumeration_success(session, combo, "exhaustive_enumeration_all")
                raise StopIteration
            elif resp == "t":
                _record_exhaustive_enumeration_attempt(session, combo, "exhaustive_enumeration_all", "closed")
            return

        for g in candidates[idx]:
            current[idx] = g
            recurse(current, idx + 1)

    try:
        recurse({}, 1)
    except StopIteration:
        pass

def exhaustive_enumeration_wheel(session: Session, wheel_num: int) -> None:
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]
    wd = lc["wheel_data"]

    step = max(1, int(lc.get("tolerance", 1)))
    dial_min = int(lc["dial_min"])
    dial_max = int(lc["dial_max"])

    # Build candidate lists
    wheel_values = {}
    for w in range(1, lc["wheels"] + 1):
        if w == wheel_num:
            wheel_values[w] = list(range(dial_min, dial_max + 1, step))
        else:
            gates = _effective_gate_list(wd[str(w)])
            if not gates:
                print(f"Wheel {w} has no known or suspected gates. Aborting.")
                return
            wheel_values[w] = list(gates)

    # Generate full ordered list
    all_combos = list(product(
        wheel_values[1],
        wheel_values[2],
        wheel_values[3]
    ))

    print(f"\n--- Exhaustive Enumeration Wheel {wheel_num} ---")
    print("For each combination:")
    print("  T = Tested (no success)")
    print("  S = Success (combination verified)")
    print("  Enter = skip")
    print("  E = exit\n")

    for combo in all_combos:
        combo = list(combo)

        if _combination_already_tested(session, combo):
            continue

        remaining = _count_remaining_enumerations(session, all_combos)
        print(f"Remaining combinations: {remaining}")
        print(f"\nTry combination: {combo}")

        resp = safe_input("Result [T/S/E/Enter]: ", session).lower()

        if resp == "e":
            return
        elif resp == "s":
            _handle_exhaustive_enumeration_success(session, combo, "exhaustive_enumeration_wheel")
            return
        elif resp == "t":
            _record_exhaustive_enumeration_attempt(session, combo, "exhaustive_enumeration_wheel", "closed")
        # Enter = skip → just continue

    print("\nAll combinations exhausted.")

def show_tutorial(session: Session) -> None:
    print("\n--- Tutorial ---\n")

    # Ensure config is normalized so wheel_data always exists
    session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
    lc = session["lock_config"]
    wd = lc.get("wheel_data", {})

    # Scope restriction
    wheels = int(lc.get("wheels", 0) or 0)
    turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
    if wheels != 3 or turn_seq != "LRL":
        print("This tutorial is only implemented for 3-wheel LRL locks at this time.")
        safe_input("\nPress Enter to return to Help menu: ", session)
        return

    print(textwrap.dedent("""
    This tutorial will walk you through manipulating the lock in your CURRENT configuration.

    Make sure your configuration is up to date with everything you know so far
    (wheel count, dial range/tolerance, AWR/AWL reference points, and any gates).
    """).strip())

    if not ask_yes_no("\nProceed using the current configuration? (y/n): ", session):
        print("\nTutorial cancelled.")
        return

    # Implementation nuance:
    # If a wheel has known gates, we silently ignore suspected gates for that wheel.
    def effective_candidates_for_wheel(w: int):
        data = wd.get(str(w), {})
        known = data.get("known_gates", []) or []
        suspected = data.get("suspected_gates", []) or []
        return known if known else suspected

    def has_known_gate(w: int) -> bool:
        return len(wd.get(str(w), {}).get("known_gates", []) or []) > 0

    def has_known_or_suspected(w: int) -> bool:
        return len(effective_candidates_for_wheel(w)) > 0

    def wheels_with_known_or_suspected():
        return [w for w in (1, 2, 3) if has_known_or_suspected(w)]

    def try_action_prompt() -> bool:
        return ask_yes_no("Try suggested action now? (y/n): ", session)

    step = 1
    while not (has_known_gate(1) and has_known_gate(2) and has_known_gate(3)):

        print("\n" + "=" * 60)
        print(f"Step {step} — Current effective gate knowledge:")

        for w in (1, 2, 3):
            k = wd.get(str(w), {}).get("known_gates", []) or []
            s = wd.get(str(w), {}).get("suspected_gates", []) or []
            using = effective_candidates_for_wheel(w)
            print(f"  Wheel {w}: known={k} | suspected={s} | using={using}")

        wks = wheels_with_known_or_suspected()

        # Decision logic
        if len(wks) == 3:
            kind, detail = ("EXHAUSTIVE_ENUM_ALL", None)
        elif len(wks) == 2:
            missing = [w for w in (1, 2, 3) if w not in wks][0]
            kind, detail = ("EXHAUSTIVE_ENUM_WHEEL", missing)
        elif len(wks) == 1:
            kind, detail = (("ISOLATE_WHEEL_2", None) if wks[0] == 3 else ("ISOLATE_WHEEL_3", None))
        else:
            kind, detail = ("ISOLATE_WHEEL_3", None)

        print("\nSuggested action:")

        if kind == "ISOLATE_WHEEL_3":
            if lc.get("awr_low_point") is None:
                print("  → FIRST: Find the All Wheels Right (AWR) wheel stack low point.")
                if try_action_prompt():
                    find_awr_low_point(session)
            else:
                print("  → ISOLATE WHEEL 3.")
                if try_action_prompt():
                    isolate_wheel_3(session)

        elif kind == "ISOLATE_WHEEL_2":
            if lc.get("awl_low_point") is None:
                print("  → FIRST: Find the All Wheels Left (AWL) wheel stack low point.")
                if try_action_prompt():
                    find_awl_low_point(session)
            else:
                print("  → ISOLATE WHEEL 2.")
                if try_action_prompt():
                    isolate_wheel_2(session)

        elif kind == "EXHAUSTIVE_ENUM_ALL":
            print("  → Exhaustive Enumeration using the current gate candidates.")
            if try_action_prompt():
                exhaustive_enumeration_all(session)

        elif kind == "EXHAUSTIVE_ENUM_WHEEL":
            print(f"  → Exhaustive Enumeration for Wheel {detail}.")
            if try_action_prompt():
                exhaustive_enumeration_wheel(session, detail)

        resp = safe_input("\nPress Enter to continue, or type E to exit tutorial: ", session).strip().lower()
        if resp == "e":
            print("\nExiting tutorial.")
            return

        # Reload config in case user updated it
        session["lock_config"] = normalize_lock_config(session.get("lock_config", {}))
        lc = session["lock_config"]
        wd = lc.get("wheel_data", {})

        # Defensive scope check
        wheels = int(lc.get("wheels", 0) or 0)
        turn_seq = str(lc.get("turn_sequence", "")).strip().upper()
        if wheels != 3 or turn_seq != "LRL":
            print("\nTutorial is only implemented for 3-wheel LRL locks at this time.")
            safe_input("\nPress Enter to return to Help menu: ", session)
            return

        step += 1

    print("\nAll three wheels have at least one KNOWN gate.")
    safe_input("\nPress Enter to return to Help menu: ", session)


if __name__ == "__main__":
    main()
