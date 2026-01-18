# Combination Lock Analysis Suite (CLAS)

Combination Lock Analysis Suite (CLAS) is an open-source, educational
locksport utility for recording, analyzing, and guiding mechanical
combination dial analysis workflows.

The tool is intended for learning, training, and locksport with
mechanical combination locks that you own or have explicit permission
to work on.

## Features

- Records left and right contact point measurements
- Sweep and high–low test data collection and visualization
- Guided, step-by-step tutorial that suggests the next action based on
  your current lock configuration
- Session-based workflow with saved configuration and measurements
- Crash recovery: user can recover and repair sessions after an unexpected crash
- Core functions can be ran with Command Line Interface (CLI) or via web FastAPI
- Auditable Python implementation (kept intentionally small and easy to review)

## Usage

### Requirements

- **Python**
  - **CLI/Core:** Python 3.9+ recommended
  - **Web adapter (FastAPI stub):** Python 3.10+ recommended
- Dependencies are split by interface:
  - `requirements.txt` = core + CLI
  - `requirements-web.txt` = core + CLI + web adapter extras

Install (CLI only):

```bash
pip install -r requirements.txt
````

Install (CLI + Web adapter):

```bash
pip install -r requirements-web.txt
```

### Running CLAS (CLI)

From the project root directory:

```bash
python3 clas_cli.py
```

You can also open a specific session file:

```bash
python3 clas_cli.py --session path/to/session.json
```

Follow the on-screen menus to configure the lock, collect measurements,
analyze results, or use the built-in tutorial.

### Global Commands (CLI)

At **any** input prompt:

* `q` quits the program (clean exit)
* `s` saves the primary session JSON (and then returns you to the same prompt)
* `u` undoes the last accepted action and returns you to the previous prompt
* `a` aborts the current workflow/menu level (like “back”)

> Note: These letters are reserved. If a workflow would normally use them as normal input, the prompt/options should be changed to avoid conflicts.

### Tutorial

CLAS includes an interactive tutorial designed to walk you through the
analysis process step-by-step.

The tutorial uses your current configuration and recorded data to
suggest appropriate next actions such as isolating wheels or sweeping
a wheel per candidate gates.

You can access the tutorial from:

```
Help → Tutorial
```

### Running the Web Adapter (optional)

CLAS includes a minimal FastAPI web adapter stub to demonstrate how the same
core engine can run behind a web interface.

Run (example):

```bash
python -m uvicorn clas_web:app --reload --host 127.0.0.1 --port 8000
```

Basic flow:

* `POST /sessions` creates a new session and returns a `session_id`
* `GET /sessions/{session_id}/prompt` returns the next prompt to display
* `POST /sessions/{session_id}/action` sends user input (or commands like undo/abort)
* `POST /sessions/{session_id}/undo` is a convenience endpoint

## Data and Sessions

* All measurements and configuration are stored locally in JSON files
* Example session data is provided in `example-session.json`
* No network access or telemetry is used with CLI

### Crash recovery mirror file (CLI)

In addition to the normal session JSON, the CLI maintains a recovery mirror file
in the same folder as the session:

* `session-name-YYYYMMDD-HHMMSS.json.recovery-data`

Rules:

* Updated after **every accepted action** (even if you never press `s`)
* Deleted on clean program exit (`q`) or session change
* If the program crashes unexpectedly, the file remains and will be offered
  as a recovery option next time you start the CLI

## Project Status and Community Contributions

CLAS is an early-stage, community-oriented project. As such, it should be considered a draft tool: there may be bugs, rough edges, incomplete workflows, or design decisions that can be improved.

Constructive feedback and contributions are strongly encouraged. Preferred ways to engage include:

* Reporting bugs or issues
* Suggesting improvements or new features
* Submitting pull requests to fix problems or extend functionality
* Discussing design decisions or workflows in good faith

CLAS is intentionally open-source so that it can improve through shared effort and peer review. The project is licensed under GPL-3.0-only, which ensures that improvements and derivative works remain open and available to the community rather than being converted into closed, proprietary tools.

If you find limitations or shortcomings, the most productive path is to help improve the shared tool for everyone’s benefit.

## License

This project is licensed under the GNU General Public License v3.0
(GPL-3.0-only). See the `LICENSE` file for details.

## Disclaimer

This software is provided for educational and lawful purposes only.
The author does not condone or support illegal or unauthorized access
to property or devices.
