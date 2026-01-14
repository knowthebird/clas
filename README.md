# Combination Lock Analysis Suite (CLAS)

Combination Lock Analysis Suite (CLAS) is an open-source, educational
locksport utility for recording, analyzing, and guiding mechanical
combination dial analysis workflows.

The tool is intended for learning, training, and locksport with
mechanical combination locks that you own or have explicit permission
to work on.

## Features

- Single-file, auditable Python implementation
- Records left and right contact point measurements
- Sweep and high–low test data collection and visualization
- Guided, step-by-step tutorial that suggests the next action based on
  your current lock configuration
- Session-based workflow with saved configuration and measurements

## Usage

### Requirements

- Python 3.8 or newer
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Running CLAS

From the project root directory:

```bash
python3 clas.py
```

Follow the on-screen menus to configure the lock, collect measurements,
analyze results, or use the built-in tutorial.

### Tutorial

CLAS includes an interactive tutorial designed to walk you through the
analysis process step-by-step.

The tutorial uses your current configuration and recorded data to
suggest appropriate next actions such as isolating wheels or exhaustive
enumeration of candidate gates.

You can access the tutorial from:

```
Help → Tutorial
```

## Data and Sessions

* All measurements and configuration are stored locally in JSON files
* Example session data is provided in `example-session.json`
* No network access or telemetry is used

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