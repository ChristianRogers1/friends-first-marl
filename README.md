# 6955-Project

PettingZoo-based experiment code for the CS 6955 project.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want to import the package from the repo without installing it, run scripts with:

```bash
PYTHONPATH=src
```

## Run

Smoke test the environment and experiment code:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Run the MADDPG experiment pipeline:

```bash
PYTHONPATH=src python scripts/run_maddpg_experiment.py --train-episodes 2500 --eval-episodes 200
```

Run the simpler non-MADDPG baseline script:

```bash
PYTHONPATH=src python scripts/run_screen_confirm_experiment.py
```
