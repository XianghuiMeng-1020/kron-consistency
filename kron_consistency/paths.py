from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
IEEE_DIR = DATA_DIR / "ieee"
TRAFFIC_DIR = DATA_DIR / "traffic"
PLANETOID_DIR = DATA_DIR / "planetoid"
