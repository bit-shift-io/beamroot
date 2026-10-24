import json
import time
from pathlib import Path


def create_run_dir(name: str, root: str | Path = "runs", config: dict | None = None) -> Path:
    """Create runs/<timestamp>-<name>/ and write config.json."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{stamp}-{name}"
    n = 1
    while path.exists():
        path = Path(root) / f"{stamp}-{name}-{n}"
        n += 1
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config or {}, indent=2))
    return path


def log_metrics(run_dir: Path, record: dict) -> None:
    with open(Path(run_dir) / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
