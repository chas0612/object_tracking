"""Explicit private-runtime locations; ignored checkouts do not follow branch switches."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack"
LATEST_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack-next"
LATEST_SHA = "842bdfeb61f7f15dd649172077cc72ad6a45d718"
CHECKPOINT_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"


def checkpoint_path(root: Path) -> Path:
    for path in (root / "checkpoints/gotrack_checkpoint.pt", root / "gotrack_checkpoint.pt"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"GoTrack checkpoint missing under {root}")
