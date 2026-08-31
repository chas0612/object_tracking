from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import distribute_foundpose_gotrack_phased as phased

_foundpose_spec = importlib.util.spec_from_file_location(
    "foundpose_init_standalone", REPO_ROOT / "autodex/perception/foundpose_init.py",
)
assert _foundpose_spec is not None and _foundpose_spec.loader is not None
_foundpose_module = importlib.util.module_from_spec(_foundpose_spec)
_foundpose_spec.loader.exec_module(_foundpose_module)
FoundPoseInit = _foundpose_module.FoundPoseInit


def _task(task_id: str, object_name: str, episode: int) -> dict:
    return {
        "task_id": task_id,
        "object_name": object_name,
        "source_object": object_name,
        "episode": str(episode),
        "phases": {
            phase: {"status": "pending", "attempts": 0, "worker_id": None}
            for phase in phased.PHASES
        },
    }


def _schedule(tmp_path: Path, tasks: list[dict]) -> Path:
    schedule = tmp_path / "schedule"
    (schedule / "tasks").mkdir(parents=True)
    (schedule / "claims").mkdir()
    for task in tasks:
        (schedule / "tasks" / f"{task['task_id']}.json").write_text(
            json.dumps(task), encoding="utf-8",
        )
    return schedule


def test_phase_is_blocked_until_all_prerequisites_complete(tmp_path: Path) -> None:
    task = _task("apple_0", "apple", 0)
    schedule = _schedule(tmp_path, [task])

    assert phased._claim(schedule, "foundpose", "w0", False, 3, 0, 1) is None
    path = schedule / "tasks/apple_0.json"
    task = json.loads(path.read_text(encoding="utf-8"))
    task["phases"]["mask"]["status"] = "completed"
    path.write_text(json.dumps(task), encoding="utf-8")

    claimed = phased._claim(schedule, "foundpose", "w0", False, 3, 0, 1)
    assert claimed is not None
    assert claimed["phases"]["foundpose"]["status"] == "running"


def test_foundpose_object_sharding_keeps_object_on_one_worker(tmp_path: Path) -> None:
    tasks = [_task(f"apple_{episode}", "apple", episode) for episode in range(3)]
    tasks += [_task(f"banana_{episode}", "banana", episode) for episode in range(2)]
    for task in tasks:
        task["phases"]["mask"]["status"] = "completed"
    schedule = _schedule(tmp_path, tasks)
    apple_rank = phased._object_shard("apple", 2)
    other_rank = 1 - apple_rank

    claimed = phased._claim(
        schedule, "foundpose", "owner", False, 3, apple_rank, 2,
    )
    assert claimed is not None
    if phased._object_shard("banana", 2) != apple_rank:
        assert claimed["object_name"] == "apple"

    wrong_worker_objects = set()
    while True:
        item = phased._claim(
            schedule, "foundpose", "other", False, 3, other_rank, 2,
        )
        if item is None:
            break
        wrong_worker_objects.add(item["object_name"])
    assert "apple" not in wrong_worker_objects


def test_reset_running_releases_phase_claim(tmp_path: Path) -> None:
    task = _task("apple_0", "apple", 0)
    schedule = _schedule(tmp_path, [task])
    claimed = phased._claim(schedule, "mask", "w0", False, 3, 0, 1)
    assert claimed is not None
    claim = schedule / "claims/mask/apple_0.lock"
    assert claim.is_dir()

    assert phased._reset_running(schedule, "mask", True) == 0
    task = json.loads((schedule / "tasks/apple_0.json").read_text(encoding="utf-8"))
    assert task["phases"]["mask"]["status"] == "pending"
    assert not claim.exists()


def test_foundpose_rebind_reuses_shared_backbone(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    repre_dir = assets / "object_repre/v1/apple/1"
    repre_dir.mkdir(parents=True)
    (repre_dir / "repre.pth").write_bytes(b"complete")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def to(self, device):
            self.calls.append(("to", device))
            return self

        def eval(self):
            self.calls.append(("eval",))
            return self

        def onboarding(self, *, repre_dir):
            self.calls.append(("onboarding", Path(repre_dir)))

        def post_onboarding_processing(self):
            self.calls.append(("post",))

    model = FakeModel()
    session = FoundPoseInit(
        mesh_path=str(tmp_path / "apple.obj"),
        assets_root=str(assets), obj_name="apple", shared_model=model,
    )
    assert session.model is model
    assert ("onboarding", assets / "object_repre/v1/apple") in model.calls
    assert model.calls[-1] == ("post",)
