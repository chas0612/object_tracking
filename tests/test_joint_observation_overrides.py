"""CPU-only regressions for opt-in joint observation settings."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autodex.tracking.latest_gotrack import adapter_config, joint_observation_overrides


def main():
    assert joint_observation_overrides(SimpleNamespace()) == {}
    assert joint_observation_overrides(SimpleNamespace(joint_residual_ratio=None, min_moving_anchors=None)) == {}
    assert joint_observation_overrides(SimpleNamespace(joint_residual_ratio=1., min_moving_anchors=12)) == {
        'theta_residual_ratio': 1., 'min_moving_anchors': 12}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        (root/'configs/tracker').mkdir(parents=True)
        (root/'configs/tracker/trifit.json').write_text('{}')
        args=SimpleNamespace(backend='trifit',checkpoint_path=root/'checkpoint.pt',
                             output_root=root/'output',num_anchors=512,observer_compile=True)
        default=adapter_config(root,args,6)
        assert default['official_config_path']==str(root/'configs/tracker/runner_ycbv.json')
        profile=root/'profile.json'
        profile.write_text('{"anchor_confidence_threshold":0.5}')
        args.official_config=profile
        overridden=adapter_config(root,args,6)
        snapshot=Path(overridden['official_config_path'])
        assert snapshot!=profile and snapshot.is_relative_to(args.output_root)
        assert json.loads(snapshot.read_text())==json.loads(profile.read_text())
        assert overridden['official_config_sha256']==hashlib.sha256(snapshot.read_bytes()).hexdigest()
    base = [sys.executable, str(REPO / 'src/process/run_gotrack_latest.py'), '--backend', 'trifit']
    for option in ('gotrack-dir', 'input-root', 'output-root', 'mesh-path', 'init-pose-source', 'object-name', 'checkpoint-path'):
        base += [f'--{option}', 'unused']
    base += ['--camera-ids', 'a', 'b']
    for options in (['--joint-residual-ratio', '1.'],
                    ['--max-joint-step-deg', '10'],
                    ['--articulation-json','unused','--max-joint-step-deg','nan'],
                    ['--articulation-json','unused','--max-joint-step-deg','0'],
                    ['--official-config','/not/an/existing/config.json'],
                    ['--articulation-json','unused','--joint-residual-ratio','nan'],
                    ['--articulation-json','unused','--joint-residual-ratio','0'],
                    ['--articulation-json','unused','--min-moving-anchors','2']):
        result = subprocess.run(base + options, capture_output=True, text=True)
        assert result.returncode == 2, result.stderr
        assert 'error:' in result.stderr and 'Traceback' not in result.stderr
    print('PASS: optional defaults, explicit gates, and CPU-side CLI validation')


if __name__ == '__main__':
    main()
