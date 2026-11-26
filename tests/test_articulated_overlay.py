"""CPU regression: fewer than six accepted masks must not break init output."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'src/process/articulated'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('real_first_pose', SOURCE / 'real_first_pose.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        points = np.array([[1., 1., 1.]])
        articulation = SimpleNamespace(posed=lambda pose, theta: (points, points))
        camera = SimpleNamespace(width=12, height=10, project=lambda p: p[:, :2])
        for count in (1, 3, 4, 5, 6):
            masks, cameras = {}, {}
            for index in range(count):
                cid = str(index)
                masks[cid] = np.ones((10, 12), dtype=bool)
                cameras[cid] = camera
                cv2.imwrite(str(root / f'{cid}.png'), np.full((10, 12, 3), 128, np.uint8))
            target = root / f'overlay_{count}.png'
            module._overlay_sheet(target, root, masks, cameras, articulation, np.eye(4), 0.)
            assert cv2.imread(str(target)).shape == (20, 36, 3)
        try:
            module._overlay_sheet(root / 'empty.png', root, {}, {}, articulation, np.eye(4), 0.)
        except ValueError:
            pass
        else:
            raise AssertionError('Empty input must produce a clear error')
    print('PASS: articulated overlay with 1/3/4/5/6 views and empty input')


if __name__ == '__main__':
    main()
