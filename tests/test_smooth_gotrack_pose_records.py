import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).parents[1] / "src/process/smooth_gotrack_pose_records.py"
SPEC = importlib.util.spec_from_file_location("smooth_gotrack_pose_records", MODULE_PATH)
smooth = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smooth)


def poses_from_translation(values: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    poses[:, :3, 3] = values
    return poses


class PoseFilterTest(unittest.TestCase):
    def test_filters_reduce_stationary_translation_noise(self) -> None:
        rng = np.random.default_rng(7)
        noisy = rng.normal(scale=0.001, size=(300, 3))
        poses = poses_from_translation(noisy)
        gaussian = smooth._smooth_segment(poses, radius=3, sigma=1.5)
        one_euro = smooth._one_euro_segment(
            poses,
            np.arange(len(poses), dtype=np.float64) / 30.0,
            translation_min_cutoff_hz=1.0,
            translation_beta=200.0,
            rotation_min_cutoff_hz=1.0,
            rotation_beta=5.0,
            derivative_cutoff_hz=1.0,
        )
        raw_rms = np.sqrt(np.mean(np.square(noisy)))
        self.assertLess(np.sqrt(np.mean(np.square(gaussian[:, :3, 3]))), raw_rms)
        self.assertLess(np.sqrt(np.mean(np.square(one_euro[:, :3, 3]))), raw_rms)

    def test_one_euro_preserves_abrupt_translation_better_than_gaussian(self) -> None:
        translations = np.zeros((60, 3), dtype=np.float64)
        translations[30:, 0] = 0.060
        poses = poses_from_translation(translations)
        gaussian = smooth._smooth_segment(poses, radius=3, sigma=1.5)
        one_euro = smooth._one_euro_segment(
            poses,
            np.arange(len(poses), dtype=np.float64) / 30.0,
            translation_min_cutoff_hz=1.0,
            translation_beta=200.0,
            rotation_min_cutoff_hz=1.0,
            rotation_beta=5.0,
            derivative_cutoff_hz=1.0,
        )
        raw_peak = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).max()
        gaussian_peak = np.linalg.norm(np.diff(gaussian[:, :3, 3], axis=0), axis=1).max()
        one_euro_peak = np.linalg.norm(np.diff(one_euro[:, :3, 3], axis=0), axis=1).max()
        self.assertGreater(one_euro_peak / raw_peak, 0.9)
        self.assertGreater(one_euro_peak, gaussian_peak * 2.0)
        # A causal filter must not move before the abrupt event; the symmetric
        # Gaussian necessarily leaks future motion into preceding frames.
        np.testing.assert_allclose(one_euro[29, :3, 3], poses[29, :3, 3], atol=1e-12)
        self.assertGreater(np.linalg.norm(gaussian[29, :3, 3]), 0.0)

    def test_gated_gaussian_keeps_motion_boundary_raw(self) -> None:
        translations = np.zeros((60, 3), dtype=np.float64)
        translations[:, 1] = np.random.default_rng(8).normal(scale=0.0003, size=60)
        translations[30:, 0] += 0.060
        poses = poses_from_translation(translations)
        filtered, translation_enabled, rotation_enabled = smooth._gated_gaussian_segment(
            poses,
            radius=3,
            sigma=1.5,
            fps=30.0,
            translation_threshold_mm_s=30.0,
            rotation_threshold_deg_s=15.0,
            motion_span_frames=11,
            minimum_enabled_run_frames=15,
            hard_translation_threshold_mm_s=300.0,
            hard_rotation_threshold_deg_s=300.0,
        )
        np.testing.assert_allclose(filtered[27:33], poses[27:33], atol=1e-12)
        self.assertFalse(translation_enabled[27:33].any())
        self.assertTrue(translation_enabled[:20].any())
        self.assertTrue(rotation_enabled.all())
        self.assertLess(np.std(filtered[:20, 1, 3]), np.std(poses[:20, 1, 3]))

    def test_one_euro_rotation_stays_on_so3(self) -> None:
        poses = poses_from_translation(np.zeros((80, 3), dtype=np.float64))
        poses[40:, :3, :3] = Rotation.from_euler("z", 120, degrees=True).as_matrix()
        filtered = smooth._one_euro_segment(
            poses,
            np.arange(len(poses), dtype=np.float64) / 30.0,
            translation_min_cutoff_hz=1.0,
            translation_beta=200.0,
            rotation_min_cutoff_hz=1.0,
            rotation_beta=5.0,
            derivative_cutoff_hz=1.0,
        )
        identity = np.einsum("nij,nkj->nik", filtered[:, :3, :3], filtered[:, :3, :3])
        np.testing.assert_allclose(identity, np.repeat(np.eye(3)[None], len(filtered), axis=0), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(filtered[:, :3, :3]), 1.0, atol=1e-10)

    def test_gated_gaussian_separates_translation_and_rotation(self) -> None:
        translations = np.zeros((60, 3), dtype=np.float64)
        translations[:, 2] = np.random.default_rng(9).normal(scale=0.0002, size=60)
        poses = poses_from_translation(translations)
        poses[30:, :3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        filtered, translation_enabled, rotation_enabled = smooth._gated_gaussian_segment(
            poses,
            radius=3,
            sigma=1.5,
            fps=30.0,
            translation_threshold_mm_s=30.0,
            rotation_threshold_deg_s=15.0,
            motion_span_frames=11,
            minimum_enabled_run_frames=15,
            hard_translation_threshold_mm_s=300.0,
            hard_rotation_threshold_deg_s=300.0,
        )
        self.assertTrue(translation_enabled.all())
        self.assertFalse(rotation_enabled[27:33].any())
        np.testing.assert_allclose(
            filtered[27:33, :3, :3], poses[27:33, :3, :3], atol=1e-12
        )
        self.assertLess(np.std(filtered[:, 2, 3]), np.std(poses[:, 2, 3]))

    def test_gated_gaussian_ramps_correction_at_raw_boundary(self) -> None:
        translations = np.zeros((80, 3), dtype=np.float64)
        translations[:, 1] = np.random.default_rng(10).normal(scale=0.0004, size=80)
        translations[40:, 0] += 0.060
        poses = poses_from_translation(translations)
        hard, hard_enabled, _ = smooth._gated_gaussian_segment(
            poses,
            radius=3,
            sigma=1.5,
            fps=30.0,
            translation_threshold_mm_s=15.0,
            rotation_threshold_deg_s=7.5,
            motion_span_frames=11,
            minimum_enabled_run_frames=15,
            hard_translation_threshold_mm_s=300.0,
            hard_rotation_threshold_deg_s=300.0,
            transition_frames=0,
        )
        ramped, ramped_enabled, _ = smooth._gated_gaussian_segment(
            poses,
            radius=3,
            sigma=1.5,
            fps=30.0,
            translation_threshold_mm_s=15.0,
            rotation_threshold_deg_s=7.5,
            motion_span_frames=11,
            minimum_enabled_run_frames=15,
            hard_translation_threshold_mm_s=300.0,
            hard_rotation_threshold_deg_s=300.0,
            transition_frames=3,
        )
        np.testing.assert_array_equal(ramped_enabled, hard_enabled)
        boundaries = np.flatnonzero(np.diff(hard_enabled.astype(np.int8)) != 0) + 1
        self.assertGreater(len(boundaries), 0)
        for boundary in boundaries:
            hard_jump = np.linalg.norm(hard[boundary, :3, 3] - poses[boundary, :3, 3])
            ramped_jump = np.linalg.norm(ramped[boundary, :3, 3] - poses[boundary, :3, 3])
            if hard_enabled[boundary]:
                self.assertLess(ramped_jump, hard_jump)
            else:
                self.assertEqual(ramped_jump, 0.0)
        np.testing.assert_allclose(ramped[37:43, 0, 3], poses[37:43, 0, 3], atol=1e-12)

    def test_auto_time_falls_back_on_tail_recovery_timestamp_reset(self) -> None:
        rows = [
            {"frame_index": 10, "time_sec": 4.0},
            {"frame_index": 11, "time_sec": 4.03},
            {"frame_index": 12, "time_sec": 1.0},
        ]
        times, source = smooth._segment_times(rows, [0, 1, 2], "auto", 30.0)
        self.assertEqual(source, "frame-index-fallback")
        np.testing.assert_allclose(times, [0.0, 1.0 / 30.0, 2.0 / 30.0])

    def test_redundant_pose_fields_are_consistent(self) -> None:
        row = {
            "pose_world": None,
            "rotation_world": None,
            "translation_world_m": None,
            "translation_world_mm": None,
        }
        pose = np.eye(4)
        pose[:3, 3] = [0.1, -0.2, 0.3]
        smooth._update_pose_fields(row, pose)
        np.testing.assert_allclose(row["rotation_world"], pose[:3, :3])
        np.testing.assert_allclose(row["translation_world_m"], pose[:3, 3])
        np.testing.assert_allclose(row["translation_world_mm"], [100.0, -200.0, 300.0])


if __name__ == "__main__":
    unittest.main()
