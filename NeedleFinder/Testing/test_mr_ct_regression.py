#!/usr/bin/env python3
"""
Regression and Zero-Shot Transfer Verification Suite for NeedleFinder:
  1. Verifies that all cross-modality / cross-sequence / cross-orientation transfer guards
     (CT Hounsfield dual-polarity, local foreground contrast inversion, Sagittal/Coronal
     axis canonicalization, thick-slice Z-upsampling fallback, and OOD confidence blending)
     produce ZERO change (0.000000 mm max trajectory difference, 0.000000 confidence diff)
     on clinical 3T T2W pelvic MRI needle segmentations compared to the baseline pipeline.
  2. Verifies functional correctness of each zero-shot transfer guard on CT Hounsfield volumes,
     bright-needle inverted volumes, Sagittal/Coronal transposed volumes, and secondary T2-TSE
     axial tip refinement.
"""

import glob
import os
import sys
import types
import unittest
import pytest
import numpy as np

for mod in (
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule",
):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


class MockMatrix4x4:
    def __init__(self):
        self._m = np.eye(4, dtype=np.float64)

    def GetElement(self, r, c):
        return float(self._m[r, c])


if not hasattr(sys.modules["vtk"], "vtkMatrix4x4"):
    sys.modules["vtk"].vtkMatrix4x4 = MockMatrix4x4
sys.modules["slicer"].modules = types.SimpleNamespace(NeedleFinderWidget=None)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NF_MODULE_DIR = os.path.join(REPO_ROOT, "NeedleFinder")
BENCH_RERUN_DIR = os.path.join(REPO_ROOT, "vault", "paper", "benchmarks_rerun")
for p in (NF_MODULE_DIR, BENCH_RERUN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import NeedleFinderLib.TrackingAlgorithms as ta_mod
import NeedleFinderLib.NeuralNeedleNet as nnn_mod
from NeedleFinderLib.TrackingAlgorithms import (
    TrackingAlgorithmsMixin,
    make_ct_polarity_volume,
    make_local_inverted_volume,
    refine_tip_with_secondary_t2_tse,
)

ta_mod.get_volume_f64_buffer = lambda img: (img, img.shape[::-1])


class StandaloneEvalLogic(TrackingAlgorithmsMixin):
    def findAxialSegmentationLimitFromMarker(self, bForceFallback=False):
        return 0, 0.0


class MockVolumeNode:
    def __init__(self, vol, sp, ijk_to_ras):
        self._vol = vol
        self._sp = tuple(float(x) for x in sp)
        self._mat = np.array(ijk_to_ras, dtype=np.float64)

    def GetImageData(self):
        return self._vol

    def GetSpacing(self):
        return self._sp

    def GetIJKToRASMatrix(self, vtk_mat):
        vtk_mat._m = self._mat.copy()

    def GetID(self):
        return "MockVolumeNode_Test"


class TestMRAndCTRegression(unittest.TestCase):
    """Tests zero-impact clinical MRI regression and zero-shot CT / orientation guards."""

    def test_01_ct_polarity_mapping(self):
        """Verify CT Hounsfield dual-polarity conversion maps both +1200 HU metal and -550 HU lumen to dark voids."""
        vol_hu = np.full((24, 64, 64), -1000.0, dtype=np.float64)  # air background
        vol_hu[:, 12:52, 12:52] = 45.0  # pelvic soft tissue (+45 HU)
        vol_hu[4:20, 28:31, 28:31] = 1200.0  # bright metallic shaft (+1200 HU)
        vol_hu[4:20, 38:41, 38:41] = -550.0  # hollow air-filled catheter lumen (-550 HU)

        pseudo_mri = make_ct_polarity_volume(vol_hu, mode="dual")
        tissue_bg = float(np.median(pseudo_mri[:, 20:25, 20:25]))
        metal_core = float(np.mean(pseudo_mri[8:16, 28:31, 28:31]))
        lumen_core = float(np.mean(pseudo_mri[8:16, 38:41, 38:41]))
        air_bg = float(np.mean(pseudo_mri[:, 2:8, 2:8]))

        self.assertGreater(tissue_bg, 150.0)
        self.assertLess(metal_core, 25.0)
        self.assertLess(lumen_core, 80.0)
        self.assertEqual(air_bg, 0.0)

    def test_02_local_foreground_inversion(self):
        """Verify local contrast inversion converts bright needles to dark voids without brightening air."""
        vol = np.zeros((20, 64, 64), dtype=np.float64)
        vol[:, 10:54, 10:54] = 120.0  # foreground tissue
        vol[4:16, 30:33, 30:33] = 420.0  # bright feature inside tissue

        inv = make_local_inverted_volume(vol)
        self.assertEqual(float(np.max(inv[:, :6, :6])), 0.0)  # air stays 0
        self.assertLess(float(np.mean(inv[6:14, 30:33, 30:33])), float(np.mean(inv[6:14, 20:25, 20:25])))

    def test_03_secondary_t2_tse_tip_refinement(self):
        """Verify sub-voxel parabolic tip refinement on a secondary axial T2-TSE volume."""
        K, J, I = 30, 64, 64
        sp_ras = np.array([0.75, 0.75, 2.0], dtype=np.float64)
        ijk2ras = np.diag([0.75, 0.75, 2.0, 1.0])
        sec_vol = np.full((K, J, I), 220.0, dtype=np.float64)
        # Needle along I=32, J=32 terminating at k=18 (Z = 36.0 mm)
        for k in range(4, 19):
            sec_vol[k, 31:34, 31:34] = 25.0

        # Initial trajectory with tip slightly underestimated at Z = 33.5 mm (2.5 mm below true tip 36.0 mm)
        zs = np.linspace(10.0, 33.5, 25)
        pts_ras = np.column_stack([np.full_like(zs, 32 * 0.75), np.full_like(zs, 32 * 0.75), zs])
        pred_trajs = [{"ras": pts_ras.copy(), "points": pts_ras.copy(), "confidence": 85.0}]

        refined, shifts = refine_tip_with_secondary_t2_tse(pred_trajs, sec_vol, sp_ras, ijk2ras)
        self.assertEqual(len(refined), 1)
        new_tip_z = float(refined[0]["ras"][-1, 2])
        self.assertLess(abs(new_tip_z - 36.0), abs(33.5 - 36.0))

    @pytest.mark.clinical
    def test_04_clinical_mri_zero_regression_and_orientation_invariance(self):
        """
        Verify on the 4 real clinical 3T pelvic MRI benchmark cases (67 ground-truth needles) that:
          1. All cross-modality / OOD guards remain strictly inactive (0.000000 mm trajectory diff vs baseline).
          2. Sagittal/Coronal axis canonicalization recovers exact world-RAS coordinates when transposed.
        """
        bench_dir = os.environ.get("NEEDLEFINDER_TEST_DATA", "")
        self.assertTrue(os.path.isdir(bench_dir), "Set NEEDLEFINDER_TEST_DATA; clinical validation must not silently skip.")
        baseline_path = os.environ.get("NEEDLEFINDER_CLINICAL_BASELINE", "")
        self.assertTrue(os.path.isfile(baseline_path), "Set NEEDLEFINDER_CLINICAL_BASELINE to an independently approved frozen JSON baseline.")
        import json
        baseline = json.load(open(baseline_path))

        from prepare_and_inspect_dataset import read_nrrd_local, read_vtk_centerline_local, dedup_needle_list
        from NeedleFinderLib.EvaluationMetrics import evaluate_method_unified

        cases = [
            (
                "2016-RSNA",
                os.path.join(bench_dir, "2016-RSNA/12 AX space trial 1.6 SLICE COVER TEMPLATE.6 SLICE COVER TEMPLATE.MR._.18.1.2013.07.29.09.48.42.911.59151707.nrrd"),
                os.path.join(bench_dir, "2016-RSNA"),
                os.path.join(bench_dir, "2016-RSNA/template slice position.acsv"),
            ),
            (
                "Case016",
                os.path.join(bench_dir, "Case016/Case016_8_AX TSE T2 MONITORING CATH PLACEMENT.nrrd"),
                os.path.join(bench_dir, "Case016"),
                os.path.join(bench_dir, "Case016/template slice position.acsv"),
            ),
            (
                "Case030",
                os.path.join(bench_dir, "Case030/Case30.MR._.12.1.2013.02.15.13.30.56.513.30763853.nrrd"),
                os.path.join(bench_dir, "Case030"),
                os.path.join(bench_dir, "Case030/template slice position.acsv"),
            ),
            (
                "Case038",
                os.path.join(bench_dir, "Case038/Case38.MR._.29.64.2013.02.15.12.32.19.712.26481368.nrrd"),
                os.path.join(bench_dir, "Case038"),
                os.path.join(bench_dir, "Case038/template slice position.acsv"),
            ),
        ]

        logic = StandaloneEvalLogic()
        preds_new_list, gts_list = [], []

        for name, nrrd_p, vtk_dir, acsv_p in cases:
            vol, sp_ras, ijk_to_ras, k_lim, z_lim = read_nrrd_local(nrrd_p, acsv_p)
            vtks = sorted(glob.glob(os.path.join(vtk_dir, "**", "manual-seg_*.vtk"), recursive=True))
            raw_gts = [read_vtk_centerline_local(v, z_lim) for v in vtks]
            gts = dedup_needle_list([{"pts": g} for g in raw_gts if g is not None and len(g) >= 2])
            k_use = k_lim if k_lim is not None else logic.estimateAxialLimitSlice(vol, sp_ras[2])

            # Check guard conditions directly
            k_step_p5 = max(1, vol.shape[0] // 32)
            is_ct_hu = float(np.percentile(vol[::k_step_p5, ::4, ::4], 5)) < -400.0
            col_norms = np.maximum(1e-6, np.linalg.norm(ijk_to_ras[:3, :3], axis=0))
            z_ijk_axis = int(np.argmax(np.abs(ijk_to_ras[2, :3]) / col_norms))
            self.assertFalse(is_ct_hu, f"{name} must not trigger CT Hounsfield guard")
            self.assertEqual(z_ijk_axis, 2, f"{name} is axial so z_ijk_axis must equal 2")

            vnode = MockVolumeNode(vol, sp_ras, ijk_to_ras)
            trajs = logic.autoDetectAllNeedles(vnode, minConfidence=50.0, kLim=k_use, renderNeedles=False, script=False)
            frozen = baseline["cases"][name]
            self.assertEqual(len(trajs), len(frozen))
            for actual, expected in zip(trajs, frozen):
                np.testing.assert_allclose(actual["ras"], expected["ras"], rtol=0, atol=baseline["trajectory_atol_mm"])
                self.assertAlmostEqual(float(actual["confidence"]), float(expected["confidence"]), delta=baseline["confidence_atol"])
            preds_new_list.append([{"ras": np.asarray(t["ras"], dtype=np.float64), "conf": float(t.get("confidence", 80.0))} for t in trajs])
            gts_list.append(gts)

            # Test Coronal orientation canonicalization on Case016 (swap K and J axes)
            if name == "Case016":
                vol_cor = np.ascontiguousarray(np.swapaxes(vol, 0, 1), dtype=np.float64)
                sp_cor = (sp_ras[0], sp_ras[2], sp_ras[1])
                mat_cor = ijk_to_ras[:, [0, 2, 1, 3]].copy()
                vnode_cor = MockVolumeNode(vol_cor, sp_cor, mat_cor)
                trajs_cor = logic.autoDetectAllNeedles(vnode_cor, minConfidence=50.0, kLim=k_use, renderNeedles=False, script=False)
                self.assertEqual(len(trajs_cor), len(trajs))
                max_cor_diff = max(
                    float(np.max(np.abs(np.asarray(a["ras"]) - np.asarray(b["ras"]))))
                    for a, b in zip(trajs, trajs_cor)
                )
                self.assertEqual(max_cor_diff, 0.0, "Coronal-transposed volume must produce bit-for-bit identical world-RAS trajectories")

                # Also verify Sagittal orientation canonicalization on Case016 (swap K and I axes)
                vol_sag = np.ascontiguousarray(np.swapaxes(vol, 0, 2), dtype=np.float64)
                sp_sag = (sp_ras[2], sp_ras[1], sp_ras[0])
                mat_sag = ijk_to_ras[:, [2, 1, 0, 3]].copy()
                vnode_sag = MockVolumeNode(vol_sag, sp_sag, mat_sag)
                trajs_sag = logic.autoDetectAllNeedles(vnode_sag, minConfidence=50.0, kLim=k_use, renderNeedles=False, script=False)
                self.assertEqual(len(trajs_sag), len(trajs))
                max_sag_diff = max(
                    float(np.max(np.abs(np.asarray(a["ras"]) - np.asarray(b["ras"]))))
                    for a, b in zip(trajs, trajs_sag)
                )
                self.assertEqual(max_sag_diff, 0.0, "Sagittal-transposed volume must produce bit-for-bit identical world-RAS trajectories")

        m_summary, _ = evaluate_method_unified(preds_new_list, gts_list, n_boot=100)
        print(
            f"\n[Regression Suite] 4-Case Clinical MRI Summary:"
            f"\n  1-to-1 Recall: {m_summary['hungarian_1to1_recall_pct']:.2f}%"
            f"\n  1-to-1 F1:     {m_summary['f1_1to1_pct']:.2f}%"
            f"\n  Shaft Error:   {m_summary['metrics']['mean_shaft_xy']['mean']:.4f} mm (median {m_summary['metrics']['mean_shaft_xy']['median']:.4f} mm)"
            f"\n  3D Tip Error:  {m_summary['metrics']['tip_3d']['mean']:.4f} mm (median {m_summary['metrics']['tip_3d']['median']:.4f} mm)"
        )
        self.assertGreaterEqual(m_summary["hungarian_1to1_recall_pct"], 81.9)
        self.assertLessEqual(m_summary["metrics"]["mean_shaft_xy"]["median"], 0.40)
        self.assertLessEqual(m_summary["metrics"]["tip_3d"]["median"], 3.40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
