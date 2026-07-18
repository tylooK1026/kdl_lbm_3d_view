from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from volume_core import (
    analyze_volume,
    calculate_slice_volume_fraction,
    choose_default_phase,
    crop_label_volume,
    discover_tiff_files,
    downsample_labels,
    flatten_zyx_for_vtk,
    make_demo_volume,
    normalize_label_volume,
    pad_label_volume,
    scaled_capture_font_size,
)


class VolumeCoreTests(unittest.TestCase):
    def test_capture_font_size_scales_with_export_resolution(self) -> None:
        self.assertEqual(scaled_capture_font_size(22, 1), 22)
        self.assertEqual(scaled_capture_font_size(22, 2), 44)
        self.assertEqual(scaled_capture_font_size(22, 4), 88)

    def test_discover_tiff_files_is_top_level_case_insensitive_and_sorted(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("b.TIFF", "A.tif", "notes.csv", "c.png"):
                (folder / name).touch()
            nested = folder / "nested"
            nested.mkdir()
            (nested / "ignored.tif").touch()

            result = discover_tiff_files(folder)

            self.assertEqual([path.name for path in result], ["A.tif", "b.TIFF"])

    def test_normalize_integer_volume(self) -> None:
        source = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        result = normalize_label_volume(source)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertEqual(result.dtype, np.int32)
        self.assertTrue(result.flags.c_contiguous)

    def test_normalize_integer_like_float(self) -> None:
        source = np.array([[[0.0, 1.0], [2.0, 3.0]]])
        source = np.concatenate([source, source], axis=0)
        result = normalize_label_volume(source)
        np.testing.assert_array_equal(result, source.astype(np.int32))

    def test_reject_continuous_float(self) -> None:
        source = np.zeros((2, 2, 2), dtype=np.float32)
        source[0, 0, 0] = 0.25
        with self.assertRaises(ValueError):
            normalize_label_volume(source)

    def test_downsample_updates_spacing(self) -> None:
        source = np.arange(8 * 6 * 4).reshape(8, 6, 4)
        sampled, spacing = downsample_labels(source, 2, (0.5, 0.75, 1.5))
        self.assertEqual(sampled.shape, (4, 3, 2))
        self.assertEqual(spacing, (1.0, 1.5, 3.0))
        np.testing.assert_array_equal(sampled, source[::2, ::2, ::2])

    def test_crop_uses_xyz_bounds_and_keeps_zyx_layout(self) -> None:
        source = np.arange(4 * 5 * 6, dtype=np.int32).reshape(4, 5, 6)
        cropped = crop_label_volume(source, (1, 5, 1, 4, 1, 3))
        self.assertEqual(cropped.shape, (2, 3, 4))
        self.assertTrue(cropped.flags.c_contiguous)
        np.testing.assert_array_equal(cropped, source[1:3, 1:4, 1:5])

    def test_crop_rejects_empty_or_out_of_range_bounds(self) -> None:
        source = np.zeros((4, 5, 6), dtype=np.int32)
        invalid_bounds = (
            (2, 2, 0, 5, 0, 4),
            (-1, 3, 0, 5, 0, 4),
            (0, 7, 0, 5, 0, 4),
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    crop_label_volume(source, bounds)

    def test_slice_fraction_can_run_on_cropped_working_volume(self) -> None:
        source = np.empty((2, 2, 4), dtype=np.int32)
        source[:, :, 0] = 3
        source[:, :, 1] = 4
        source[:, :, 2:] = 5
        cropped = crop_label_volume(source, (0, 2, 0, 2, 0, 2))
        _, full_fractions = calculate_slice_volume_fraction(source, "Z")
        _, crop_fractions = calculate_slice_volume_fraction(cropped, "Z")
        np.testing.assert_allclose(full_fractions, [0.25, 0.25])
        np.testing.assert_allclose(crop_fractions, [0.5, 0.5])

    def test_vtk_flatten_keeps_x_fastest(self) -> None:
        source = np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(flatten_zyx_for_vtk(source), np.arange(12))

    def test_background_and_default_phase(self) -> None:
        source = np.zeros((4, 4, 4), dtype=np.int32)
        source[1:3, 1:3, 1:3] = 7
        info = analyze_volume(source)
        self.assertEqual(info.background_label, 0)
        self.assertEqual(choose_default_phase(info), 7)

    def test_demo_contains_three_material_phases(self) -> None:
        demo = make_demo_volume(48)
        self.assertEqual(demo.shape, (48, 48, 48))
        self.assertEqual(set(np.unique(demo)), {0, 3, 4, 5})

    def test_padding_closes_volume_with_unused_label(self) -> None:
        source = np.ones((3, 4, 5), dtype=np.int32)
        padded, padding_label = pad_label_volume(source)
        self.assertEqual(padded.shape, (5, 6, 7))
        self.assertNotEqual(padding_label, 1)
        self.assertTrue(np.all(padded[0] == padding_label))
        np.testing.assert_array_equal(padded[1:-1, 1:-1, 1:-1], source)

    def test_slice_fraction_on_z_axis(self) -> None:
        volume = np.array(
            [
                [[3, 3], [4, 5]],
                [[3, 0], [0, 0]],
            ],
            dtype=np.int32,
        )
        slices, fractions = calculate_slice_volume_fraction(volume, "Z")
        np.testing.assert_array_equal(slices, [1, 2])
        np.testing.assert_allclose(fractions, [0.5, 1.0])

    def test_slice_fraction_on_x_axis_and_empty_denominator(self) -> None:
        volume = np.array(
            [
                [[3, 0], [4, 0]],
                [[5, 0], [3, 0]],
            ],
            dtype=np.int32,
        )
        slices, fractions = calculate_slice_volume_fraction(volume, "X")
        np.testing.assert_array_equal(slices, [1, 2])
        self.assertAlmostEqual(fractions[0], 0.5)
        self.assertTrue(np.isnan(fractions[1]))


if __name__ == "__main__":
    unittest.main()
