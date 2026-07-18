"""Pure NumPy helpers for the TIFF phase viewer prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def discover_tiff_files(folder: Path) -> list[Path]:
    """Return top-level TIFF files in a stable, case-insensitive order."""

    if not folder.is_dir():
        raise ValueError(f"不是有效文件夹：{folder}")
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.name.casefold(),
    )


def scaled_capture_font_size(font_size: int, capture_scale: int) -> int:
    """Keep screen-space text visually consistent in supersampled captures."""

    if font_size < 1:
        raise ValueError("字体大小必须是正整数。")
    if capture_scale < 1:
        raise ValueError("截图倍率必须是正整数。")
    return int(font_size) * int(capture_scale)


@dataclass(frozen=True)
class VolumeInfo:
    """Summary information for a categorical 3-D volume."""

    shape_zyx: tuple[int, int, int]
    labels: tuple[int, ...]
    counts: tuple[int, ...]
    source_dtype: str

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.shape_zyx))

    @property
    def background_label(self) -> int:
        """Use the most common label as a practical background guess."""

        return self.labels[int(np.argmax(self.counts))]


def normalize_label_volume(array: np.ndarray) -> np.ndarray:
    """Return a contiguous int32 label volume in (Z, Y, X) order.

    Singleton dimensions are removed. Floating point input is accepted only
    when every finite value is effectively an integer. This deliberately
    rejects continuous greyscale images because interpolating phase IDs would
    create labels that are not physically present.
    """

    source = np.asarray(array)
    source = np.squeeze(source)

    if source.ndim != 3:
        raise ValueError(
            "需要三维标签矩阵；去除单例维度后的形状是 "
            f"{source.shape}，维数为 {source.ndim}。"
        )
    if source.size == 0:
        raise ValueError("体数据为空。")

    if np.issubdtype(source.dtype, np.floating):
        if not np.isfinite(source).all():
            raise ValueError("标签体中包含 NaN 或无穷值。")
        rounded = np.rint(source)
        if not np.allclose(source, rounded, rtol=0.0, atol=1e-6):
            raise ValueError(
                "检测到非整数灰度值；这个原型只接受离散相编号，"
                "不会对连续灰度自动分相。"
            )
        source = rounded
    elif not (
        np.issubdtype(source.dtype, np.integer)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise ValueError(f"不支持的数据类型：{source.dtype}")

    minimum = int(source.min())
    maximum = int(source.max())
    limits = np.iinfo(np.int32)
    if minimum < limits.min or maximum > limits.max:
        raise ValueError(
            f"标签范围 [{minimum}, {maximum}] 超出 int32；请先重新映射相编号。"
        )

    return np.ascontiguousarray(source, dtype=np.int32)


def analyze_volume(volume: np.ndarray, max_labels: int = 512) -> VolumeInfo:
    """Collect labels and counts, rejecting likely greyscale images."""

    if volume.ndim != 3:
        raise ValueError("analyze_volume 需要三维数组。")

    labels, counts = np.unique(volume, return_counts=True)
    if labels.size > max_labels:
        raise ValueError(
            f"发现 {labels.size} 个不同数值，超过原型上限 {max_labels}。"
            "这通常说明 TIFF 是灰度图而不是相标签图。"
        )

    return VolumeInfo(
        shape_zyx=tuple(int(v) for v in volume.shape),
        labels=tuple(int(v) for v in labels),
        counts=tuple(int(v) for v in counts),
        source_dtype=str(volume.dtype),
    )


def choose_default_phase(info: VolumeInfo) -> int:
    """Prefer a non-background phase, falling back to the only label."""

    background = info.background_label
    for label in info.labels:
        if label != background:
            return label
    return background


def crop_label_volume(
    volume: np.ndarray,
    bounds_xyz: Iterable[int],
) -> np.ndarray:
    """Crop a labelled volume with zero-based, half-open XYZ bounds.

    ``bounds_xyz`` is ordered as ``(x_start, x_stop, y_start, y_stop,
    z_start, z_stop)``. The returned array keeps the application's native
    ``(Z, Y, X)`` storage order and is C-contiguous so it can be passed to the
    VTK conversion helpers without another layout conversion.
    """

    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("裁剪操作需要非空 (Z, Y, X) 三维矩阵。")

    bounds = tuple(bounds_xyz)
    if len(bounds) != 6:
        raise ValueError("裁剪范围必须包含 X/Y/Z 三个方向的起止索引。")
    if any(not isinstance(value, (int, np.integer)) for value in bounds):
        raise ValueError("裁剪范围必须使用整数体素索引。")

    x_start, x_stop, y_start, y_stop, z_start, z_stop = (
        int(value) for value in bounds
    )
    z_size, y_size, x_size = volume.shape
    valid = (
        0 <= x_start < x_stop <= x_size
        and 0 <= y_start < y_stop <= y_size
        and 0 <= z_start < z_stop <= z_size
    )
    if not valid:
        raise ValueError(
            "裁剪范围超出数据边界或产生空体积；"
            f"当前形状 Z×Y×X 为 {z_size}×{y_size}×{x_size}。"
        )

    cropped = volume[
        z_start:z_stop,
        y_start:y_stop,
        x_start:x_stop,
    ]
    return np.ascontiguousarray(cropped)


def downsample_labels(
    volume: np.ndarray,
    factor: int,
    spacing_xyz: Iterable[float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Nearest-neighbour subsample a label volume and update its spacing."""

    if factor not in (1, 2, 4, 8):
        raise ValueError("降采样倍数必须是 1、2、4 或 8。")

    spacing = tuple(float(v) for v in spacing_xyz)
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("spacing_xyz 必须包含三个正数。")

    sampled = np.ascontiguousarray(volume[::factor, ::factor, ::factor])
    scaled_spacing = tuple(v * factor for v in spacing)
    return sampled, scaled_spacing


def flatten_zyx_for_vtk(volume: np.ndarray) -> np.ndarray:
    """Flatten (Z, Y, X) so X is the fastest-changing VTK point index."""

    if volume.ndim != 3:
        raise ValueError("VTK 展平操作需要 (Z, Y, X) 三维数组。")
    return np.ascontiguousarray(volume).ravel(order="C")


def pad_label_volume(volume: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad a label volume with a safe, otherwise unused int32 label."""

    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("外边界填充需要非空三维标签体。")
    minimum = int(volume.min())
    maximum = int(volume.max())
    limits = np.iinfo(np.int32)
    if minimum > limits.min:
        padding_label = minimum - 1
    elif maximum < limits.max:
        padding_label = maximum + 1
    else:
        raise ValueError("无法为体数据选择安全的外边界填充值。")

    padded = np.pad(
        volume,
        1,
        mode="constant",
        constant_values=padding_label,
    )
    return np.ascontiguousarray(padded, dtype=np.int32), padding_label


def calculate_slice_volume_fraction(
    volume: np.ndarray,
    axis: str,
    numerator_label: int = 3,
    denominator_labels: Iterable[int] = (3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a label fraction independently on every X/Y/Z slice.

    Slice numbers are returned as a user-facing 1-based sequence. A slice with
    no denominator-label voxels receives NaN instead of an artificial zero.
    The input convention is always (Z, Y, X).
    """

    if volume.ndim != 3:
        raise ValueError("逐层体积分数计算需要 (Z, Y, X) 三维矩阵。")

    axis_name = axis.strip().upper()
    axis_map = {"X": 2, "Y": 1, "Z": 0}
    if axis_name not in axis_map:
        raise ValueError("方向必须是 X、Y 或 Z。")

    denominator = tuple(dict.fromkeys(int(v) for v in denominator_labels))
    if not denominator:
        raise ValueError("分母相编号列表不能为空。")

    slice_axis = axis_map[axis_name]
    slice_count = volume.shape[slice_axis]
    fractions = np.full(slice_count, np.nan, dtype=np.float64)
    numerator = int(numerator_label)

    # Process one 2-D slice at a time. This avoids allocating one or more
    # temporary boolean arrays as large as the complete 3-D volume.
    for index in range(slice_count):
        if slice_axis == 0:
            layer = volume[index, :, :]
        elif slice_axis == 1:
            layer = volume[:, index, :]
        else:
            layer = volume[:, :, index]
        numerator_count = int(np.count_nonzero(layer == numerator))
        denominator_count = sum(
            int(np.count_nonzero(layer == label)) for label in denominator
        )
        if denominator_count:
            fractions[index] = numerator_count / denominator_count

    slice_numbers = np.arange(1, fractions.size + 1, dtype=np.int64)
    return slice_numbers, fractions


def make_demo_volume(size: int = 96) -> np.ndarray:
    """Create a small three-phase categorical dataset for immediate testing."""

    if size < 32:
        raise ValueError("演示体尺寸至少为 32。")

    z, y, x = np.ogrid[:size, :size, :size]
    center = (size - 1) / 2.0
    labels = np.zeros((size, size, size), dtype=np.int32)

    sphere = (
        (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
        <= (0.34 * size) ** 2
    )
    labels[sphere] = 3

    radial_xy = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    torus = (radial_xy - 0.27 * size) ** 2 + (z - center) ** 2 <= (
        0.075 * size
    ) ** 2
    labels[torus] = 4

    ellipsoid = (
        ((x - (center - 0.12 * size)) / (0.15 * size)) ** 2
        + ((y - (center + 0.08 * size)) / (0.11 * size)) ** 2
        + ((z - center) / (0.20 * size)) ** 2
        <= 1.0
    )
    labels[ellipsoid] = 5

    return labels
