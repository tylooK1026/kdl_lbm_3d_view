#!/usr/bin/env python3
"""Multi-phase interactive Surface Nets viewer for labelled TIFF volumes."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from preset_store import PresetStore, PresetStoreError

from volume_core import (
    VolumeInfo,
    analyze_volume,
    calculate_slice_volume_fraction,
    crop_label_volume,
    discover_tiff_files,
    downsample_labels,
    flatten_zyx_for_vtk,
    make_demo_volume,
    normalize_label_volume,
    pad_label_volume,
    scaled_capture_font_size,
)


INSTALL_HELP = """
缺少运行依赖：{name}

请在本程序目录执行：
  python -m pip install -r requirements.txt

然后运行：
  python phase_viewer.py --demo
""".strip()


try:
    import tifffile
    from PySide6.QtCore import (
        QEvent,
        QObject,
        QSignalBlocker,
        QStandardPaths,
        Qt,
        QTimer,
    )
    from PySide6.QtGui import QAction, QCloseEvent, QColor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDockWidget,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    from vtkmodules.qt.QVTKRenderWindowInteractor import (
        QVTKRenderWindowInteractor,
    )
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkSurfaceNets3D
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkInteractionWidgets import (
        vtkHandleWidget,
        vtkSphereHandleRepresentation,
    )
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkBillboardTextActor3D,
        vtkCamera,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkWindowToImageFilter,
    )

    import vtkmodules.vtkInteractionStyle  # noqa: F401
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(INSTALL_HELP.format(name=exc.name)) from exc


PALETTE = (
    (0.22, 0.64, 0.86),
    (0.96, 0.57, 0.18),
    (0.35, 0.76, 0.43),
    (0.86, 0.32, 0.34),
    (0.63, 0.45, 0.82),
    (0.33, 0.78, 0.75),
    (0.94, 0.77, 0.22),
    (0.88, 0.48, 0.69),
    (0.58, 0.65, 0.70),
    (0.70, 0.52, 0.31),
)

APP_VERSION = "1.0.0"
PRESET_SETTINGS_SCHEMA = 1


class ScrollControlGuard(QObject):
    """Route wheel gestures to a scroll panel instead of editing a control."""

    def __init__(self, scroll_area: QScrollArea, enabled) -> None:
        super().__init__(scroll_area)
        self.scroll_area = scroll_area
        self.enabled = enabled

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not self.enabled() or event.type() != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        if not isinstance(watched, (QAbstractSpinBox, QComboBox, QSlider)):
            return super().eventFilter(watched, event)

        vertical = self.scroll_area.verticalScrollBar()
        horizontal = self.scroll_area.horizontalScrollBar()
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        if pixel_delta.y():
            vertical.setValue(vertical.value() - pixel_delta.y())
        elif angle_delta.y():
            steps = angle_delta.y() / 120.0
            distance = max(vertical.singleStep() * 3, 36)
            vertical.setValue(vertical.value() - round(steps * distance))
        elif pixel_delta.x():
            horizontal.setValue(horizontal.value() - pixel_delta.x())
        elif angle_delta.x():
            steps = angle_delta.x() / 120.0
            distance = max(horizontal.singleStep() * 3, 36)
            horizontal.setValue(horizontal.value() - round(steps * distance))
        event.accept()
        return True


@dataclass
class PhaseStyle:
    label: int
    visible: bool
    color: tuple[float, float, float]
    opacity: float = 1.0
    smoothing: bool = True
    iterations: int = 20
    relaxation: float = 0.5
    max_move_voxels: float = 0.5
    lighting: bool = True
    interpolation: str = "Phong"
    ambient: float = 0.14
    diffuse: float = 0.82
    specular: float = 0.18
    specular_power: float = 18.0


@dataclass
class TextAnnotation:
    identifier: int
    text: str
    position_xyz: tuple[float, float, float]
    color: tuple[float, float, float]
    font_size: int = 22
    font_family: str = "Times New Roman"
    bold: bool = True
    italic: bool = False
    actor: vtkBillboardTextActor3D | None = None


class PhaseViewer(QMainWindow):
    """ParaView-inspired multi-phase comparison application."""

    def __init__(
        self,
        initial_path: Path | None = None,
        initial_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_demo: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"LBM_post_process V{APP_VERSION}")
        self.resize(1580, 920)

        self.source_volume: np.ndarray | None = None
        self.source_volume_info: VolumeInfo | None = None
        self.volume: np.ndarray | None = None
        self.volume_info: VolumeInfo | None = None
        self.crop_bounds_xyz = (0, 1, 0, 1, 0, 1)
        self.background_label = 0
        self.source_name = ""
        self.source_axes = ""
        self.current_tiff_path: Path | None = None
        self.phase_styles: dict[int, PhaseStyle] = {}
        self._phase_style_cache: dict[int, PhaseStyle] = {}
        self.phase_pipelines: dict[int, dict[str, object]] = {}
        self._row_for_label: dict[int, int] = {}
        self._color_buttons: dict[int, QPushButton] = {}
        self._opacity_spins: dict[int, QDoubleSpinBox] = {}
        self._visibility_checks: dict[int, QCheckBox] = {}
        self._smoothing_checks: dict[int, QCheckBox] = {}
        self._lighting_checks: dict[int, QCheckBox] = {}
        self._global_pipelines: list[object] = []
        self._vtk_image: vtkImageData | None = None
        self._padding_label = 0
        self._outline_actor: vtkActor | None = None
        self._updating_phase_ui = False
        self._property_color = PALETTE[0]
        self.background_color = (0.055, 0.065, 0.085)
        self.background_color_2 = (0.14, 0.16, 0.20)
        self.outline_color = (0.78, 0.82, 0.90)
        self.outline_width = 1.5
        self._render_spacing = initial_spacing
        self.annotations: dict[int, TextAnnotation] = {}
        self._next_annotation_id = 1
        self._annotation_color = (1.0, 0.82, 0.20)
        self.annotation_handle_widget: vtkHandleWidget | None = None
        self.annotation_handle_representation: (
            vtkSphereHandleRepresentation | None
        ) = None
        self._updating_annotation_handle = False
        self.batch_folder: Path | None = None
        self.batch_tiff_paths: list[Path] = []
        self.slice_fraction_results: tuple[np.ndarray, np.ndarray] | None = None
        self.slice_fraction_axis = "Z"
        config_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        if not config_directory:
            config_directory = str(Path.home() / ".tiff_phase_viewer")
        self.preset_store = PresetStore(
            Path(config_directory) / "visualization_presets.json"
        )
        self._scroll_control_guard: ScrollControlGuard | None = None

        self._build_central_view()
        self._build_phase_dock()
        self._build_properties_dock(initial_spacing)
        self._build_analysis_dock()
        self._build_results_dock()
        self._build_actions()
        self._setup_vtk()

        if initial_path is not None:
            QTimer.singleShot(0, lambda: self.load_tiff(initial_path))
        elif use_demo or initial_path is None:
            QTimer.singleShot(0, self.load_demo)

    def _build_central_view(self) -> None:
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)

    def _build_phase_dock(self) -> None:
        dock = QDockWidget("相列表 / Pipeline", self)
        dock.setObjectName("phaseDock")
        dock.setMinimumWidth(410)

        body = QWidget(dock)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        self.phase_table = QTableWidget(0, 6, body)
        self.phase_table.setHorizontalHeaderLabels(
            ["显示", "相编号", "颜色", "透明度", "平滑", "光照"]
        )
        self.phase_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.phase_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.phase_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.phase_table.verticalHeader().setVisible(False)
        header = self.phase_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.phase_table.itemSelectionChanged.connect(
            self._load_selected_phase_controls
        )
        layout.addWidget(self.phase_table, 1)

        buttons = QHBoxLayout()
        show_materials = QPushButton("非背景相")
        show_all = QPushButton("全部显示")
        hide_all = QPushButton("全部隐藏")
        show_materials.clicked.connect(
            lambda: self._set_visibility_group("materials")
        )
        show_all.clicked.connect(lambda: self._set_visibility_group("all"))
        hide_all.clicked.connect(lambda: self._set_visibility_group("none"))
        buttons.addWidget(show_materials)
        buttons.addWidget(show_all)
        buttons.addWidget(hide_all)
        layout.addLayout(buttons)

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.phase_dock = dock

    def _build_properties_dock(
        self,
        spacing: tuple[float, float, float],
    ) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("propertiesDock")
        dock.setMinimumWidth(340)

        scroll = QScrollArea(dock)
        scroll.setWidgetResizable(True)
        panel = QWidget(scroll)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        preset_group = QGroupBox("设置预设", panel)
        preset_layout = QVBoxLayout(preset_group)
        preset_form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.preset_combo.lineEdit().setPlaceholderText("输入名称或选择已保存预设")
        preset_form.addRow("预设名称", self.preset_combo)

        self.wheel_guard_checkbox = QCheckBox(
            "右侧滚动防误触（滚轮不修改参数）"
        )
        self.wheel_guard_checkbox.setChecked(True)
        preset_form.addRow(self.wheel_guard_checkbox)
        preset_layout.addLayout(preset_form)

        preset_buttons = QHBoxLayout()
        save_preset_button = QPushButton("保存 / 覆盖")
        load_preset_button = QPushButton("加载")
        delete_preset_button = QPushButton("删除")
        save_preset_button.clicked.connect(self.save_named_preset)
        load_preset_button.clicked.connect(self.load_named_preset)
        delete_preset_button.clicked.connect(self.delete_named_preset)
        preset_buttons.addWidget(save_preset_button)
        preset_buttons.addWidget(load_preset_button)
        preset_buttons.addWidget(delete_preset_button)
        preset_layout.addLayout(preset_buttons)

        preset_note = QLabel(
            "预设会在关闭程序后保留，可恢复相样式、裁剪、边框、背景、"
            "3D 标注、相机角度、体素尺寸、截图与计算参数。"
        )
        preset_note.setWordWrap(True)
        preset_layout.addWidget(preset_note)
        layout.addWidget(preset_group)
        self._refresh_preset_combo()

        data_group = QGroupBox("数据与全局显示", panel)
        data_layout = QVBoxLayout(data_group)
        self.info_label = QLabel("尚未载入数据")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        data_layout.addWidget(self.info_label)

        data_form = QFormLayout()
        self.spacing_spins: list[QDoubleSpinBox] = []
        for axis, value in zip("XYZ", spacing):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0.000001, 1_000_000.0)
            spin.setValue(value)
            spin.setSuffix(f"  ({axis})")
            self.spacing_spins.append(spin)
            data_form.addRow(f"{axis} 体素尺寸", spin)

        self.downsample_combo = QComboBox()
        for factor in (1, 2, 4, 8):
            self.downsample_combo.addItem(f"{factor}×", factor)
        data_form.addRow("预览降采样", self.downsample_combo)

        self.gradient_checkbox = QCheckBox("使用渐变背景")
        self.gradient_checkbox.setChecked(True)
        self.gradient_checkbox.toggled.connect(self.update_background)
        data_form.addRow(self.gradient_checkbox)

        self.background_button = QPushButton()
        self.background_button.clicked.connect(
            partial(self.choose_background_color, False)
        )
        self.background_button_2 = QPushButton()
        self.background_button_2.clicked.connect(
            partial(self.choose_background_color, True)
        )
        self._update_color_button(
            self.background_button,
            self.background_color,
            "选择下方背景色",
        )
        self._update_color_button(
            self.background_button_2,
            self.background_color_2,
            "选择上方背景色",
        )
        data_form.addRow("背景色", self.background_button)
        data_form.addRow("渐变色", self.background_button_2)

        self.outline_checkbox = QCheckBox("显示当前工作区域边框")
        self.outline_checkbox.setChecked(True)
        self.outline_checkbox.toggled.connect(self.update_outline_style)
        data_form.addRow(self.outline_checkbox)

        self.outline_color_button = QPushButton()
        self.outline_color_button.clicked.connect(self.choose_outline_color)
        self._update_color_button(
            self.outline_color_button,
            self.outline_color,
            "选择模型边框颜色",
        )
        data_form.addRow("边框颜色", self.outline_color_button)

        self.outline_width_spin = QDoubleSpinBox()
        self.outline_width_spin.setDecimals(1)
        self.outline_width_spin.setRange(0.5, 12.0)
        self.outline_width_spin.setSingleStep(0.5)
        self.outline_width_spin.setValue(self.outline_width)
        self.outline_width_spin.valueChanged.connect(self.update_outline_style)
        data_form.addRow("边框粗细", self.outline_width_spin)

        self.edges_checkbox = QCheckBox("显示三角网格边")
        self.edges_checkbox.toggled.connect(self.update_actor_styles)
        data_form.addRow(self.edges_checkbox)
        data_layout.addLayout(data_form)

        rebuild_all_button = QPushButton("重建所有可见相")
        rebuild_all_button.setMinimumHeight(38)
        rebuild_all_button.clicked.connect(self.rebuild_all)
        data_layout.addWidget(rebuild_all_button)
        layout.addWidget(data_group)

        crop_group = QGroupBox("裁剪工作区域", panel)
        crop_layout = QVBoxLayout(crop_group)
        self.crop_shape_label = QLabel("载入数据后可设置裁剪范围")
        self.crop_shape_label.setWordWrap(True)
        crop_layout.addWidget(self.crop_shape_label)

        crop_form = QFormLayout()
        self.crop_start_spins: dict[str, QSpinBox] = {}
        self.crop_end_spins: dict[str, QSpinBox] = {}
        for axis in "XYZ":
            row = QHBoxLayout()
            start_spin = QSpinBox()
            start_spin.setRange(1, 1)
            end_spin = QSpinBox()
            end_spin.setRange(1, 1)
            row.addWidget(QLabel("起始"))
            row.addWidget(start_spin)
            row.addWidget(QLabel("结束"))
            row.addWidget(end_spin)
            self.crop_start_spins[axis] = start_spin
            self.crop_end_spins[axis] = end_spin
            crop_form.addRow(f"{axis} 索引", row)
        crop_layout.addLayout(crop_form)

        crop_buttons = QHBoxLayout()
        apply_crop_button = QPushButton("应用裁剪")
        apply_crop_button.setMinimumHeight(36)
        apply_crop_button.clicked.connect(self.apply_crop)
        reset_crop_button = QPushButton("恢复完整范围")
        reset_crop_button.clicked.connect(self.reset_crop)
        crop_buttons.addWidget(apply_crop_button)
        crop_buttons.addWidget(reset_crop_button)
        crop_layout.addLayout(crop_buttons)

        crop_note = QLabel(
            "索引从 1 开始，结束值包含在裁剪范围内。应用后，三维显示、"
            "截图和逐层体积分数都使用裁剪后的工作体积。"
        )
        crop_note.setWordWrap(True)
        crop_layout.addWidget(crop_note)
        layout.addWidget(crop_group)

        batch_group = QGroupBox("TIFF 文件夹批量处理", panel)
        batch_layout = QVBoxLayout(batch_group)
        self.batch_folder_label = QLabel("尚未选择 TIFF 文件夹")
        self.batch_folder_label.setWordWrap(True)
        self.batch_folder_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        batch_layout.addWidget(self.batch_folder_label)

        choose_folder_button = QPushButton("选择 TIFF 文件夹")
        choose_folder_button.clicked.connect(self.choose_tiff_folder)
        batch_layout.addWidget(choose_folder_button)

        batch_form = QFormLayout()
        self.batch_reference_combo = QComboBox()
        self.batch_reference_combo.setEnabled(False)
        batch_form.addRow("参考 TIFF", self.batch_reference_combo)
        batch_layout.addLayout(batch_form)

        load_reference_button = QPushButton("载入选中的参考 TIFF")
        load_reference_button.clicked.connect(self.load_selected_batch_reference)
        batch_layout.addWidget(load_reference_button)

        run_batch_button = QPushButton("批量输出 PNG 和体积分数 CSV")
        run_batch_button.setMinimumHeight(38)
        run_batch_button.clicked.connect(self.run_batch_folder)
        batch_layout.addWidget(run_batch_button)

        batch_note = QLabel(
            "先在参考 TIFF 上完成裁剪、相样式、相机和计算公式设置。批处理会将"
            "同一套设置应用于文件夹顶层全部 TIFF，并检查原始矩阵形状。"
        )
        batch_note.setWordWrap(True)
        batch_layout.addWidget(batch_note)
        layout.addWidget(batch_group)

        phase_group = QGroupBox("当前相属性", panel)
        phase_form = QFormLayout(phase_group)
        self.selected_phase_label = QLabel("未选择")
        phase_form.addRow("相编号", self.selected_phase_label)

        self.selected_visible_checkbox = QCheckBox("显示当前相")
        phase_form.addRow(self.selected_visible_checkbox)

        self.selected_color_button = QPushButton()
        self.selected_color_button.clicked.connect(
            self.choose_selected_phase_color
        )
        self._update_color_button(
            self.selected_color_button,
            self._property_color,
            "选择当前相颜色",
        )
        phase_form.addRow("表面颜色", self.selected_color_button)

        self.selected_opacity_spin = QDoubleSpinBox()
        self.selected_opacity_spin.setDecimals(2)
        self.selected_opacity_spin.setRange(0.0, 1.0)
        self.selected_opacity_spin.setSingleStep(0.05)
        self.selected_opacity_spin.setValue(1.0)
        phase_form.addRow("透明度", self.selected_opacity_spin)

        self.selected_smoothing_checkbox = QCheckBox("启用平滑")
        self.selected_smoothing_checkbox.setChecked(True)
        phase_form.addRow(self.selected_smoothing_checkbox)

        self.selected_iterations_spin = QSpinBox()
        self.selected_iterations_spin.setRange(1, 100)
        self.selected_iterations_spin.setValue(20)
        phase_form.addRow("迭代次数", self.selected_iterations_spin)

        self.selected_relaxation_spin = QDoubleSpinBox()
        self.selected_relaxation_spin.setDecimals(3)
        self.selected_relaxation_spin.setRange(0.01, 1.0)
        self.selected_relaxation_spin.setSingleStep(0.05)
        self.selected_relaxation_spin.setValue(0.5)
        phase_form.addRow("松弛因子", self.selected_relaxation_spin)

        self.selected_constraint_spin = QDoubleSpinBox()
        self.selected_constraint_spin.setDecimals(3)
        self.selected_constraint_spin.setRange(0.0, 2.0)
        self.selected_constraint_spin.setSingleStep(0.1)
        self.selected_constraint_spin.setValue(0.5)
        self.selected_constraint_spin.setSuffix(" 体素")
        phase_form.addRow("最大位移", self.selected_constraint_spin)

        self.selected_lighting_checkbox = QCheckBox("启用光照")
        self.selected_lighting_checkbox.setChecked(True)
        phase_form.addRow(self.selected_lighting_checkbox)

        self.selected_interpolation_combo = QComboBox()
        self.selected_interpolation_combo.addItems(["Flat", "Gouraud", "Phong"])
        self.selected_interpolation_combo.setCurrentText("Phong")
        phase_form.addRow("表面插值", self.selected_interpolation_combo)

        self.selected_ambient_spin = QDoubleSpinBox()
        self.selected_ambient_spin.setDecimals(2)
        self.selected_ambient_spin.setRange(0.0, 1.0)
        self.selected_ambient_spin.setSingleStep(0.05)
        self.selected_ambient_spin.setValue(0.14)
        phase_form.addRow("Ambient", self.selected_ambient_spin)

        self.selected_diffuse_spin = QDoubleSpinBox()
        self.selected_diffuse_spin.setDecimals(2)
        self.selected_diffuse_spin.setRange(0.0, 1.0)
        self.selected_diffuse_spin.setSingleStep(0.05)
        self.selected_diffuse_spin.setValue(0.82)
        phase_form.addRow("Diffuse", self.selected_diffuse_spin)

        self.selected_specular_spin = QDoubleSpinBox()
        self.selected_specular_spin.setDecimals(2)
        self.selected_specular_spin.setRange(0.0, 1.0)
        self.selected_specular_spin.setSingleStep(0.05)
        self.selected_specular_spin.setValue(0.18)
        phase_form.addRow("Specular", self.selected_specular_spin)

        self.selected_specular_power_spin = QDoubleSpinBox()
        self.selected_specular_power_spin.setDecimals(1)
        self.selected_specular_power_spin.setRange(1.0, 128.0)
        self.selected_specular_power_spin.setSingleStep(1.0)
        self.selected_specular_power_spin.setValue(18.0)
        phase_form.addRow("Specular Power", self.selected_specular_power_spin)

        apply_phase_button = QPushButton("应用并重建当前相")
        apply_phase_button.setMinimumHeight(38)
        apply_phase_button.clicked.connect(self.apply_selected_phase)
        phase_form.addRow(apply_phase_button)
        layout.addWidget(phase_group)

        annotation_group = QGroupBox("3D 文字区域标注", panel)
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_form = QFormLayout()
        self.annotation_text_edit = QLineEdit()
        self.annotation_text_edit.setPlaceholderText("例如：孔隙富集区 A")
        annotation_form.addRow("标注文字", self.annotation_text_edit)

        self.annotation_position_spins: dict[str, QDoubleSpinBox] = {}
        for axis in "XYZ":
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            spin.setSingleStep(1.0)
            self.annotation_position_spins[axis] = spin
            annotation_form.addRow(f"{axis} 模型坐标", spin)

        center_button = QPushButton("使用当前裁剪区域中心")
        center_button.clicked.connect(self.use_crop_center_for_annotation)
        annotation_form.addRow(center_button)

        self.annotation_color_button = QPushButton()
        self.annotation_color_button.clicked.connect(self.choose_annotation_color)
        self._update_color_button(
            self.annotation_color_button,
            self._annotation_color,
            "选择 3D 文字颜色",
        )
        annotation_form.addRow("文字颜色", self.annotation_color_button)

        self.annotation_font_size_spin = QSpinBox()
        self.annotation_font_size_spin.setRange(8, 96)
        self.annotation_font_size_spin.setValue(22)
        annotation_form.addRow("文字大小", self.annotation_font_size_spin)

        self.annotation_font_family_combo = QComboBox()
        self.annotation_font_family_combo.addItems(
            ["Times New Roman", "Arial", "Courier New"]
        )
        annotation_form.addRow("英文字体", self.annotation_font_family_combo)

        font_style_row = QHBoxLayout()
        self.annotation_bold_checkbox = QCheckBox("加粗")
        self.annotation_bold_checkbox.setChecked(True)
        self.annotation_italic_checkbox = QCheckBox("斜体")
        font_style_row.addWidget(self.annotation_bold_checkbox)
        font_style_row.addWidget(self.annotation_italic_checkbox)
        annotation_form.addRow("字体样式", font_style_row)

        self.annotation_handle_checkbox = QCheckBox("显示所选标注的拖动手柄")
        self.annotation_handle_checkbox.setChecked(True)
        self.annotation_handle_checkbox.toggled.connect(
            self._sync_annotation_handle
        )
        annotation_form.addRow(self.annotation_handle_checkbox)
        annotation_layout.addLayout(annotation_form)

        annotation_buttons = QHBoxLayout()
        add_annotation_button = QPushButton("新增")
        update_annotation_button = QPushButton("更新")
        remove_annotation_button = QPushButton("删除")
        add_annotation_button.clicked.connect(self.add_annotation)
        update_annotation_button.clicked.connect(self.update_annotation)
        remove_annotation_button.clicked.connect(self.remove_annotation)
        annotation_buttons.addWidget(add_annotation_button)
        annotation_buttons.addWidget(update_annotation_button)
        annotation_buttons.addWidget(remove_annotation_button)
        annotation_layout.addLayout(annotation_buttons)

        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(110)
        self.annotation_list.currentItemChanged.connect(
            self.load_selected_annotation
        )
        annotation_layout.addWidget(self.annotation_list)

        clear_annotations_button = QPushButton("清空全部标注")
        clear_annotations_button.clicked.connect(self.clear_annotations)
        annotation_layout.addWidget(clear_annotations_button)

        annotation_note = QLabel(
            "坐标使用自由模型坐标，可以输入负数。选择标注后拖动文字位置处的"
            "彩色球形手柄，X/Y/Z 会实时回写；文字始终朝向相机。"
        )
        annotation_note.setWordWrap(True)
        annotation_layout.addWidget(annotation_note)
        layout.addWidget(annotation_group)

        view_group = QGroupBox("视图", panel)
        view_layout = QFormLayout(view_group)
        reset_button = QPushButton("重置相机")
        reset_button.clicked.connect(self.reset_camera)
        self.screenshot_scale_combo = QComboBox()
        self.screenshot_scale_combo.addItem("1× 当前分辨率", 1)
        self.screenshot_scale_combo.addItem("2× 高清", 2)
        self.screenshot_scale_combo.addItem("4× 超清", 4)
        self.screenshot_scale_combo.setCurrentIndex(1)
        screenshot_button = QPushButton("保存模型截图")
        screenshot_button.clicked.connect(self.save_screenshot)
        view_layout.addRow(reset_button)
        view_layout.addRow("截图清晰度", self.screenshot_scale_combo)
        view_layout.addRow(screenshot_button)
        layout.addWidget(view_group)

        note = QLabel(
            "中央仅显示各相设置后的平滑表面。颜色和透明度立即生效；"
            "几何平滑参数需要重建。截图只包含模型渲染区域。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        scroll.setWidget(panel)
        self._scroll_control_guard = ScrollControlGuard(
            scroll,
            lambda: self.wheel_guard_checkbox.isChecked(),
        )
        for control_type in (QAbstractSpinBox, QComboBox, QSlider):
            for control in panel.findChildren(control_type):
                control.installEventFilter(self._scroll_control_guard)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.properties_dock = dock

    def _refresh_preset_combo(self, selected_name: str = "") -> None:
        try:
            names = self.preset_store.names()
        except PresetStoreError as exc:
            names = []
            self.statusBar().showMessage(str(exc))
        blocker = QSignalBlocker(self.preset_combo)
        current_text = selected_name or self.preset_combo.currentText().strip()
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        self.preset_combo.setCurrentText(current_text)
        del blocker

    @staticmethod
    def _phase_style_settings(style: PhaseStyle) -> dict[str, object]:
        return {
            "visible": style.visible,
            "color": list(style.color),
            "opacity": style.opacity,
            "smoothing": style.smoothing,
            "iterations": style.iterations,
            "relaxation": style.relaxation,
            "max_move_voxels": style.max_move_voxels,
            "lighting": style.lighting,
            "interpolation": style.interpolation,
            "ambient": style.ambient,
            "diffuse": style.diffuse,
            "specular": style.specular,
            "specular_power": style.specular_power,
        }

    @staticmethod
    def _color_from_settings(value: object, field: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"预设中的{field}无效。")
        color = tuple(float(component) for component in value)
        if any(component < 0.0 or component > 1.0 for component in color):
            raise ValueError(f"预设中的{field}超出 0–1 范围。")
        return color

    @classmethod
    def _phase_style_from_settings(
        cls,
        label: int,
        payload: object,
        fallback: PhaseStyle,
    ) -> PhaseStyle:
        if not isinstance(payload, dict):
            raise ValueError(f"预设中相 {label} 的样式无效。")
        interpolation = str(payload.get("interpolation", fallback.interpolation))
        if interpolation not in {"Flat", "Gouraud", "Phong"}:
            raise ValueError(f"预设中相 {label} 的插值方式无效。")
        style = PhaseStyle(
            label=label,
            visible=bool(payload.get("visible", fallback.visible)),
            color=cls._color_from_settings(
                payload.get("color", list(fallback.color)),
                f"相 {label} 颜色",
            ),
            opacity=float(payload.get("opacity", fallback.opacity)),
            smoothing=bool(payload.get("smoothing", fallback.smoothing)),
            iterations=int(payload.get("iterations", fallback.iterations)),
            relaxation=float(payload.get("relaxation", fallback.relaxation)),
            max_move_voxels=float(
                payload.get("max_move_voxels", fallback.max_move_voxels)
            ),
            lighting=bool(payload.get("lighting", fallback.lighting)),
            interpolation=interpolation,
            ambient=float(payload.get("ambient", fallback.ambient)),
            diffuse=float(payload.get("diffuse", fallback.diffuse)),
            specular=float(payload.get("specular", fallback.specular)),
            specular_power=float(
                payload.get("specular_power", fallback.specular_power)
            ),
        )
        if not 0.0 <= style.opacity <= 1.0:
            raise ValueError(f"预设中相 {label} 的透明度无效。")
        if not 1 <= style.iterations <= 100:
            raise ValueError(f"预设中相 {label} 的平滑迭代次数无效。")
        if not 0.01 <= style.relaxation <= 1.0:
            raise ValueError(f"预设中相 {label} 的松弛因子无效。")
        if not 0.0 <= style.max_move_voxels <= 2.0:
            raise ValueError(f"预设中相 {label} 的最大位移无效。")
        if any(
            not 0.0 <= value <= 1.0
            for value in (style.ambient, style.diffuse, style.specular)
        ):
            raise ValueError(f"预设中相 {label} 的光照参数无效。")
        if not 1.0 <= style.specular_power <= 128.0:
            raise ValueError(f"预设中相 {label} 的高光指数无效。")
        return style

    def _camera_settings(self) -> dict[str, object]:
        camera = self.renderer.GetActiveCamera()
        return {
            "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()),
            "view_up": list(camera.GetViewUp()),
            "view_angle": float(camera.GetViewAngle()),
            "parallel_projection": bool(camera.GetParallelProjection()),
            "parallel_scale": float(camera.GetParallelScale()),
            "clipping_range": list(camera.GetClippingRange()),
            "window_center": list(camera.GetWindowCenter()),
        }

    @staticmethod
    def _camera_vector(
        payload: dict[str, object],
        key: str,
        length: int,
    ) -> tuple[float, ...]:
        value = payload.get(key)
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"预设中的相机参数 {key} 无效。")
        return tuple(float(component) for component in value)

    def _restore_camera_settings(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("预设中的相机设置无效。")
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*self._camera_vector(payload, "position", 3))
        camera.SetFocalPoint(*self._camera_vector(payload, "focal_point", 3))
        camera.SetViewUp(*self._camera_vector(payload, "view_up", 3))
        camera.SetViewAngle(float(payload.get("view_angle", 30.0)))
        camera.SetParallelProjection(bool(payload.get("parallel_projection", False)))
        camera.SetParallelScale(float(payload.get("parallel_scale", 1.0)))
        camera.SetClippingRange(*self._camera_vector(payload, "clipping_range", 2))
        camera.SetWindowCenter(*self._camera_vector(payload, "window_center", 2))

    def _capture_preset_settings(self) -> dict[str, object]:
        if self.source_volume_info is None:
            raise ValueError("请先打开一个 TIFF 或演示体，再保存设置预设。")
        annotations = []
        for annotation in sorted(
            self.annotations.values(), key=lambda item: item.identifier
        ):
            annotations.append(
                {
                    "identifier": annotation.identifier,
                    "text": annotation.text,
                    "position_xyz": list(annotation.position_xyz),
                    "color": list(annotation.color),
                    "font_size": annotation.font_size,
                    "font_family": annotation.font_family,
                    "bold": annotation.bold,
                    "italic": annotation.italic,
                }
            )
        return {
            "schema_version": PRESET_SETTINGS_SCHEMA,
            "application_version": APP_VERSION,
            "source_shape_zyx": list(self.source_volume_info.shape_zyx),
            "spacing_xyz": list(self.current_spacing()),
            "downsample_factor": int(self.downsample_combo.currentData()),
            "background": {
                "gradient": self.gradient_checkbox.isChecked(),
                "color": list(self.background_color),
                "color_2": list(self.background_color_2),
            },
            "outline": {
                "visible": self.outline_checkbox.isChecked(),
                "color": list(self.outline_color),
                "width": self.outline_width_spin.value(),
            },
            "show_mesh_edges": self.edges_checkbox.isChecked(),
            "crop_bounds_xyz": list(self.crop_bounds_xyz),
            "phase_styles": {
                str(label): self._phase_style_settings(style)
                for label, style in self._phase_style_cache.items()
            },
            "selected_phase": self.selected_label(),
            "annotations": annotations,
            "selected_annotation": self._selected_annotation_id(),
            "annotation_editor": {
                "text": self.annotation_text_edit.text(),
                "position_xyz": [
                    self.annotation_position_spins[axis].value()
                    for axis in "XYZ"
                ],
                "color": list(self._annotation_color),
                "font_size": self.annotation_font_size_spin.value(),
                "font_family": self.annotation_font_family_combo.currentText(),
                "bold": self.annotation_bold_checkbox.isChecked(),
                "italic": self.annotation_italic_checkbox.isChecked(),
                "show_handle": self.annotation_handle_checkbox.isChecked(),
            },
            "camera": self._camera_settings(),
            "screenshot_scale": int(self.screenshot_scale_combo.currentData()),
            "slice_fraction": {
                "axis": self.fraction_axis_combo.currentText(),
                "numerator": self.numerator_label_spin.value(),
                "denominator": self.denominator_labels_edit.text(),
            },
            "wheel_guard": self.wheel_guard_checkbox.isChecked(),
        }

    @staticmethod
    def _set_combo_to_data(combo: QComboBox, value: object, field: str) -> None:
        index = combo.findData(value)
        if index < 0:
            raise ValueError(f"预设中的{field}无效。")
        combo.setCurrentIndex(index)

    def _validate_crop_settings(
        self, value: object
    ) -> tuple[int, int, int, int, int, int]:
        if self.source_volume_info is None:
            raise ValueError("请先打开 TIFF，再加载设置预设。")
        if not isinstance(value, list) or len(value) != 6:
            raise ValueError("预设中的裁剪范围无效。")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("预设中的裁剪范围必须是整数。")
        bounds = tuple(int(item) for item in value)
        z_size, y_size, x_size = self.source_volume_info.shape_zyx
        x_start, x_stop, y_start, y_stop, z_start, z_stop = bounds
        if not (
            0 <= x_start < x_stop <= x_size
            and 0 <= y_start < y_stop <= y_size
            and 0 <= z_start < z_stop <= z_size
        ):
            raise ValueError(
                "预设裁剪范围不适合当前 TIFF；当前原始形状 Z×Y×X 为 "
                f"{z_size}×{y_size}×{x_size}。"
            )
        return bounds

    def _restore_annotations(self, payload: object) -> None:
        if not isinstance(payload, list):
            raise ValueError("预设中的 3D 标注列表无效。")
        self._clear_annotations_without_confirmation()
        blocker = QSignalBlocker(self.annotation_list)
        maximum_identifier = 0
        for entry in payload:
            if not isinstance(entry, dict):
                raise ValueError("预设中包含无效的 3D 标注。")
            identifier = int(entry.get("identifier", maximum_identifier + 1))
            position_value = entry.get("position_xyz")
            if not isinstance(position_value, list) or len(position_value) != 3:
                raise ValueError("预设中的 3D 标注坐标无效。")
            font_family = str(entry.get("font_family", "Times New Roman"))
            if font_family not in {"Times New Roman", "Arial", "Courier New"}:
                raise ValueError("预设中的 3D 标注字体无效。")
            annotation = TextAnnotation(
                identifier=identifier,
                text=str(entry.get("text", "")),
                position_xyz=tuple(float(value) for value in position_value),
                color=self._color_from_settings(
                    entry.get("color", [1.0, 0.82, 0.20]),
                    "3D 标注颜色",
                ),
                font_size=int(entry.get("font_size", 22)),
                font_family=font_family,
                bold=bool(entry.get("bold", True)),
                italic=bool(entry.get("italic", False)),
            )
            if not annotation.text.strip():
                raise ValueError("预设中的 3D 标注文字不能为空。")
            if not 8 <= annotation.font_size <= 96:
                raise ValueError("预设中的 3D 标注字号无效。")
            if identifier in self.annotations or identifier < 1:
                raise ValueError("预设中的 3D 标注编号重复或无效。")
            self.annotations[identifier] = annotation
            item = QListWidgetItem(self._annotation_item_text(annotation))
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            self.annotation_list.addItem(item)
            maximum_identifier = max(maximum_identifier, identifier)
        self._next_annotation_id = maximum_identifier + 1
        del blocker

    def _apply_preset_settings(self, settings: dict[str, object]) -> None:
        if settings.get("schema_version") != PRESET_SETTINGS_SCHEMA:
            raise ValueError("该设置预设版本不受当前程序支持。")
        if self.source_volume_info is None:
            raise ValueError("请先打开一个 TIFF 或演示体，再加载设置预设。")

        bounds = self._validate_crop_settings(settings.get("crop_bounds_xyz"))
        spacing_value = settings.get("spacing_xyz")
        if not isinstance(spacing_value, list) or len(spacing_value) != 3:
            raise ValueError("预设中的体素尺寸无效。")
        spacing = tuple(float(value) for value in spacing_value)
        if any(value <= 0.0 or value > 1_000_000.0 for value in spacing):
            raise ValueError("预设中的体素尺寸超出允许范围。")

        background = settings.get("background")
        outline = settings.get("outline")
        phase_payload = settings.get("phase_styles")
        annotation_editor = settings.get("annotation_editor")
        slice_fraction = settings.get("slice_fraction")
        if not isinstance(background, dict) or not isinstance(outline, dict):
            raise ValueError("预设中的全局显示设置无效。")
        if not isinstance(phase_payload, dict):
            raise ValueError("预设中的相样式无效。")
        if not isinstance(annotation_editor, dict):
            raise ValueError("预设中的标注编辑设置无效。")
        if not isinstance(slice_fraction, dict):
            raise ValueError("预设中的逐层计算设置无效。")

        restored_styles: dict[int, PhaseStyle] = {}
        for label, fallback in self._phase_style_cache.items():
            payload = phase_payload.get(str(label))
            restored_styles[label] = (
                self._phase_style_from_settings(label, payload, fallback)
                if payload is not None
                else copy.deepcopy(fallback)
            )
        for raw_label, payload in phase_payload.items():
            try:
                label = int(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError("预设中包含无效的相编号。") from exc
            if str(label) != raw_label or label in restored_styles:
                continue
            if not np.iinfo(np.int32).min <= label <= np.iinfo(np.int32).max:
                raise ValueError("预设中的相编号超出 int32 范围。")
            fallback = PhaseStyle(
                label=label,
                visible=False,
                color=PALETTE[len(restored_styles) % len(PALETTE)],
            )
            restored_styles[label] = self._phase_style_from_settings(
                label, payload, fallback
            )

        blockers = [
            QSignalBlocker(widget)
            for widget in (
                *self.spacing_spins,
                self.downsample_combo,
                self.gradient_checkbox,
                self.outline_checkbox,
                self.outline_width_spin,
                self.edges_checkbox,
                self.annotation_handle_checkbox,
                self.screenshot_scale_combo,
                self.fraction_axis_combo,
                self.numerator_label_spin,
                self.wheel_guard_checkbox,
            )
        ]
        for spin, value in zip(self.spacing_spins, spacing):
            spin.setValue(value)
        self._set_combo_to_data(
            self.downsample_combo,
            int(settings.get("downsample_factor", 1)),
            "预览降采样",
        )
        self.gradient_checkbox.setChecked(bool(background.get("gradient", True)))
        self.background_color = self._color_from_settings(
            background.get("color"), "背景色"
        )
        self.background_color_2 = self._color_from_settings(
            background.get("color_2"), "渐变背景色"
        )
        self.outline_checkbox.setChecked(bool(outline.get("visible", True)))
        self.outline_color = self._color_from_settings(
            outline.get("color"), "边框颜色"
        )
        self.outline_width_spin.setValue(float(outline.get("width", 1.5)))
        self.edges_checkbox.setChecked(bool(settings.get("show_mesh_edges", False)))
        self.annotation_handle_checkbox.setChecked(
            bool(annotation_editor.get("show_handle", True))
        )
        self._set_combo_to_data(
            self.screenshot_scale_combo,
            int(settings.get("screenshot_scale", 2)),
            "截图倍率",
        )
        axis = str(slice_fraction.get("axis", "Z"))
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("预设中的逐层计算方向无效。")
        self.fraction_axis_combo.setCurrentText(axis)
        self.numerator_label_spin.setValue(int(slice_fraction.get("numerator", 3)))
        self.denominator_labels_edit.setText(
            str(slice_fraction.get("denominator", "3,4,5"))
        )
        self.wheel_guard_checkbox.setChecked(bool(settings.get("wheel_guard", True)))
        del blockers

        self._phase_style_cache = restored_styles
        self._restore_annotations(settings.get("annotations", []))

        editor_position = annotation_editor.get("position_xyz", [0.0, 0.0, 0.0])
        if not isinstance(editor_position, list) or len(editor_position) != 3:
            raise ValueError("预设中的标注编辑坐标无效。")
        editor_family = str(annotation_editor.get("font_family", "Times New Roman"))
        if editor_family not in {"Times New Roman", "Arial", "Courier New"}:
            raise ValueError("预设中的标注编辑字体无效。")
        editor_color = self._color_from_settings(
            annotation_editor.get("color", [1.0, 0.82, 0.20]),
            "标注编辑颜色",
        )

        self._update_color_button(
            self.background_button, self.background_color, "选择下方背景色"
        )
        self._update_color_button(
            self.background_button_2, self.background_color_2, "选择上方背景色"
        )
        self._update_color_button(
            self.outline_color_button, self.outline_color, "选择模型边框颜色"
        )
        self._update_color_button(
            self.annotation_color_button,
            editor_color,
            "选择 3D 文字颜色",
        )
        self.update_background(render=False)

        for axis_name, start, stop in zip(
            "XYZ", bounds[::2], bounds[1::2]
        ):
            self.crop_start_spins[axis_name].setValue(start + 1)
            self.crop_end_spins[axis_name].setValue(stop)
        self._activate_crop(bounds, reset_camera=False)
        self._restore_camera_settings(settings.get("camera"))

        selected_phase = settings.get("selected_phase")
        if isinstance(selected_phase, int) and selected_phase in self._row_for_label:
            self.phase_table.selectRow(self._row_for_label[selected_phase])
        selected_annotation = settings.get("selected_annotation")
        selected_annotation_row = -1
        if isinstance(selected_annotation, int):
            for row in range(self.annotation_list.count()):
                item = self.annotation_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == selected_annotation:
                    selected_annotation_row = row
                    break
        if selected_annotation_row >= 0:
            self.annotation_list.setCurrentRow(selected_annotation_row)
        else:
            self.annotation_list.setCurrentRow(-1)
            self.annotation_text_edit.setText(
                str(annotation_editor.get("text", ""))
            )
            for axis_name, value in zip("XYZ", editor_position):
                self.annotation_position_spins[axis_name].setValue(float(value))
            self._annotation_color = editor_color
            self.annotation_font_size_spin.setValue(
                int(annotation_editor.get("font_size", 22))
            )
            self.annotation_font_family_combo.setCurrentText(editor_family)
            self.annotation_bold_checkbox.setChecked(
                bool(annotation_editor.get("bold", True))
            )
            self.annotation_italic_checkbox.setChecked(
                bool(annotation_editor.get("italic", False))
            )
            self._update_color_button(
                self.annotation_color_button,
                self._annotation_color,
                "选择 3D 文字颜色",
            )
            self._sync_annotation_handle(render=False)
        self.renderer.ResetCameraClippingRange()
        self._restore_camera_settings(settings.get("camera"))
        self.vtk_widget.GetRenderWindow().Render()

    def save_named_preset(self, checked: bool = False) -> None:
        del checked
        name = self.preset_combo.currentText()
        try:
            if self.preset_store.contains(name):
                answer = QMessageBox.question(
                    self,
                    "覆盖设置预设",
                    f"设置预设“{name.strip()}”已经存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            settings = self._capture_preset_settings()
            saved_name = self.preset_store.save(name, settings)
        except (PresetStoreError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "无法保存设置预设", str(exc))
            return
        self._refresh_preset_combo(saved_name)
        self.statusBar().showMessage(f"设置预设“{saved_name}”已保存。")

    def load_named_preset(self, checked: bool = False) -> None:
        del checked
        name = self.preset_combo.currentText()
        try:
            settings = self.preset_store.load(name)
        except (PresetStoreError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "无法加载设置预设", str(exc))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents()
            self._apply_preset_settings(settings)
        except (PresetStoreError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "无法加载设置预设", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"设置预设“{name.strip()}”已加载。")

    def delete_named_preset(self, checked: bool = False) -> None:
        del checked
        name = self.preset_combo.currentText()
        answer = QMessageBox.question(
            self,
            "删除设置预设",
            f"确定删除设置预设“{name.strip()}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted_name = self.preset_store.delete(name)
        except (PresetStoreError, OSError) as exc:
            QMessageBox.warning(self, "无法删除设置预设", str(exc))
            return
        self._refresh_preset_combo()
        self.statusBar().showMessage(f"设置预设“{deleted_name}”已删除。")

    def _build_analysis_dock(self) -> None:
        dock = QDockWidget("计算 / Slice Volume Fraction", self)
        dock.setObjectName("analysisDock")
        dock.setMinimumWidth(340)
        dock.setMinimumHeight(245)

        body = QWidget(dock)
        form = QFormLayout(body)
        self.fraction_axis_combo = QComboBox()
        self.fraction_axis_combo.addItems(["X", "Y", "Z"])
        self.fraction_axis_combo.setCurrentText("Z")
        form.addRow("切片方向", self.fraction_axis_combo)

        self.numerator_label_spin = QSpinBox()
        self.numerator_label_spin.setRange(
            int(np.iinfo(np.int32).min),
            int(np.iinfo(np.int32).max),
        )
        self.numerator_label_spin.setValue(3)
        form.addRow("分子相编号", self.numerator_label_spin)

        self.denominator_labels_edit = QLineEdit("3,4,5")
        self.denominator_labels_edit.setPlaceholderText("例如：3,4,5")
        form.addRow("分母相编号", self.denominator_labels_edit)

        formula = QLabel("默认：count(3) / count(3,4,5)")
        formula.setWordWrap(True)
        form.addRow("计算公式", formula)

        calculate_button = QPushButton("计算并显示表格")
        calculate_button.setMinimumHeight(36)
        calculate_button.clicked.connect(self.calculate_slice_fractions)
        form.addRow(calculate_button)

        note = QLabel(
            "使用当前裁剪工作体积的全分辨率矩阵，不受三维预览降采样影响。"
        )
        note.setWordWrap(True)
        form.addRow(note)

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.splitDockWidget(self.properties_dock, dock, Qt.Orientation.Vertical)
        self.analysis_dock = dock

    def _build_results_dock(self) -> None:
        dock = QDockWidget("Slice Volume Fraction", self)
        dock.setObjectName("sliceResultsDock")
        body = QWidget(dock)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        self.fraction_result_label = QLabel("尚未计算")
        layout.addWidget(self.fraction_result_label)

        self.fraction_table = QTableWidget(0, 2, body)
        self.fraction_table.setHorizontalHeaderLabels(
            ["Original Slice Number", "Volume Fraction"]
        )
        self.fraction_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.fraction_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.fraction_table.verticalHeader().setVisible(False)
        result_header = self.fraction_table.horizontalHeader()
        result_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        result_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.fraction_table, 1)

        save_button = QPushButton("保存 CSV")
        save_button.clicked.connect(self.save_fraction_csv)
        layout.addWidget(save_button)

        dock.setWidget(body)
        dock.setMinimumHeight(230)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.hide()
        self.results_dock = dock

    def _build_actions(self) -> None:
        open_action = QAction("打开 TIFF", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_tiff)

        open_folder_action = QAction("打开 TIFF 文件夹", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.choose_tiff_folder)

        demo_action = QAction("演示体", self)
        demo_action.triggered.connect(self.load_demo)

        rebuild_action = QAction("重建", self)
        rebuild_action.setShortcut("Ctrl+R")
        rebuild_action.triggered.connect(self.rebuild_all)

        screenshot_action = QAction("截图", self)
        screenshot_action.setShortcut("Ctrl+Shift+S")
        screenshot_action.triggered.connect(self.save_screenshot)

        quit_action = QAction("退出", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(open_action)
        file_menu.addAction(open_folder_action)
        file_menu.addAction(demo_action)
        file_menu.addSeparator()
        file_menu.addAction(screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.phase_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        view_menu.addAction(self.analysis_dock.toggleViewAction())
        view_menu.addAction(self.results_dock.toggleViewAction())

        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.addAction(open_action)
        toolbar.addAction(open_folder_action)
        toolbar.addAction(demo_action)
        toolbar.addSeparator()
        toolbar.addAction(rebuild_action)
        toolbar.addAction(screenshot_action)
        self.addToolBar(toolbar)

    def _setup_vtk(self) -> None:
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)

        self.renderer = vtkRenderer()
        self.renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.1)
        if hasattr(self.renderer, "SetUseFXAA"):
            self.renderer.SetUseFXAA(True)

        render_window.AddRenderer(self.renderer)
        self.update_background(render=False)

        self.interactor = render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()
        self._setup_annotation_handle()
        self.statusBar().showMessage("正在初始化…")

    def choose_tiff(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择多页标签 TIFF",
            "",
            "TIFF 文件 (*.tif *.tiff);;所有文件 (*)",
        )
        if filename:
            self._clear_batch_folder()
            self.load_tiff(Path(filename))

    def choose_tiff_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择包含 TIFF 的文件夹",
            str(self.batch_folder or ""),
        )
        if not directory:
            return
        folder = Path(directory)
        try:
            paths = discover_tiff_files(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法读取文件夹", str(exc))
            return
        if not paths:
            QMessageBox.information(
                self,
                "没有 TIFF",
                "所选文件夹顶层没有 .tif 或 .tiff 文件。",
            )
            return

        self.batch_folder = folder
        self.batch_tiff_paths = paths
        self.batch_reference_combo.clear()
        for path in paths:
            self.batch_reference_combo.addItem(path.name)
        self.batch_reference_combo.setEnabled(True)
        self.batch_folder_label.setText(
            f"{folder}\n共发现 {len(paths)} 个 TIFF；当前参考：{paths[0].name}"
        )
        self.batch_reference_combo.setCurrentIndex(0)
        self.load_tiff(paths[0])

    def load_selected_batch_reference(self, checked: bool = False) -> None:
        del checked
        index = self.batch_reference_combo.currentIndex()
        if index < 0 or index >= len(self.batch_tiff_paths):
            QMessageBox.information(self, "尚无文件夹", "请先选择 TIFF 文件夹。")
            return
        path = self.batch_tiff_paths[index]
        if self.load_tiff(path):
            assert self.batch_folder is not None
            self.batch_folder_label.setText(
                f"{self.batch_folder}\n共发现 {len(self.batch_tiff_paths)} 个 TIFF；"
                f"当前参考：{path.name}"
            )

    def _clear_batch_folder(self) -> None:
        self.batch_folder = None
        self.batch_tiff_paths = []
        if hasattr(self, "batch_reference_combo"):
            self.batch_reference_combo.clear()
            self.batch_reference_combo.setEnabled(False)
            self.batch_folder_label.setText("尚未选择 TIFF 文件夹")

    def _read_tiff_series(
        self,
        path: Path,
        warn_large: bool,
    ) -> tuple[np.ndarray, str] | None:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("TIFF 中没有可读取的图像序列。")
            series = tif.series[0]
            axes = getattr(series, "axes", "")
            estimated_bytes = int(
                np.prod(series.shape) * np.dtype(series.dtype).itemsize
            )
            if warn_large and estimated_bytes > 1024**3:
                answer = QMessageBox.question(
                    self,
                    "大文件提示",
                    "该序列未压缩数据约为 "
                    f"{estimated_bytes / 1024**3:.2f} GiB。"
                    "\n本程序会一次读入内存，是否继续？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return None
            return series.asarray(), axes

    def load_tiff(self, path: Path) -> bool:
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            result = self._read_tiff_series(path, warn_large=True)
            if result is None:
                return False
            array, axes = result
            self._set_volume(array, path.name, axes)
            self.current_tiff_path = path
            return True
        except Exception as exc:
            QMessageBox.critical(self, "无法载入 TIFF", str(exc))
            self.statusBar().showMessage("载入失败")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def load_demo(self) -> None:
        self._clear_batch_folder()
        self.current_tiff_path = None
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._set_volume(make_demo_volume(), "内置三相演示体", "ZYX")
        except Exception as exc:
            QMessageBox.critical(self, "演示体错误", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _set_volume(self, array: np.ndarray, name: str, axes: str) -> None:
        volume = normalize_label_volume(array)
        info = analyze_volume(volume)
        self.source_volume = volume
        self.source_volume_info = info
        self.background_label = info.background_label
        self.source_name = name
        self.source_axes = axes or "未知"

        non_background = [
            label for label in info.labels if label != self.background_label
        ]
        visible_labels = set(non_background[:8])
        if not visible_labels and info.labels:
            visible_labels.add(info.labels[0])

        self._phase_style_cache = {
            label: PhaseStyle(
                label=label,
                visible=label in visible_labels,
                color=PALETTE[index % len(PALETTE)],
                opacity=0.82 if len(visible_labels) > 1 else 1.0,
            )
            for index, label in enumerate(info.labels)
        }
        self._clear_annotations_without_confirmation()
        self._configure_crop_controls(info.shape_zyx)
        z_size, y_size, x_size = info.shape_zyx
        full_bounds = (0, x_size, 0, y_size, 0, z_size)
        self.statusBar().showMessage("数据已载入，正在生成可见相…")
        self._activate_crop(full_bounds, reset_camera=True)

    def _configure_crop_controls(
        self,
        shape_zyx: tuple[int, int, int],
    ) -> None:
        z_size, y_size, x_size = shape_zyx
        for axis, size in zip("XYZ", (x_size, y_size, z_size)):
            start_spin = self.crop_start_spins[axis]
            end_spin = self.crop_end_spins[axis]
            start_spin.setRange(1, size)
            end_spin.setRange(1, size)
            start_spin.setValue(1)
            end_spin.setValue(size)

    def _activate_crop(
        self,
        bounds_xyz: tuple[int, int, int, int, int, int],
        reset_camera: bool,
    ) -> None:
        assert self.source_volume is not None
        cropped = crop_label_volume(self.source_volume, bounds_xyz)
        info = analyze_volume(cropped)
        self.crop_bounds_xyz = bounds_xyz
        self.volume = cropped
        self.volume_info = info
        self.phase_styles = {
            label: self._phase_style_cache[label] for label in info.labels
        }
        self._reset_slice_results()
        self._populate_phase_table()
        self._update_volume_information()
        if self.annotation_list.currentItem() is None:
            self.use_crop_center_for_annotation()
        self.rebuild_all(reset_camera=reset_camera)

    def _reset_slice_results(self) -> None:
        self.slice_fraction_results = None
        self.fraction_table.setRowCount(0)
        self.fraction_result_label.setText("尚未计算")
        self.results_dock.hide()

    def _update_volume_information(self) -> None:
        if self.source_volume_info is None or self.volume_info is None:
            return
        source_z, source_y, source_x = self.source_volume_info.shape_zyx
        crop_z, crop_y, crop_x = self.volume_info.shape_zyx
        x_start, x_stop, y_start, y_stop, z_start, z_stop = self.crop_bounds_xyz
        memory_mib = self.volume.nbytes / 1024**2 if self.volume is not None else 0
        self.info_label.setText(
            f"程序版本：V{APP_VERSION}\n"
            f"{self.source_name}\n"
            f"原始形状 Z×Y×X：{source_z} × {source_y} × {source_x}\n"
            f"工作形状 Z×Y×X：{crop_z} × {crop_y} × {crop_x}\n"
            f"轴信息：{self.source_axes}\n"
            f"当前相数量：{len(self.volume_info.labels)}；"
            f"工作体积内存：{memory_mib:.1f} MiB\n"
            f"推测背景相：{self.background_label}"
        )
        self.crop_shape_label.setText(
            f"当前工作形状：[X={crop_x}, Y={crop_y}, Z={crop_z}]\n"
            f"原始索引范围：X {x_start + 1}–{x_stop}，"
            f"Y {y_start + 1}–{y_stop}，Z {z_start + 1}–{z_stop}"
        )

    def _bounds_from_crop_controls(
        self,
    ) -> tuple[int, int, int, int, int, int]:
        values: list[int] = []
        for axis in "XYZ":
            start = self.crop_start_spins[axis].value()
            stop = self.crop_end_spins[axis].value()
            if start > stop:
                raise ValueError(f"{axis} 方向的起始索引不能大于结束索引。")
            values.extend((start - 1, stop))
        return (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
        )

    def apply_crop(self, checked: bool = False) -> None:
        del checked
        if self.source_volume is None:
            QMessageBox.information(self, "尚无数据", "请先打开 TIFF 或演示体。")
            return
        try:
            bounds = self._bounds_from_crop_controls()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            self._activate_crop(bounds, reset_camera=True)
            z_size, y_size, x_size = self.volume.shape
            self.statusBar().showMessage(
                f"裁剪已应用；当前工作形状 X×Y×Z："
                f"{x_size} × {y_size} × {z_size}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法应用裁剪", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def reset_crop(self, checked: bool = False) -> None:
        del checked
        if self.source_volume_info is None:
            return
        z_size, y_size, x_size = self.source_volume_info.shape_zyx
        for axis, size in zip("XYZ", (x_size, y_size, z_size)):
            self.crop_start_spins[axis].setValue(1)
            self.crop_end_spins[axis].setValue(size)
        self.apply_crop()

    def _populate_phase_table(self) -> None:
        self._updating_phase_ui = True
        self.phase_table.setRowCount(0)
        self._row_for_label.clear()
        self._color_buttons.clear()
        self._opacity_spins.clear()
        self._visibility_checks.clear()
        self._smoothing_checks.clear()
        self._lighting_checks.clear()

        if self.volume_info is None:
            self._updating_phase_ui = False
            return

        for row, (label, count) in enumerate(
            zip(self.volume_info.labels, self.volume_info.counts)
        ):
            style = self.phase_styles[label]
            self.phase_table.insertRow(row)
            self._row_for_label[label] = row

            visible = QCheckBox()
            visible.setChecked(style.visible)
            visible.setToolTip("显示或隐藏这个相")
            visible.toggled.connect(partial(self._set_phase_visibility, label))
            self.phase_table.setCellWidget(row, 0, visible)
            self._visibility_checks[label] = visible

            percentage = 100.0 * count / self.volume_info.voxel_count
            label_item = QTableWidgetItem(str(label))
            label_item.setData(Qt.ItemDataRole.UserRole, label)
            label_item.setToolTip(
                f"相 {label}\n体素数：{count:,}\n体积分数：{percentage:.3f}%"
            )
            self.phase_table.setItem(row, 1, label_item)

            color_button = QPushButton()
            color_button.setFixedWidth(64)
            self._update_color_button(color_button, style.color, f"相 {label} 颜色")
            color_button.clicked.connect(partial(self.choose_phase_color, label))
            self.phase_table.setCellWidget(row, 2, color_button)
            self._color_buttons[label] = color_button

            opacity = QDoubleSpinBox()
            opacity.setDecimals(2)
            opacity.setRange(0.0, 1.0)
            opacity.setSingleStep(0.05)
            opacity.setValue(style.opacity)
            opacity.setKeyboardTracking(False)
            opacity.valueChanged.connect(partial(self._set_phase_opacity, label))
            self.phase_table.setCellWidget(row, 3, opacity)
            self._opacity_spins[label] = opacity

            smoothing = QCheckBox()
            smoothing.setChecked(style.smoothing)
            smoothing.setToolTip("右侧视图是否对这个相启用平滑")
            smoothing.toggled.connect(partial(self._set_phase_smoothing, label))
            self.phase_table.setCellWidget(row, 4, smoothing)
            self._smoothing_checks[label] = smoothing

            lighting = QCheckBox()
            lighting.setChecked(style.lighting)
            lighting.setToolTip("启用或关闭这个相的表面光照")
            lighting.toggled.connect(partial(self._set_phase_lighting, label))
            self.phase_table.setCellWidget(row, 5, lighting)
            self._lighting_checks[label] = lighting

        self._updating_phase_ui = False
        selected_row = next(
            (
                self._row_for_label[label]
                for label, style in self.phase_styles.items()
                if style.visible
            ),
            0,
        )
        if self.phase_table.rowCount():
            self.phase_table.selectRow(selected_row)
            self._load_selected_phase_controls()

    def selected_label(self) -> int | None:
        row = self.phase_table.currentRow()
        if row < 0:
            return None
        item = self.phase_table.item(row, 1)
        if item is None:
            return None
        return int(item.data(Qt.ItemDataRole.UserRole))

    def _load_selected_phase_controls(self) -> None:
        label = self.selected_label()
        if label is None or label not in self.phase_styles:
            self.selected_phase_label.setText("未选择")
            return
        style = self.phase_styles[label]
        self.selected_phase_label.setText(str(label))
        self.selected_visible_checkbox.setChecked(style.visible)
        self.selected_opacity_spin.setValue(style.opacity)
        self.selected_smoothing_checkbox.setChecked(style.smoothing)
        self.selected_iterations_spin.setValue(style.iterations)
        self.selected_relaxation_spin.setValue(style.relaxation)
        self.selected_constraint_spin.setValue(style.max_move_voxels)
        self.selected_lighting_checkbox.setChecked(style.lighting)
        self.selected_interpolation_combo.setCurrentText(style.interpolation)
        self.selected_ambient_spin.setValue(style.ambient)
        self.selected_diffuse_spin.setValue(style.diffuse)
        self.selected_specular_spin.setValue(style.specular)
        self.selected_specular_power_spin.setValue(style.specular_power)
        self._property_color = style.color
        self._update_color_button(
            self.selected_color_button,
            style.color,
            f"选择相 {label} 的颜色",
        )

    def apply_selected_phase(self) -> None:
        label = self.selected_label()
        if label is None:
            return
        style = self.phase_styles[label]
        style.visible = self.selected_visible_checkbox.isChecked()
        style.color = self._property_color
        style.opacity = self.selected_opacity_spin.value()
        style.smoothing = self.selected_smoothing_checkbox.isChecked()
        style.iterations = self.selected_iterations_spin.value()
        style.relaxation = self.selected_relaxation_spin.value()
        style.max_move_voxels = self.selected_constraint_spin.value()
        style.lighting = self.selected_lighting_checkbox.isChecked()
        style.interpolation = self.selected_interpolation_combo.currentText()
        style.ambient = self.selected_ambient_spin.value()
        style.diffuse = self.selected_diffuse_spin.value()
        style.specular = self.selected_specular_spin.value()
        style.specular_power = self.selected_specular_power_spin.value()
        self._sync_table_row(label)
        self.rebuild_phase(label)

    def _sync_table_row(self, label: int) -> None:
        style = self.phase_styles[label]
        self._updating_phase_ui = True
        visibility_blocker = QSignalBlocker(self._visibility_checks[label])
        opacity_blocker = QSignalBlocker(self._opacity_spins[label])
        smoothing_blocker = QSignalBlocker(self._smoothing_checks[label])
        lighting_blocker = QSignalBlocker(self._lighting_checks[label])
        self._visibility_checks[label].setChecked(style.visible)
        self._opacity_spins[label].setValue(style.opacity)
        self._smoothing_checks[label].setChecked(style.smoothing)
        self._lighting_checks[label].setChecked(style.lighting)
        del visibility_blocker, opacity_blocker, smoothing_blocker, lighting_blocker
        self._update_color_button(
            self._color_buttons[label],
            style.color,
            f"相 {label} 颜色",
        )
        self._updating_phase_ui = False

    def _set_phase_visibility(self, label: int, visible: bool) -> None:
        if self._updating_phase_ui:
            return
        self.phase_styles[label].visible = visible
        if self.selected_label() == label:
            self.selected_visible_checkbox.setChecked(visible)
        if self._vtk_image is None:
            return
        if visible and label not in self.phase_pipelines:
            self.rebuild_phase(label)
            return
        pipeline = self.phase_pipelines.get(label)
        if pipeline:
            pipeline["actor"].SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()

    def _set_phase_opacity(self, label: int, opacity: float) -> None:
        if self._updating_phase_ui:
            return
        self.phase_styles[label].opacity = opacity
        if self.selected_label() == label:
            self.selected_opacity_spin.setValue(opacity)
        self._apply_actor_style(label)
        self.vtk_widget.GetRenderWindow().Render()

    def _set_phase_smoothing(self, label: int, enabled: bool) -> None:
        if self._updating_phase_ui:
            return
        self.phase_styles[label].smoothing = enabled
        if self.selected_label() == label:
            self.selected_smoothing_checkbox.setChecked(enabled)
        if self._vtk_image is not None and self.phase_styles[label].visible:
            self.rebuild_phase(label)

    def _set_phase_lighting(self, label: int, enabled: bool) -> None:
        if self._updating_phase_ui:
            return
        self.phase_styles[label].lighting = enabled
        if self.selected_label() == label:
            self.selected_lighting_checkbox.setChecked(enabled)
        self._apply_actor_style(label)
        self.vtk_widget.GetRenderWindow().Render()

    def choose_phase_color(self, label: int, checked: bool = False) -> None:
        del checked
        style = self.phase_styles[label]
        initial = QColor.fromRgbF(*style.color)
        color = QColorDialog.getColor(initial, self, f"选择相 {label} 的颜色")
        if not color.isValid():
            return
        style.color = color.redF(), color.greenF(), color.blueF()
        self._update_color_button(
            self._color_buttons[label],
            style.color,
            f"相 {label} 颜色",
        )
        if self.selected_label() == label:
            self._property_color = style.color
            self._update_color_button(
                self.selected_color_button,
                style.color,
                f"选择相 {label} 的颜色",
            )
        self._apply_actor_style(label)
        self.vtk_widget.GetRenderWindow().Render()

    def choose_selected_phase_color(self, checked: bool = False) -> None:
        del checked
        label = self.selected_label()
        if label is None:
            return
        initial = QColor.fromRgbF(*self._property_color)
        color = QColorDialog.getColor(initial, self, f"选择相 {label} 的颜色")
        if not color.isValid():
            return
        self._property_color = color.redF(), color.greenF(), color.blueF()
        self._update_color_button(
            self.selected_color_button,
            self._property_color,
            f"选择相 {label} 的颜色",
        )

    def _set_visibility_group(self, mode: str) -> None:
        if self.volume_info is None:
            return
        if mode == "all" and len(self.phase_styles) > 32:
            answer = QMessageBox.question(
                self,
                "显示大量相",
                f"当前有 {len(self.phase_styles)} 个相，全部重建可能较慢。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        for label, style in self.phase_styles.items():
            if mode == "all":
                style.visible = True
            elif mode == "none":
                style.visible = False
            else:
                style.visible = label != self.background_label
            self._sync_table_row(label)
        self._load_selected_phase_controls()
        self.rebuild_all(reset_camera=False)

    def current_spacing(self) -> tuple[float, float, float]:
        return tuple(spin.value() for spin in self.spacing_spins)

    def rebuild_all(
        self,
        checked: bool = False,
        reset_camera: bool = True,
        raise_errors: bool = False,
    ) -> None:
        del checked
        if self.volume is None:
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            self._vtk_image, self._padding_label = self._prepare_vtk_image()
            self.phase_pipelines.clear()
            self._disable_annotation_handle()
            self.renderer.RemoveAllViewProps()
            self._global_pipelines.clear()

            visible_labels = [
                label
                for label, style in self.phase_styles.items()
                if style.visible
            ]
            total_faces = 0
            for index, label in enumerate(visible_labels, start=1):
                self.statusBar().showMessage(
                    f"正在重建相 {label}（{index}/{len(visible_labels)}）…"
                )
                QApplication.processEvents()
                total_faces += self._build_and_add_phase(label)

            self._add_outline(self.renderer)
            self._restore_annotation_actors()
            if reset_camera:
                self.reset_camera(render=False)
            self._sync_annotation_handle(render=False)
            self.vtk_widget.GetRenderWindow().Render()
            factor = int(self.downsample_combo.currentData())
            self.statusBar().showMessage(
                f"已显示 {len(visible_labels)} 个相；降采样 {factor}×；"
                f"平滑表面共 {total_faces:,} 个面"
            )
        except Exception as exc:
            if raise_errors:
                raise
            QMessageBox.critical(self, "表面重建失败", str(exc))
            self.statusBar().showMessage("表面重建失败")
        finally:
            QApplication.restoreOverrideCursor()

    def rebuild_phase(self, label: int) -> None:
        if self._vtk_image is None:
            self.rebuild_all()
            return
        old = self.phase_pipelines.pop(label, None)
        if old:
            self.renderer.RemoveActor(old["actor"])
        if not self.phase_styles[label].visible:
            self.vtk_widget.GetRenderWindow().Render()
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage(f"正在重建相 {label}…")
            QApplication.processEvents()
            faces = self._build_and_add_phase(label)
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage(f"相 {label} 已更新；平滑表面 {faces:,} 个面")
        except Exception as exc:
            QMessageBox.critical(self, f"相 {label} 重建失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _prepare_vtk_image(self) -> tuple[vtkImageData, int]:
        assert self.volume is not None
        factor = int(self.downsample_combo.currentData())
        base_spacing = self.current_spacing()
        self._render_spacing = base_spacing
        sampled, spacing = downsample_labels(
            self.volume,
            factor,
            base_spacing,
        )
        padded, padding_label = pad_label_volume(sampled)
        image = self._make_vtk_image(padded, spacing)
        x_start, _, y_start, _, z_start, _ = self.crop_bounds_xyz
        image.SetOrigin(
            x_start * base_spacing[0] - spacing[0],
            y_start * base_spacing[1] - spacing[1],
            z_start * base_spacing[2] - spacing[2],
        )
        return image, padding_label

    @staticmethod
    def _make_vtk_image(
        volume_zyx: np.ndarray,
        spacing_xyz: tuple[float, float, float],
    ) -> vtkImageData:
        z_size, y_size, x_size = volume_zyx.shape
        image = vtkImageData()
        image.SetDimensions(x_size, y_size, z_size)
        image.SetSpacing(*spacing_xyz)
        vtk_values = numpy_to_vtk(
            flatten_zyx_for_vtk(volume_zyx),
            deep=True,
        )
        vtk_values.SetName("PhaseLabels")
        image.GetPointData().SetScalars(vtk_values)
        return image

    def _build_and_add_phase(self, label: int) -> int:
        assert self._vtk_image is not None
        style = self.phase_styles[label]
        pipeline = self._make_surface_pipeline(
            label,
            style,
            smooth=style.smoothing,
        )
        self.renderer.AddActor(pipeline["actor"])
        self.phase_pipelines[label] = pipeline
        self._apply_actor_style(label)
        return int(pipeline["cells"])

    def _make_surface_pipeline(
        self,
        label: int,
        style: PhaseStyle,
        smooth: bool,
    ) -> dict[str, object]:
        assert self._vtk_image is not None
        surface = vtkSurfaceNets3D()
        surface.SetInputData(self._vtk_image)
        surface.SetBackgroundLabel(self._padding_label)
        surface.SetValue(0, label)
        surface.SetOutputMeshTypeToTriangles()

        if smooth:
            surface.SmoothingOn()
            surface.SetNumberOfIterations(style.iterations)
            surface.SetRelaxationFactor(style.relaxation)
            surface.AutomaticSmoothingConstraintsOff()
            surface.SetConstraintStrategyToConstraintBox()
            spacing = self._vtk_image.GetSpacing()
            surface.SetConstraintBox(
                style.max_move_voxels * spacing[0],
                style.max_move_voxels * spacing[1],
                style.max_move_voxels * spacing[2],
            )
        else:
            surface.SmoothingOff()

        surface.Update()

        normals = vtkPolyDataNormals()
        normals.SetInputConnection(surface.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetAmbient(0.14)
        actor.GetProperty().SetDiffuse(0.82)
        actor.GetProperty().SetSpecular(0.18)
        actor.GetProperty().SetSpecularPower(18.0)

        return {
            "surface": surface,
            "normals": normals,
            "mapper": mapper,
            "actor": actor,
            "cells": surface.GetOutput().GetNumberOfCells(),
            "points": surface.GetOutput().GetNumberOfPoints(),
        }

    def _apply_actor_style(self, label: int) -> None:
        pipeline = self.phase_pipelines.get(label)
        if not pipeline:
            return
        style = self.phase_styles[label]
        show_edges = self.edges_checkbox.isChecked()
        actor = pipeline["actor"]
        actor.SetVisibility(style.visible)
        prop = actor.GetProperty()
        prop.SetColor(*style.color)
        prop.SetOpacity(style.opacity)
        prop.SetEdgeVisibility(show_edges)
        prop.SetEdgeColor(0.04, 0.05, 0.07)
        prop.SetLineWidth(0.7)
        if style.lighting:
            prop.LightingOn()
        else:
            prop.LightingOff()
        if style.interpolation == "Flat":
            prop.SetInterpolationToFlat()
        elif style.interpolation == "Gouraud":
            prop.SetInterpolationToGouraud()
        else:
            prop.SetInterpolationToPhong()
        prop.SetAmbient(style.ambient)
        prop.SetDiffuse(style.diffuse)
        prop.SetSpecular(style.specular)
        prop.SetSpecularPower(style.specular_power)

    def update_actor_styles(
        self,
        checked: bool = False,
        render: bool = True,
    ) -> None:
        del checked
        for label in self.phase_pipelines:
            self._apply_actor_style(label)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def choose_background_color(
        self,
        second: bool,
        checked: bool = False,
    ) -> None:
        del checked
        current = self.background_color_2 if second else self.background_color
        color = QColorDialog.getColor(
            QColor.fromRgbF(*current),
            self,
            "选择背景颜色",
        )
        if not color.isValid():
            return
        rgb = color.redF(), color.greenF(), color.blueF()
        if second:
            self.background_color_2 = rgb
            self._update_color_button(
                self.background_button_2,
                rgb,
                "选择上方背景色",
            )
        else:
            self.background_color = rgb
            self._update_color_button(
                self.background_button,
                rgb,
                "选择下方背景色",
            )
        self.update_background()

    def update_background(
        self,
        checked: bool = False,
        render: bool = True,
    ) -> None:
        del checked
        gradient = self.gradient_checkbox.isChecked()
        self.renderer.SetBackground(*self.background_color)
        self.renderer.SetBackground2(*self.background_color_2)
        self.renderer.SetGradientBackground(gradient)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def choose_outline_color(self, checked: bool = False) -> None:
        del checked
        color = QColorDialog.getColor(
            QColor.fromRgbF(*self.outline_color),
            self,
            "选择模型边框颜色",
        )
        if not color.isValid():
            return
        self.outline_color = color.redF(), color.greenF(), color.blueF()
        self._update_color_button(
            self.outline_color_button,
            self.outline_color,
            "选择模型边框颜色",
        )
        self.update_outline_style()

    def update_outline_style(
        self,
        value: bool | float = False,
        render: bool = True,
    ) -> None:
        del value
        self.outline_width = self.outline_width_spin.value()
        if self._outline_actor is not None:
            self._outline_actor.SetVisibility(self.outline_checkbox.isChecked())
            prop = self._outline_actor.GetProperty()
            prop.SetColor(*self.outline_color)
            prop.SetLineWidth(self.outline_width)
            prop.SetOpacity(0.90)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    @staticmethod
    def _update_color_button(
        button: QPushButton,
        color: tuple[float, float, float],
        tooltip: str,
    ) -> None:
        red, green, blue = (round(value * 255) for value in color)
        brightness = 0.299 * red + 0.587 * green + 0.114 * blue
        foreground = "#111111" if brightness > 145 else "#ffffff"
        button.setText(f"#{red:02X}{green:02X}{blue:02X}")
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({red}, {green}, {blue});"
            f"color: {foreground};"
            "border: 1px solid #777; padding: 4px;"
            "}"
        )
        button.setToolTip(tooltip)

    def _add_outline(self, renderer: vtkRenderer) -> None:
        x_start, x_stop, y_start, y_stop, z_start, z_stop = self.crop_bounds_xyz
        spacing_x, spacing_y, spacing_z = self._render_spacing
        outline = vtkCubeSource()
        outline.SetBounds(
            (x_start - 0.5) * spacing_x,
            (x_stop - 0.5) * spacing_x,
            (y_start - 0.5) * spacing_y,
            (y_stop - 0.5) * spacing_y,
            (z_start - 0.5) * spacing_z,
            (z_stop - 0.5) * spacing_z,
        )
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        self._outline_actor = actor
        self.update_outline_style(render=False)
        renderer.AddActor(actor)
        self._global_pipelines.extend([outline, mapper, actor])

    def use_crop_center_for_annotation(self, checked: bool = False) -> None:
        del checked
        x_start, x_stop, y_start, y_stop, z_start, z_stop = self.crop_bounds_xyz
        spacing_x, spacing_y, spacing_z = self.current_spacing()
        centers = (
            (x_start + x_stop - 1) * spacing_x / 2.0,
            (y_start + y_stop - 1) * spacing_y / 2.0,
            (z_start + z_stop - 1) * spacing_z / 2.0,
        )
        for axis, value in zip("XYZ", centers):
            self.annotation_position_spins[axis].setValue(value)

    def choose_annotation_color(self, checked: bool = False) -> None:
        del checked
        color = QColorDialog.getColor(
            QColor.fromRgbF(*self._annotation_color),
            self,
            "选择 3D 文字颜色",
        )
        if not color.isValid():
            return
        self._annotation_color = color.redF(), color.greenF(), color.blueF()
        self._update_color_button(
            self.annotation_color_button,
            self._annotation_color,
            "选择 3D 文字颜色",
        )

    def _annotation_form_values(
        self,
    ) -> tuple[
        str,
        tuple[float, float, float],
        int,
        str,
        bool,
        bool,
    ]:
        text = self.annotation_text_edit.text().strip()
        if not text:
            raise ValueError("请输入标注文字。")
        position = tuple(
            self.annotation_position_spins[axis].value() for axis in "XYZ"
        )
        return (
            text,
            position,
            self.annotation_font_size_spin.value(),
            self.annotation_font_family_combo.currentText(),
            self.annotation_bold_checkbox.isChecked(),
            self.annotation_italic_checkbox.isChecked(),
        )

    def _selected_annotation_id(self) -> int | None:
        item = self.annotation_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    @staticmethod
    def _annotation_item_text(annotation: TextAnnotation) -> str:
        x, y, z = annotation.position_xyz
        return (
            f"{annotation.identifier}: {annotation.text}  "
            f"({x:.2f}, {y:.2f}, {z:.2f})"
        )

    def add_annotation(self, checked: bool = False) -> None:
        del checked
        if self.source_volume is None:
            QMessageBox.information(self, "尚无数据", "请先打开 TIFF 或演示体。")
            return
        try:
            (
                text,
                position,
                font_size,
                font_family,
                bold,
                italic,
            ) = self._annotation_form_values()
        except ValueError as exc:
            QMessageBox.warning(self, "无法新增标注", str(exc))
            return

        identifier = self._next_annotation_id
        self._next_annotation_id += 1
        annotation = TextAnnotation(
            identifier=identifier,
            text=text,
            position_xyz=position,
            color=self._annotation_color,
            font_size=font_size,
            font_family=font_family,
            bold=bold,
            italic=italic,
        )
        self.annotations[identifier] = annotation
        self._update_annotation_actor(annotation)
        assert annotation.actor is not None
        self.renderer.AddActor(annotation.actor)

        item = QListWidgetItem(self._annotation_item_text(annotation))
        item.setData(Qt.ItemDataRole.UserRole, identifier)
        self.annotation_list.addItem(item)
        self.annotation_list.setCurrentItem(item)
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"已新增 3D 标注：{text}")

    def update_annotation(self, checked: bool = False) -> None:
        del checked
        identifier = self._selected_annotation_id()
        if identifier is None:
            QMessageBox.information(self, "未选择标注", "请先在列表中选择一个标注。")
            return
        try:
            (
                text,
                position,
                font_size,
                font_family,
                bold,
                italic,
            ) = self._annotation_form_values()
        except ValueError as exc:
            QMessageBox.warning(self, "无法更新标注", str(exc))
            return

        annotation = self.annotations[identifier]
        annotation.text = text
        annotation.position_xyz = position
        annotation.color = self._annotation_color
        annotation.font_size = font_size
        annotation.font_family = font_family
        annotation.bold = bold
        annotation.italic = italic
        self._update_annotation_actor(annotation)
        self._sync_annotation_handle(render=False)
        item = self.annotation_list.currentItem()
        if item is not None:
            item.setText(self._annotation_item_text(annotation))
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"3D 标注已更新：{text}")

    def remove_annotation(self, checked: bool = False) -> None:
        del checked
        identifier = self._selected_annotation_id()
        if identifier is None:
            return
        annotation = self.annotations.pop(identifier)
        if annotation.actor is not None:
            self.renderer.RemoveActor(annotation.actor)
        row = self.annotation_list.currentRow()
        self.annotation_list.takeItem(row)
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"已删除 3D 标注：{annotation.text}")

    def clear_annotations(self, checked: bool = False) -> None:
        del checked
        if not self.annotations:
            return
        answer = QMessageBox.question(
            self,
            "清空 3D 标注",
            f"确定删除全部 {len(self.annotations)} 条标注吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._clear_annotations_without_confirmation()
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_annotations_without_confirmation(self) -> None:
        self._disable_annotation_handle()
        if hasattr(self, "renderer"):
            for annotation in self.annotations.values():
                if annotation.actor is not None:
                    self.renderer.RemoveActor(annotation.actor)
        self.annotations.clear()
        self._next_annotation_id = 1
        if hasattr(self, "annotation_list"):
            self.annotation_list.clear()

    def load_selected_annotation(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            self._disable_annotation_handle()
            return
        identifier = int(current.data(Qt.ItemDataRole.UserRole))
        annotation = self.annotations.get(identifier)
        if annotation is None:
            return
        self.annotation_text_edit.setText(annotation.text)
        for axis, value in zip("XYZ", annotation.position_xyz):
            self.annotation_position_spins[axis].setValue(value)
        self.annotation_font_size_spin.setValue(annotation.font_size)
        self.annotation_font_family_combo.setCurrentText(annotation.font_family)
        self.annotation_bold_checkbox.setChecked(annotation.bold)
        self.annotation_italic_checkbox.setChecked(annotation.italic)
        self._annotation_color = annotation.color
        self._update_color_button(
            self.annotation_color_button,
            annotation.color,
            "选择 3D 文字颜色",
        )
        self._sync_annotation_handle()

    def _update_annotation_actor(self, annotation: TextAnnotation) -> None:
        if annotation.actor is None:
            annotation.actor = vtkBillboardTextActor3D()
        actor = annotation.actor
        actor.SetInput(annotation.text)
        actor.SetPosition(*annotation.position_xyz)
        actor.SetVisibility(True)
        actor.PickableOff()
        actor.UseBoundsOff()
        text_property = actor.GetTextProperty()
        text_property.SetColor(*annotation.color)
        text_property.SetFontSize(annotation.font_size)
        if annotation.font_family == "Times New Roman":
            text_property.SetFontFamilyToTimes()
        elif annotation.font_family == "Courier New":
            text_property.SetFontFamilyToCourier()
        else:
            text_property.SetFontFamilyToArial()
        text_property.SetBold(annotation.bold)
        text_property.SetItalic(annotation.italic)
        text_property.SetShadow(True)
        text_property.SetJustificationToCentered()
        text_property.SetVerticalJustificationToCentered()

    def _restore_annotation_actors(self) -> None:
        for annotation in self.annotations.values():
            self._update_annotation_actor(annotation)
            assert annotation.actor is not None
            self.renderer.AddActor(annotation.actor)

    def _setup_annotation_handle(self) -> None:
        representation = vtkSphereHandleRepresentation()
        representation.TranslationModeOn()
        representation.GetProperty().SetColor(1.0, 0.82, 0.20)
        representation.GetSelectedProperty().SetColor(1.0, 0.35, 0.12)
        representation.UseBoundsOff()

        widget = vtkHandleWidget()
        widget.SetInteractor(self.interactor)
        widget.SetRepresentation(representation)
        widget.SetPriority(0.9)
        widget.EnableAxisConstraintOn()
        widget.AllowHandleResizeOff()
        widget.AddObserver(
            "InteractionEvent",
            self._on_annotation_handle_interaction,
        )
        widget.AddObserver(
            "EndInteractionEvent",
            self._on_annotation_handle_interaction,
        )
        widget.SetEnabled(False)
        self.annotation_handle_representation = representation
        self.annotation_handle_widget = widget

    def _annotation_handle_radius(self) -> float:
        x_start, x_stop, y_start, y_stop, z_start, z_stop = self.crop_bounds_xyz
        spacing_x, spacing_y, spacing_z = self._render_spacing
        diagonal = float(
            np.linalg.norm(
                (
                    (x_stop - x_start) * spacing_x,
                    (y_stop - y_start) * spacing_y,
                    (z_stop - z_start) * spacing_z,
                )
            )
        )
        return max(diagonal * 0.012, max(self._render_spacing) * 0.5, 1e-6)

    def _disable_annotation_handle(self) -> None:
        if self.annotation_handle_widget is not None:
            self.annotation_handle_widget.SetEnabled(False)

    def _sync_annotation_handle(
        self,
        checked: bool = False,
        render: bool = True,
    ) -> None:
        del checked
        widget = self.annotation_handle_widget
        representation = self.annotation_handle_representation
        identifier = self._selected_annotation_id()
        enabled = (
            widget is not None
            and representation is not None
            and identifier is not None
            and self.annotation_handle_checkbox.isChecked()
        )
        if not enabled:
            self._disable_annotation_handle()
            if render:
                self.vtk_widget.GetRenderWindow().Render()
            return

        annotation = self.annotations.get(identifier)
        if annotation is None:
            self._disable_annotation_handle()
            return

        x, y, z = annotation.position_xyz
        radius = self._annotation_handle_radius()
        representation.PlaceWidget(
            (
                x - radius,
                x + radius,
                y - radius,
                y + radius,
                z - radius,
                z + radius,
            )
        )
        representation.SetSphereRadius(radius)
        representation.SetWorldPosition(annotation.position_xyz)
        representation.TranslationModeOn()
        representation.GetProperty().SetColor(*annotation.color)
        representation.GetSelectedProperty().SetColor(1.0, 0.35, 0.12)
        widget.SetEnabled(True)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _on_annotation_handle_interaction(
        self,
        caller: vtkHandleWidget,
        event: str,
    ) -> None:
        del caller, event
        if self._updating_annotation_handle:
            return
        identifier = self._selected_annotation_id()
        representation = self.annotation_handle_representation
        if identifier is None or representation is None:
            return
        annotation = self.annotations.get(identifier)
        if annotation is None:
            return

        position = tuple(
            float(value) for value in representation.GetWorldPosition()
        )
        annotation.position_xyz = position
        self._updating_annotation_handle = True
        try:
            for axis, value in zip("XYZ", position):
                self.annotation_position_spins[axis].setValue(value)
            self._update_annotation_actor(annotation)
            item = self.annotation_list.currentItem()
            if item is not None:
                item.setText(self._annotation_item_text(annotation))
        finally:
            self._updating_annotation_handle = False
        self.statusBar().showMessage(
            "3D 标注已拖动到模型坐标："
            f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )

    def reset_camera(
        self,
        checked: bool = False,
        render: bool = True,
    ) -> None:
        del checked
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(35)
        camera.Elevation(22)
        self.renderer.ResetCameraClippingRange()
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _denominator_labels(self) -> tuple[int, ...]:
        text = self.denominator_labels_edit.text().replace("，", ",")
        pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("请至少输入一个分母相编号。")
        try:
            labels = tuple(dict.fromkeys(int(piece) for piece in pieces))
        except ValueError as exc:
            raise ValueError("分母相编号必须是用逗号分隔的整数，例如 3,4,5。") from exc
        return labels

    def calculate_slice_fractions(self, checked: bool = False) -> None:
        del checked
        if self.volume is None:
            QMessageBox.information(self, "尚无数据", "请先打开 TIFF 或演示体。")
            return
        try:
            axis = self.fraction_axis_combo.currentText()
            numerator = self.numerator_label_spin.value()
            denominator = self._denominator_labels()
            slices, fractions = calculate_slice_volume_fraction(
                self.volume,
                axis,
                numerator,
                denominator,
            )
            x_start, _, y_start, _, z_start, _ = self.crop_bounds_xyz
            offset = {"X": x_start, "Y": y_start, "Z": z_start}[axis]
            slices = slices + offset
        except Exception as exc:
            QMessageBox.warning(self, "无法计算", str(exc))
            return

        self.slice_fraction_results = slices, fractions
        self.slice_fraction_axis = axis
        denominator_text = "+".join(str(label) for label in denominator)
        self.fraction_result_label.setText(
            f"{axis} 方向：count({numerator}) / count({denominator_text})；"
            f"共 {len(slices)} 层（显示原始切片编号）"
        )
        self.fraction_table.setSortingEnabled(False)
        self.fraction_table.setRowCount(len(slices))
        for row, (slice_number, fraction) in enumerate(zip(slices, fractions)):
            slice_item = QTableWidgetItem(str(int(slice_number)))
            fraction_text = "N/A" if np.isnan(fraction) else f"{fraction:.8f}"
            fraction_item = QTableWidgetItem(fraction_text)
            slice_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            fraction_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.fraction_table.setItem(row, 0, slice_item)
            self.fraction_table.setItem(row, 1, fraction_item)
        self.results_dock.show()
        self.results_dock.raise_()
        self.statusBar().showMessage(
            f"逐层体积分数计算完成：{axis} 方向，共 {len(slices)} 层"
        )

    def save_fraction_csv(self, checked: bool = False) -> None:
        del checked
        if self.slice_fraction_results is None:
            QMessageBox.information(self, "尚无结果", "请先执行逐层体积分数计算。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存逐层体积分数",
            f"slice_volume_fraction_{self.slice_fraction_axis}.csv",
            "CSV 文件 (*.csv)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        slices, fractions = self.slice_fraction_results
        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Original Slice Number", "Volume Fraction"])
                for slice_number, fraction in zip(slices, fractions):
                    value = "NaN" if np.isnan(fraction) else f"{fraction:.10g}"
                    writer.writerow([int(slice_number), value])
        except OSError as exc:
            QMessageBox.warning(self, "CSV 保存失败", str(exc))
            return
        self.statusBar().showMessage(f"CSV 已保存：{filename}")

    def run_batch_folder(self, checked: bool = False) -> None:
        del checked
        if not self.batch_tiff_paths or self.batch_folder is None:
            QMessageBox.information(self, "尚无文件夹", "请先选择 TIFF 文件夹。")
            return
        if self.current_tiff_path not in self.batch_tiff_paths:
            QMessageBox.information(
                self,
                "参考 TIFF 未载入",
                "请先从批量列表载入一个参考 TIFF，再设置裁剪和显示参数。",
            )
            return
        if self.source_volume_info is None or self.volume_info is None:
            return

        try:
            axis = self.fraction_axis_combo.currentText()
            numerator = self.numerator_label_spin.value()
            denominator = self._denominator_labels()
        except ValueError as exc:
            QMessageBox.warning(self, "批量参数错误", str(exc))
            return

        destination = QFileDialog.getExistingDirectory(
            self,
            "选择批量结果保存位置",
            str(self.batch_folder),
        )
        if not destination:
            return
        output_parent = Path(destination)
        output_root = output_parent / "phase_viewer_batch_output"
        suffix = 2
        while output_root.exists():
            output_root = output_parent / f"phase_viewer_batch_output_{suffix}"
            suffix += 1
        image_dir = output_root / "visualizations"
        fraction_dir = output_root / "slice_volume_fraction"
        try:
            image_dir.mkdir(parents=True, exist_ok=True)
            fraction_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "无法创建输出文件夹", str(exc))
            return

        reference_shape = self.source_volume_info.shape_zyx
        reference_background = self.background_label
        template_cache = copy.deepcopy(self._phase_style_cache)
        scale = int(self.screenshot_scale_combo.currentData())
        x_start, _, y_start, _, z_start, _ = self.crop_bounds_xyz
        offset = {"X": x_start, "Y": y_start, "Z": z_start}[axis]

        saved_state = {
            "source_volume": self.source_volume,
            "source_volume_info": self.source_volume_info,
            "volume": self.volume,
            "volume_info": self.volume_info,
            "source_name": self.source_name,
            "source_axes": self.source_axes,
            "background_label": self.background_label,
            "phase_styles": self.phase_styles,
            "phase_style_cache": self._phase_style_cache,
            "current_tiff_path": self.current_tiff_path,
        }
        saved_camera = vtkCamera()
        saved_camera.DeepCopy(self.renderer.GetActiveCamera())

        combined_path = output_root / f"all_slice_volume_fraction_{axis}.csv"
        manifest_path = output_root / "batch_manifest.csv"
        settings_path = output_root / "batch_settings.csv"
        manifest_rows: list[list[object]] = []
        succeeded = 0
        failed = 0
        canceled = False
        fatal_error: str | None = None
        progress = QProgressDialog(
            "准备批量处理…",
            "取消",
            0,
            len(self.batch_tiff_paths),
            self,
        )
        progress.setWindowTitle("TIFF 文件夹批量处理")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            with settings_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as settings_stream:
                settings_writer = csv.writer(settings_stream)
                settings_writer.writerow(["Setting", "Value"])
                settings_writer.writerows(
                    [
                        ["Application Version", APP_VERSION],
                        ["Input Folder", str(self.batch_folder)],
                        ["Reference TIFF", str(self.current_tiff_path)],
                        ["TIFF Count", len(self.batch_tiff_paths)],
                        ["Original Shape ZYX", " x ".join(map(str, reference_shape))],
                        [
                            "Crop XYZ (1-based inclusive)",
                            f"X {x_start + 1}-{self.crop_bounds_xyz[1]}; "
                            f"Y {y_start + 1}-{self.crop_bounds_xyz[3]}; "
                            f"Z {z_start + 1}-{self.crop_bounds_xyz[5]}",
                        ],
                        [
                            "Working Shape ZYX",
                            " x ".join(map(str, self.volume_info.shape_zyx)),
                        ],
                        [
                            "Spacing XYZ",
                            " x ".join(map(str, self.current_spacing())),
                        ],
                        ["Fraction Axis", axis],
                        ["Numerator Label", numerator],
                        [
                            "Denominator Labels",
                            ",".join(map(str, denominator)),
                        ],
                        ["Screenshot Scale", f"{scale}x"],
                    ]
                )

            with combined_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as combined_stream:
                combined_writer = csv.writer(combined_stream)
                combined_writer.writerow(
                    ["Source File", "Original Slice Number", "Volume Fraction"]
                )

                for index, path in enumerate(self.batch_tiff_paths, start=1):
                    progress.setValue(index - 1)
                    progress.setLabelText(
                        f"正在处理 {index}/{len(self.batch_tiff_paths)}：{path.name}"
                    )
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        canceled = True
                        break

                    output_stem = f"{index:04d}_{path.stem}"
                    image_path = image_dir / f"{output_stem}.png"
                    fraction_path = (
                        fraction_dir
                        / f"{output_stem}_slice_volume_fraction_{axis}.csv"
                    )
                    try:
                        read_result = self._read_tiff_series(
                            path,
                            warn_large=False,
                        )
                        if read_result is None:
                            raise ValueError("读取已取消。")
                        array, axes = read_result
                        source_volume = normalize_label_volume(array)
                        source_info = analyze_volume(source_volume)
                        if source_info.shape_zyx != reference_shape:
                            raise ValueError(
                                "原始形状不一致："
                                f"期望 {reference_shape}，实际 {source_info.shape_zyx}。"
                            )

                        working_volume = crop_label_volume(
                            source_volume,
                            self.crop_bounds_xyz,
                        )
                        working_info = analyze_volume(working_volume)
                        batch_cache: dict[int, PhaseStyle] = {}
                        for label_index, label in enumerate(source_info.labels):
                            template = template_cache.get(label)
                            batch_cache[label] = (
                                copy.deepcopy(template)
                                if template is not None
                                else PhaseStyle(
                                    label=label,
                                    visible=False,
                                    color=PALETTE[label_index % len(PALETTE)],
                                )
                            )

                        self.source_volume = source_volume
                        self.source_volume_info = source_info
                        self.volume = working_volume
                        self.volume_info = working_info
                        self.source_name = path.name
                        self.source_axes = axes or "未知"
                        self.background_label = reference_background
                        self._phase_style_cache = batch_cache
                        self.phase_styles = {
                            label: batch_cache[label]
                            for label in working_info.labels
                        }
                        self.rebuild_all(
                            reset_camera=False,
                            raise_errors=True,
                        )
                        self.renderer.GetActiveCamera().DeepCopy(saved_camera)
                        self.renderer.ResetCameraClippingRange()
                        self.vtk_widget.GetRenderWindow().Render()
                        self._write_render_png(image_path, scale)

                        slices, fractions = calculate_slice_volume_fraction(
                            working_volume,
                            axis,
                            numerator,
                            denominator,
                        )
                        slices = slices + offset
                        with fraction_path.open(
                            "w",
                            newline="",
                            encoding="utf-8-sig",
                        ) as fraction_stream:
                            fraction_writer = csv.writer(fraction_stream)
                            fraction_writer.writerow(
                                ["Original Slice Number", "Volume Fraction"]
                            )
                            for slice_number, fraction in zip(slices, fractions):
                                value = (
                                    "NaN"
                                    if np.isnan(fraction)
                                    else f"{fraction:.10g}"
                                )
                                fraction_writer.writerow(
                                    [int(slice_number), value]
                                )
                                combined_writer.writerow(
                                    [path.name, int(slice_number), value]
                                )

                        succeeded += 1
                        manifest_rows.append(
                            [
                                index,
                                path.name,
                                str(image_path.relative_to(output_root)),
                                str(fraction_path.relative_to(output_root)),
                                "OK",
                                "",
                            ]
                        )
                    except Exception as exc:
                        failed += 1
                        manifest_rows.append(
                            [index, path.name, "", "", "FAILED", str(exc)]
                        )
                    finally:
                        QApplication.processEvents()

            with manifest_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as manifest_stream:
                manifest_writer = csv.writer(manifest_stream)
                manifest_writer.writerow(
                    [
                        "Index",
                        "Source File",
                        "Visualization PNG",
                        "Slice Fraction CSV",
                        "Status",
                        "Message",
                    ]
                )
                manifest_writer.writerows(manifest_rows)
        except OSError as exc:
            fatal_error = str(exc)
            QMessageBox.warning(self, "批量输出失败", str(exc))
        finally:
            progress.setValue(len(self.batch_tiff_paths))
            progress.close()
            self.source_volume = saved_state["source_volume"]
            self.source_volume_info = saved_state["source_volume_info"]
            self.volume = saved_state["volume"]
            self.volume_info = saved_state["volume_info"]
            self.source_name = saved_state["source_name"]
            self.source_axes = saved_state["source_axes"]
            self.background_label = saved_state["background_label"]
            self.phase_styles = saved_state["phase_styles"]
            self._phase_style_cache = saved_state["phase_style_cache"]
            self.current_tiff_path = saved_state["current_tiff_path"]
            self._update_volume_information()
            self.rebuild_all(reset_camera=False)
            self.renderer.GetActiveCamera().DeepCopy(saved_camera)
            self.renderer.ResetCameraClippingRange()
            self.vtk_widget.GetRenderWindow().Render()

        if fatal_error is not None:
            self.batch_folder_label.setText(
                f"{self.batch_folder}\n批量输出失败：{fatal_error}\n"
                f"输出位置：{output_root}"
            )
            return

        status = "；用户已取消" if canceled else ""
        self.batch_folder_label.setText(
            f"{self.batch_folder}\n共 {len(self.batch_tiff_paths)} 个 TIFF；"
            f"成功 {succeeded}，失败 {failed}{status}\n输出：{output_root}"
        )
        QMessageBox.information(
            self,
            "批量处理完成" if not canceled else "批量处理已停止",
            f"成功：{succeeded}\n失败：{failed}\n输出目录：{output_root}",
        )

    def save_screenshot(self, checked: bool = False) -> None:
        del checked
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存模型截图",
            "phase_model.png",
            "PNG 图片 (*.png)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".png"):
            filename += ".png"
        scale = int(self.screenshot_scale_combo.currentData())
        try:
            width, height = self._write_render_png(Path(filename), scale)
        except Exception as exc:
            QMessageBox.warning(self, "截图失败", str(exc))
            return
        self.statusBar().showMessage(
            f"模型截图已保存：{filename}（{width} × {height}）"
        )

    def _write_render_png(self, path: Path, scale: int) -> tuple[int, int]:
        render_window = self.vtk_widget.GetRenderWindow()
        handle_was_enabled = bool(
            self.annotation_handle_widget is not None
            and self.annotation_handle_widget.GetEnabled()
        )
        saved_font_sizes: list[tuple[object, int]] = []
        try:
            if handle_was_enabled:
                self._disable_annotation_handle()

            # vtkBillboardTextActor3D uses a screen-pixel font size. VTK's
            # WindowToImageFilter supersamples the geometry but does not scale
            # that font size automatically, so compensate only while writing.
            for annotation in self.annotations.values():
                if annotation.actor is None:
                    continue
                text_property = annotation.actor.GetTextProperty()
                original_size = int(text_property.GetFontSize())
                saved_font_sizes.append((text_property, original_size))
                text_property.SetFontSize(
                    scaled_capture_font_size(original_size, scale)
                )

            render_window.Render()
            capture = vtkWindowToImageFilter()
            capture.SetInput(render_window)
            capture.SetScale(scale)
            capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff()
            capture.Modified()
            capture.Update()

            writer = vtkPNGWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            if writer.GetErrorCode() != 0:
                raise OSError("VTK 无法写入指定的 PNG 文件。")
        finally:
            for text_property, original_size in saved_font_sizes:
                text_property.SetFontSize(original_size)
            if handle_was_enabled:
                self._sync_annotation_handle(render=False)
            render_window.Render()

        width, height = render_window.GetSize()
        return width * scale, height * scale

    def closeEvent(self, event: QCloseEvent) -> None:
        self._disable_annotation_handle()
        if hasattr(self.interactor, "TerminateApp"):
            self.interactor.TerminateApp()
        if hasattr(self.vtk_widget, "Finalize"):
            self.vtk_widget.Finalize()
        super().closeEvent(event)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="显示、平滑并分析多相标签 TIFF 的 Surface Nets 表面。"
    )
    parser.add_argument("tiff", nargs="?", type=Path, help="可选 TIFF 路径")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
        help="X/Y/Z 方向体素尺寸，默认均为 1",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="载入内置演示体；未提供 TIFF 时默认启用",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        info = analyze_volume(make_demo_volume(size=24))
        if len(info.shape_zyx) != 3 or not info.labels:
            raise RuntimeError("Packaged dependency self-test failed.")
        print(f"LBM_post_process V{APP_VERSION} self-test passed")
        return 0
    if any(value <= 0 for value in args.spacing):
        raise SystemExit("--spacing 的三个数必须都大于 0。")

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("LBM")
    app.setApplicationName("LBM_post_process")
    app.setStyle("Fusion")
    window = PhaseViewer(
        initial_path=args.tiff,
        initial_spacing=tuple(args.spacing),
        use_demo=args.demo,
    )
    window.show()
    QTimer.singleShot(
        100,
        lambda: window.resizeDocks(
            [window.phase_dock, window.properties_dock],
            [430, 365],
            Qt.Orientation.Horizontal,
        ),
    )
    QTimer.singleShot(
        150,
        lambda: window.resizeDocks(
            [window.properties_dock, window.analysis_dock],
            [570, 260],
            Qt.Orientation.Vertical,
        ),
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
