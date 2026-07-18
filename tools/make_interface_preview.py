"""Generate a static layout preview for the multi-phase viewer UI."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH, HEIGHT = 1600, 920
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "interface_preview_v5.png"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill="#f4f5f7", outline="#aeb4bd")
    draw.rectangle((x0, y0, x1, y0 + 34), fill="#dfe3e8")
    draw.text((x0 + 12, y0 + 8), title, font=font(15, True), fill="#24282e")


def checkbox(draw: ImageDraw.ImageDraw, x: int, y: int, checked: bool = True) -> None:
    draw.rounded_rectangle((x, y, x + 16, y + 16), radius=2, fill="white", outline="#737b86")
    if checked:
        draw.line((x + 3, y + 8, x + 7, y + 12, x + 14, y + 3), fill="#2d76c2", width=3)


def button(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: str = "#e6e9ed",
    text_fill: str = "#22262c",
) -> None:
    draw.rounded_rectangle(box, radius=4, fill=fill, outline="#9aa1aa")
    x0, y0, x1, y1 = box
    bounds = draw.textbbox((0, 0), text, font=font(13))
    tw = bounds[2] - bounds[0]
    th = bounds[3] - bounds[1]
    draw.text(
        ((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - 1),
        text,
        font=font(13),
        fill=text_fill,
    )


def gradient_view(base: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    height = y1 - y0
    grad = Image.new("RGB", (x1 - x0, height))
    gd = ImageDraw.Draw(grad)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(
            round(a * (1 - t) + b * t)
            for a, b in zip((43, 48, 61), (10, 13, 20))
        )
        gd.line((0, y, x1 - x0, y), fill=color)
    base.paste(grad, (x0, y0))


def blob_layer(
    size: tuple[int, int],
    center: tuple[int, int],
    radii: tuple[int, int],
    color: tuple[int, int, int, int],
    angle: float = 0,
) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    rx, ry = radii
    points = []
    for i in range(96):
        theta = 2 * math.pi * i / 96
        wobble = 1.0 + 0.08 * math.sin(theta * 3 + angle) + 0.04 * math.cos(theta * 7)
        x = cx + rx * wobble * math.cos(theta)
        y = cy + ry * wobble * math.sin(theta)
        points.append((x, y))
    draw.polygon(points, fill=color)
    highlight = (
        cx - int(rx * 0.45),
        cy - int(ry * 0.52),
        cx + int(rx * 0.12),
        cy + int(ry * 0.02),
    )
    draw.ellipse(highlight, fill=(255, 255, 255, 45))
    return layer.filter(ImageFilter.GaussianBlur(1.1))


def jagged_blob(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: tuple[int, int],
    color: tuple[int, int, int, int],
) -> None:
    cx, cy = center
    rx, ry = radius
    points = []
    for i in range(48):
        theta = 2 * math.pi * i / 48
        x = round((cx + rx * math.cos(theta)) / 9) * 9
        y = round((cy + ry * math.sin(theta)) / 9) * 9
        points.append((x, y))
    draw.polygon(points, fill=color, outline=(225, 235, 245, 180), width=2)


def draw_scene(image: Image.Image, box: tuple[int, int, int, int], smooth: bool) -> None:
    x0, y0, x1, y1 = box
    gradient_view(image, box)
    draw = ImageDraw.Draw(image, "RGBA")
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2 + 10

    grid = (112, 124, 145, 45)
    for offset in range(-240, 260, 48):
        draw.line((x0 + 30, mid_y + offset // 3, x1 - 30, mid_y + offset // 3), fill=grid)
    for offset in range(-220, 240, 55):
        draw.line((mid_x + offset, y0 + 70, mid_x + offset, y1 - 55), fill=grid)

    if smooth:
        layers = [
            blob_layer(image.size, (mid_x - 30, mid_y + 20), (155, 128), (51, 167, 224, 198), 0.2),
            blob_layer(image.size, (mid_x + 70, mid_y - 42), (112, 84), (245, 143, 46, 190), 1.4),
            blob_layer(image.size, (mid_x + 22, mid_y + 82), (86, 58), (82, 191, 111, 190), 2.6),
        ]
        for layer in layers:
            image.alpha_composite(layer)
    else:
        jagged_blob(draw, (mid_x - 30, mid_y + 20), (155, 128), (51, 167, 224, 198))
        jagged_blob(draw, (mid_x + 70, mid_y - 42), (112, 84), (245, 143, 46, 190))
        jagged_blob(draw, (mid_x + 22, mid_y + 82), (86, 58), (82, 191, 111, 190))

    # Current cropped working-volume outline and a billboard-style annotation.
    front = (mid_x - 205, mid_y - 145, mid_x + 205, mid_y + 150)
    back = (front[0] + 34, front[1] - 24, front[2] + 34, front[3] - 24)
    outline_color = (242, 215, 84, 230)
    draw.rectangle(front, outline=outline_color, width=3)
    draw.rectangle(back, outline=outline_color, width=3)
    for first, second in zip(
        ((front[0], front[1]), (front[2], front[1]), (front[0], front[3]), (front[2], front[3])),
        ((back[0], back[1]), (back[2], back[1]), (back[0], back[3]), (back[2], back[3])),
    ):
        draw.line((*first, *second), fill=outline_color, width=3)

    marker = (mid_x + 80, mid_y - 82)
    draw.ellipse(
        (marker[0] - 9, marker[1] - 9, marker[0] + 9, marker[1] + 9),
        fill=(255, 92, 31, 255),
        outline=(255, 233, 104, 255),
        width=3,
    )
    draw.text((marker[0] + 10, marker[1] - 26), "Region A", font=font(19, True), fill=(255, 221, 73, 255))

    draw.rectangle((x0, y0, x1, y1), outline=(130, 140, 155, 255), width=1)
    title = "UNSMOOTHED LABEL SURFACES" if not smooth else "MULTI-PHASE SMOOTHED SURFACES"
    draw.text((x0 + 18, y0 + 17), title, font=font(15, True), fill=(243, 246, 252, 255))
    subtitle = "Interactive multi-phase view • Smooth surfaces"
    draw.text((x0 + 18, y0 + 40), subtitle, font=font(12), fill=(185, 195, 210, 255))

    # Compact orientation triad.
    ox, oy = x0 + 48, y1 - 42
    draw.line((ox, oy, ox + 25, oy), fill=(229, 83, 83, 255), width=3)
    draw.line((ox, oy, ox - 12, oy - 21), fill=(91, 206, 124, 255), width=3)
    draw.line((ox, oy, ox, oy - 30), fill=(91, 148, 232, 255), width=3)
    draw.text((ox + 27, oy - 8), "X", font=font(11, True), fill="white")
    draw.text((ox - 22, oy - 31), "Y", font=font(11, True), fill="white")
    draw.text((ox + 5, oy - 39), "Z", font=font(11, True), fill="white")


def main() -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), "#d2d6dc")
    draw = ImageDraw.Draw(image, "RGBA")

    # Window chrome, menu and toolbar.
    draw.rectangle((0, 0, WIDTH, 38), fill="#333942")
    draw.text((18, 9), "TIFF Multi-Phase Surface Viewer V0.5.2", font=font(16, True), fill="white")
    draw.text((1325, 10), "UI LAYOUT PREVIEW", font=font(13, True), fill="#9fd7ff")
    draw.rectangle((0, 38, WIDTH, 70), fill="#f3f4f6")
    draw.text((14, 47), "File     View", font=font(13), fill="#24282e")
    draw.rectangle((0, 70, WIDTH, 112), fill="#e6e9ed")
    button(draw, (14, 77, 100, 105), "Open TIFF")
    button(draw, (108, 77, 222, 105), "Open Folder")
    button(draw, (230, 77, 298, 105), "Demo")
    button(draw, (313, 77, 389, 105), "Rebuild")
    button(draw, (397, 77, 474, 105), "Capture")

    content_top, content_bottom = 112, 892
    left_x1, right_x0 = 410, 1240

    panel(draw, (0, content_top, left_x1, content_bottom), "Phase List / Pipeline")
    table_y = 154
    draw.rectangle((8, table_y, left_x1 - 8, table_y + 34), fill="#e8ebef", outline="#aeb4bd")
    headers = [("Vis", 14), ("Phase", 56), ("Color", 125), ("Opacity", 202), ("Smooth", 282), ("Light", 354)]
    for text, x in headers:
        draw.text((x, table_y + 9), text, font=font(12, True), fill="#31363d")

    rows = [
        (0, False, "#7b8592", "0.00", False, True),
        (3, True, "#38a5dc", "0.82", True, True),
        (4, True, "#f48f2f", "0.66", True, True),
        (5, True, "#55bf70", "0.54", False, False),
    ]
    y = table_y + 35
    for index, (phase, visible, color, opacity, smooth, lighting) in enumerate(rows):
        fill = "#dfeeff" if index == 1 else ("#ffffff" if index % 2 else "#f6f7f9")
        draw.rectangle((8, y, left_x1 - 8, y + 43), fill=fill, outline="#d3d7dc")
        checkbox(draw, 23, y + 13, visible)
        draw.text((75, y + 11), str(phase), font=font(14), fill="#24282e")
        draw.rounded_rectangle((126, y + 8, 181, y + 34), radius=4, fill=color, outline="#777")
        draw.text((214, y + 11), opacity, font=font(13), fill="#24282e")
        checkbox(draw, 303, y + 13, smooth)
        checkbox(draw, 366, y + 13, lighting)
        y += 43

    draw.text((15, y + 22), "Volume fraction and phase metadata appear on hover.", font=font(11), fill="#626a74")
    button(draw, (12, content_bottom - 49, 131, content_bottom - 14), "Materials")
    button(draw, (143, content_bottom - 49, 262, content_bottom - 14), "Show all")
    button(draw, (274, content_bottom - 49, 398, content_bottom - 14), "Hide all")

    model_bottom = 650
    draw_scene(image, (left_x1, content_top, right_x0, model_bottom), smooth=True)

    panel(draw, (left_x1, model_bottom, right_x0, content_bottom), "Slice Volume Fraction • Z axis • count(3) / count(3+4+5)")
    table_left = left_x1 + 12
    table_top = model_bottom + 43
    table_right = right_x0 - 12
    draw.rectangle((table_left, table_top, table_right, table_top + 28), fill="#e8ebef", outline="#aeb4bd")
    draw.text((table_left + 20, table_top + 6), "Original Slice Number", font=font(12, True), fill="#31363d")
    draw.text((table_left + 235, table_top + 6), "Volume Fraction", font=font(12, True), fill="#31363d")
    fractions = [(31, "0.42187500"), (32, "0.43750000"), (33, "0.45161290"), (34, "0.46875000"), (35, "0.48275862")]
    row_y = table_top + 28
    for index, (slice_number, fraction) in enumerate(fractions):
        fill = "#ffffff" if index % 2 else "#f6f7f9"
        draw.rectangle((table_left, row_y, table_right, row_y + 27), fill=fill, outline="#d3d7dc")
        draw.text((table_left + 50, row_y + 5), str(slice_number), font=font(12), fill="#24282e")
        draw.text((table_left + 260, row_y + 5), fraction, font=font(12), fill="#24282e")
        row_y += 27
    button(draw, (table_right - 115, content_bottom - 42, table_right, content_bottom - 10), "Save CSV")

    panel(draw, (right_x0, content_top, WIDTH, content_bottom), "Properties")
    px = right_x0 + 12
    py = content_top + 48
    draw.text((px, py), "DATA & GLOBAL DISPLAY", font=font(12, True), fill="#4c5661")
    py += 29
    properties = [
        ("Dataset", "labels_200.tif"),
        ("Original XYZ", "200 × 200 × 200"),
        ("Working XYZ", "180 × 200 × 170"),
        ("Spacing XYZ", "0.4, 0.4, 1.2"),
    ]
    for name, value in properties:
        draw.text((px, py + 4), name, font=font(12), fill="#444b54")
        draw.rounded_rectangle((right_x0 + 130, py, WIDTH - 15, py + 27), radius=3, fill="white", outline="#b7bdc5")
        draw.text((right_x0 + 141, py + 5), value, font=font(12), fill="#23272d")
        py += 34

    checkbox(draw, px, py + 3, True)
    draw.text((px + 24, py + 2), "Gradient background", font=font(12), fill="#333941")
    draw.rounded_rectangle((WIDTH - 112, py - 1, WIDTH - 67, py + 24), radius=3, fill="#111722", outline="#777")
    draw.rounded_rectangle((WIDTH - 59, py - 1, WIDTH - 14, py + 24), radius=3, fill="#273045", outline="#777")
    py += 34

    draw.text((px, py + 4), "Frame color", font=font(12), fill="#444b54")
    draw.rounded_rectangle((right_x0 + 130, py, WIDTH - 15, py + 27), radius=3, fill="#f2d754", outline="#777")
    draw.text((right_x0 + 141, py + 5), "#F2D754", font=font(12), fill="#23272d")
    py += 32
    draw.text((px, py + 4), "Frame width", font=font(12), fill="#444b54")
    draw.rounded_rectangle((right_x0 + 130, py, WIDTH - 15, py + 27), radius=3, fill="white", outline="#b7bdc5")
    draw.text((right_x0 + 141, py + 5), "3.0", font=font(12), fill="#23272d")
    py += 39

    draw.line((px, py, WIDTH - 13, py), fill="#bdc2c9", width=1)
    py += 15
    draw.text((px, py), "CROP WORKING VOLUME", font=font(12, True), fill="#4c5661")
    py += 25
    crop_props = [("X range", "1 – 180"), ("Y range", "1 – 200"), ("Z range", "31 – 200")]
    for name, value in crop_props:
        draw.text((px, py + 4), name, font=font(12), fill="#444b54")
        draw.rounded_rectangle((right_x0 + 130, py, WIDTH - 15, py + 27), radius=3, fill="white", outline="#b7bdc5")
        draw.text((right_x0 + 141, py + 5), value, font=font(12), fill="#23272d")
        py += 31
    button(draw, (px, py + 2, WIDTH - 14, py + 35), "Apply crop & rebuild", fill="#d9eafd")
    py += 48

    draw.line((px, py, WIDTH - 13, py), fill="#bdc2c9", width=1)
    py += 15
    draw.text((px, py), "TIFF FOLDER BATCH", font=font(12, True), fill="#4c5661")
    py += 25
    batch_props = [
        ("Folder", "experiment_01/ (24 TIFF)"),
        ("Reference", "sample_001.tif"),
        ("Output", "PNG + per-file CSV"),
    ]
    for name, value in batch_props:
        draw.text((px, py + 4), name, font=font(12), fill="#444b54")
        draw.rounded_rectangle((right_x0 + 130, py, WIDTH - 15, py + 27), radius=3, fill="white", outline="#b7bdc5")
        draw.text((right_x0 + 141, py + 5), value, font=font(12), fill="#23272d")
        py += 31
    button(draw, (px, py + 2, WIDTH - 14, py + 35), "Batch export all TIFF files", fill="#d9eafd")

    # Status bar.
    draw.rectangle((0, 892, WIDTH, HEIGHT), fill="#f4f5f7")
    draw.line((0, 892, WIDTH, 892), fill="#aeb4bd")
    draw.text((12, 899), "Preset saved • Scroll guard on • 24 TIFF files • Preview 1×", font=font(12), fill="#3e454e")
    draw.text((1322, 899), "Surface Nets • Ready", font=font(12), fill="#3e454e")

    preview = image.convert("RGB").resize((1200, 690), Image.Resampling.LANCZOS)
    preview.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
