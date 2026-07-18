"""Generate the LBM_post_process PNG and multi-resolution Windows icon."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PNG_PATH = ASSETS / "lbm_post_process.png"
ICO_PATH = ASSETS / "lbm_post_process.ico"
SIZE = 512


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    # Dark scientific-visualization tile with a subtle vertical gradient.
    for y in range(28, 484):
        t = (y - 28) / 456
        color = (
            round(23 * (1 - t) + 8 * t),
            round(52 * (1 - t) + 22 * t),
            round(78 * (1 - t) + 42 * t),
            255,
        )
        draw.line((28, y, 484, y), fill=color, width=1)
    draw.rounded_rectangle(
        (28, 28, 484, 484),
        radius=92,
        outline=(117, 215, 241, 255),
        width=12,
    )

    # Three phase regions and a crop frame echo the application's 3-D view.
    draw.ellipse((74, 98, 268, 292), fill=(47, 171, 224, 236))
    draw.ellipse((224, 91, 420, 287), fill=(244, 145, 48, 236))
    draw.ellipse((158, 220, 354, 416), fill=(76, 190, 113, 236))
    draw.rounded_rectangle(
        (84, 82, 428, 424),
        radius=24,
        outline=(255, 222, 84, 255),
        width=9,
    )

    title_font = load_font(92, bold=True)
    label = "LBM"
    bounds = draw.textbbox((0, 0), label, font=title_font, stroke_width=3)
    text_width = bounds[2] - bounds[0]
    draw.text(
        ((SIZE - text_width) / 2, 196),
        label,
        font=title_font,
        fill=(255, 255, 255, 255),
        stroke_width=4,
        stroke_fill=(11, 35, 52, 230),
    )
    draw.text(
        (151, 315),
        "POST",
        font=load_font(45, bold=True),
        fill=(233, 244, 249, 255),
        stroke_width=2,
        stroke_fill=(11, 35, 52, 220),
    )

    image.save(PNG_PATH)
    image.save(
        ICO_PATH,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(PNG_PATH)
    print(ICO_PATH)


if __name__ == "__main__":
    main()
