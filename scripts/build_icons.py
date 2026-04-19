"""Generate release icons from the source PNG."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image


ICONSET_VARIANTS = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="assets/app-icon.png",
        help="Source icon PNG.",
    )
    parser.add_argument(
        "--out-dir",
        default="build/icons",
        help="Output directory for generated icon assets.",
    )
    return parser.parse_args()


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def write_png_variants(image: Image.Image, out_dir: Path) -> None:
    largest_png = out_dir / "app-icon-1024.png"
    resize_square(image, 1024).save(largest_png)


def write_ico(image: Image.Image, out_dir: Path) -> Path:
    ico_path = out_dir / "app-icon.ico"
    resize_square(image, 256).save(ico_path, format="ICO", sizes=ICO_SIZES)
    return ico_path


def write_iconset(image: Image.Image, out_dir: Path) -> Path:
    iconset_dir = out_dir / "app.iconset"
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for filename, size in ICONSET_VARIANTS:
        resize_square(image, size).save(iconset_dir / filename)
    return iconset_dir


def write_icns(image: Image.Image, iconset_dir: Path, out_dir: Path) -> Path | None:
    icns_path = out_dir / "app-icon.icns"
    try:
        image.save(
            icns_path,
            format="ICNS",
            sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
        )
        return icns_path
    except OSError:
        pass

    if sys.platform != "darwin":
        return None

    iconutil = subprocess.run(
        ["xcrun", "--find", "iconutil"],
        check=False,
        capture_output=True,
        text=True,
    )
    if iconutil.returncode != 0:
        return None

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)],
        check=True,
    )
    return icns_path


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    source = (root / args.source).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as raw_image:
        image = raw_image.convert("RGBA")
        write_png_variants(image, out_dir)
        ico_path = write_ico(image, out_dir)
        iconset_dir = write_iconset(image, out_dir)
        icns_path = write_icns(image, iconset_dir, out_dir)

    print(f"generated ico: {ico_path}")
    if icns_path is not None:
        print(f"generated icns: {icns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
