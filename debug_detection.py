"""Generate visual debug images for contour-based crop detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from image_core import ImageCore


def parse_aspect(text: str) -> float:
    value = text.strip()
    if ":" in value:
        left, right = value.split(":", 1)
        return float(left) / float(right)
    if "/" in value:
        left, right = value.split("/", 1)
        return float(left) / float(right)
    return float(value)


def iter_tiff_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def add_label(image_rgb: np.ndarray, label: str) -> np.ndarray:
    canvas = image_rgb.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


def make_contact_sheet(bundle) -> np.ndarray:
    overlay = add_label(bundle.overlay_rgb, "Overlay")
    threshold = add_label(to_rgb(bundle.threshold_mask), "Threshold")
    edges = add_label(to_rgb(bundle.edge_mask), "Edges")
    return np.concatenate([overlay, threshold, edges], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize CV contour detection on TIFF proxy images.")
    parser.add_argument("input_path", help="A TIFF file or a folder containing TIFF files.")
    parser.add_argument("--aspect", default="3:2", help="Target aspect ratio, e.g. 3:2, 1:1, 6:7.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for debug PNG output. Defaults to Detection_Debug next to the input.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.05,
        help="Minimum contour area ratio used by the detector.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    aspect_ratio = parse_aspect(args.aspect)
    image_core = ImageCore(proxy_max_long_edge=2000)
    files = iter_tiff_files(input_path)
    if not files:
        raise FileNotFoundError(f"No TIFF files found under {input_path}")

    base_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "Detection_Debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, file_path in enumerate(files, start=1):
        bundle = image_core.load_and_prepare_proxy(file_path)
        debug_bundle = image_core.build_detection_debug_bundle(
            proxy_8bit=bundle.proxy_8bit,
            target_aspect=aspect_ratio,
            min_area_ratio=args.min_area_ratio,
        )
        sheet = make_contact_sheet(debug_bundle)
        output_path = output_dir / f"{file_path.stem}_debug.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        chosen = debug_bundle.chosen_state
        if chosen is None:
            print(f"[{index}/{len(files)}] {file_path.name} -> no candidate, saved {output_path}")
        else:
            print(
                f"[{index}/{len(files)}] {file_path.name} -> "
                f"cx={chosen.cx:.1f}, cy={chosen.cy:.1f}, w={chosen.width:.1f}, "
                f"h={chosen.height:.1f}, angle={chosen.angle_deg:.1f} | {output_path}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
