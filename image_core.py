"""Core image pipeline for 16-bit TIFF film copy cropping.

Step-1 scope:
- Load original 16-bit TIFF safely
- Build an 8-bit proxy image for UI/fast CV
- Run a basic contour-based initial crop detection on proxy space
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tifffile


@dataclass
class ProxyImageBundle:
    """Container for original/proxy data and transform metadata."""

    file_path: Path
    original_16bit: np.ndarray
    proxy_8bit: np.ndarray
    proxy_scale: float  # proxy_pixel = original_pixel * proxy_scale


@dataclass
class CropState:
    """Crop rectangle in proxy coordinate system."""

    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float = 0.0


class ImageCore:
    """Image processing utilities for the film crop reviewer."""

    def __init__(self, proxy_max_long_edge: int = 2000) -> None:
        if proxy_max_long_edge <= 0:
            raise ValueError("proxy_max_long_edge must be > 0")
        self.proxy_max_long_edge = proxy_max_long_edge

    def load_tiff_16bit(self, file_path: str | Path) -> np.ndarray:
        """Load TIFF as-is to preserve 16-bit data.

        Prefers tifffile for exact dtype preservation and broad TIFF support.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        image = tifffile.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read TIFF: {path}")

        if image.dtype != np.uint16:
            # Some workflows may produce other dtypes; for this tool we enforce 16-bit.
            raise ValueError(f"Expected uint16 TIFF, got {image.dtype} for {path}")

        return image

    def build_proxy_8bit(self, img16: np.ndarray) -> Tuple[np.ndarray, float]:
        """Create an 8-bit resized proxy image from a 16-bit image.

        Returns:
            proxy_8bit: HxW (grayscale) or HxWxC (color), uint8
            scale: proxy/original scale factor in x,y (uniform)
        """
        if img16.dtype != np.uint16:
            raise ValueError("build_proxy_8bit expects uint16 input")

        # normalize 16-bit -> 8-bit while preserving contrast range
        img16_f = img16.astype(np.float32)
        min_v = float(np.min(img16_f))
        max_v = float(np.max(img16_f))
        if max_v <= min_v:
            norm = np.zeros_like(img16_f, dtype=np.uint8)
        else:
            norm_f = (img16_f - min_v) * (255.0 / (max_v - min_v))
            norm = np.clip(norm_f, 0, 255).astype(np.uint8)

        h, w = norm.shape[:2]
        long_edge = max(h, w)
        if long_edge <= self.proxy_max_long_edge:
            return norm, 1.0

        scale = self.proxy_max_long_edge / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        proxy = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return proxy, scale

    def load_and_prepare_proxy(self, file_path: str | Path) -> ProxyImageBundle:
        """Load 16-bit TIFF and immediately construct a proxy image."""
        path = Path(file_path)
        img16 = self.load_tiff_16bit(path)
        proxy, scale = self.build_proxy_8bit(img16)
        return ProxyImageBundle(
            file_path=path,
            original_16bit=img16,
            proxy_8bit=proxy,
            proxy_scale=scale,
        )

    def detect_initial_crop_by_contour(
        self,
        proxy_8bit: np.ndarray,
        target_aspect: float,
        min_area_ratio: float = 0.05,
    ) -> Optional[CropState]:
        """Find an initial crop using threshold + largest valid contour.

        Detection happens ONLY in proxy coordinate space.
        """
        if target_aspect <= 0:
            raise ValueError("target_aspect must be > 0")

        gray = self._to_gray(proxy_8bit)

        # Otsu threshold assumes bright film region against darker surroundings.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fill gaps / remove speckles.
        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_h, img_w = gray.shape[:2]
        min_area = img_h * img_w * float(min_area_ratio)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            candidates.append((area, rect))

        if not candidates:
            return None

        # Choose largest candidate for baseline implementation.
        _, rect = max(candidates, key=lambda x: x[0])
        (cx, cy), (rw, rh), angle = rect

        if rw <= 1 or rh <= 1:
            return None

        # Normalize to keep width/height aligned with target aspect.
        box_aspect = rw / rh if rh != 0 else 1.0
        if box_aspect >= target_aspect:
            height = rh
            width = rh * target_aspect
        else:
            width = rw
            height = rw / target_aspect

        # Ensure the crop stays inside image boundaries.
        width = min(width, img_w)
        height = min(height, img_h)
        cx = float(np.clip(cx, width / 2.0, img_w - width / 2.0))
        cy = float(np.clip(cy, height / 2.0, img_h - height / 2.0))

        return CropState(
            cx=float(cx),
            cy=float(cy),
            width=float(width),
            height=float(height),
            angle_deg=float(angle),
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            # Assume RGB-like multi-channel; convert to grayscale for thresholding.
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {img.shape}")
