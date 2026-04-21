"""Core image pipeline for 16-bit TIFF film copy cropping.

Step-1 scope:
- Load original 16-bit TIFF safely
- Build an 8-bit proxy image for UI/fast CV
- Run a basic contour-based initial crop detection on proxy space
- Predict follow-up crop states using prior approved geometry
- Export rotated crop rectangles back on the original 16-bit TIFF
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
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


@dataclass
class DetectionDebugBundle:
    """Debug images for understanding contour/edge detection behavior."""

    overlay_rgb: np.ndarray
    threshold_mask: np.ndarray
    edge_mask: np.ndarray
    chosen_state: Optional[CropState]


@dataclass
class FilmBaseDebugBundle:
    """Debug images for film-base-guided segmentation."""

    overlay_rgb: np.ndarray
    sample_preview_rgb: np.ndarray
    distance_heatmap_rgb: np.ndarray
    non_base_mask: np.ndarray
    sample_rect: Optional[tuple[int, int, int, int]]
    chosen_state: Optional[CropState]
    base_rgb_mean: tuple[float, float, float]


@dataclass
class FilmBaseColorModel:
    """A roll-wide film-base color model sampled from one click region."""

    rgb_mean: tuple[float, float, float]
    lab_mean: tuple[float, float, float]
    lab_std: tuple[float, float, float]


class FilmType(str, Enum):
    """Brightness relationship between film base and image area."""

    NEGATIVE = "negative"
    REVERSAL = "reversal"


@dataclass
class ProjectionDetectionResult:
    """Projection-based edge positions with per-edge confidence."""

    left: Optional[int]
    right: Optional[int]
    top: Optional[int]
    bottom: Optional[int]
    left_conf: float
    right_conf: float
    top_conf: float
    bottom_conf: float


class ImageCore:
    """Image processing utilities for the film crop reviewer."""

    def __init__(self, proxy_max_long_edge: int = 2000) -> None:
        if proxy_max_long_edge <= 0:
            raise ValueError("proxy_max_long_edge must be > 0")
        self.proxy_max_long_edge = proxy_max_long_edge
        self.proxy_intensity_gain: Optional[float] = None
        self.proxy_target_peak = 252.0

    def configure_proxy_intensity_gain(self, gain: Optional[float]) -> None:
        """Set a linear 16-bit -> 8-bit gain shared by the current roll."""
        if gain is not None and gain <= 0:
            raise ValueError("proxy intensity gain must be > 0")
        self.proxy_intensity_gain = gain

    def estimate_roll_proxy_intensity_gain(
        self,
        file_paths: list[str | Path],
        target_peak: Optional[float] = None,
    ) -> float:
        """Estimate one safe linear gain for all TIFFs in the current roll."""
        if not file_paths:
            raise ValueError("file_paths must not be empty")

        peak = float(self.proxy_target_peak if target_peak is None else target_peak)
        if not 1.0 <= peak <= 255.0:
            raise ValueError("target_peak must be between 1 and 255")

        roll_max = 0.0
        for file_path in file_paths:
            image = self.load_tiff_16bit(file_path)
            roll_max = max(roll_max, float(np.max(image)))

        if roll_max <= 0:
            return 1.0
        return peak / roll_max

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

    def build_proxy_8bit(
        self,
        img16: np.ndarray,
        intensity_gain: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Create an 8-bit resized proxy image from a 16-bit image.

        Returns:
            proxy_8bit: HxW (grayscale) or HxWxC (color), uint8
            scale: proxy/original scale factor in x,y (uniform)
        """
        if img16.dtype != np.uint16:
            raise ValueError("build_proxy_8bit expects uint16 input")

        gain = intensity_gain if intensity_gain is not None else self.proxy_intensity_gain
        if gain is None:
            max_v = float(np.max(img16))
            gain = self.proxy_target_peak / max(max_v, 1.0)

        proxy_linear = img16.astype(np.float32) * float(gain)
        norm = np.clip(proxy_linear, 0, 255).astype(np.uint8)

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

        angle = self._normalize_rect_angle(rw, rh, angle, target_aspect)

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

    def detect_followup_crop_by_contour(
        self,
        proxy_8bit: np.ndarray,
        previous_state: CropState,
        min_area_ratio: float = 0.05,
        max_center_shift_ratio: float = 0.35,
        max_angle_delta_deg: float = 8.0,
    ) -> Optional[CropState]:
        """Predict the next crop by reusing prior width/height and detecting only pose.

        For the first approved image we allow a full contour-based rectangle.
        For subsequent images, width/height stay fixed and only center position
        plus a small angle drift are updated.
        """
        target_aspect = previous_state.width / max(1.0, previous_state.height)
        detected = self.detect_initial_crop_by_contour(
            proxy_8bit=proxy_8bit,
            target_aspect=target_aspect,
            min_area_ratio=min_area_ratio,
        )
        if detected is None:
            return None

        dx = detected.cx - previous_state.cx
        dy = detected.cy - previous_state.cy
        max_center_shift = max(previous_state.width, previous_state.height) * max_center_shift_ratio
        if math.hypot(dx, dy) > max_center_shift:
            return None

        angle_deg = self._normalize_angle_near(detected.angle_deg, previous_state.angle_deg)
        if abs(angle_deg - previous_state.angle_deg) > max_angle_delta_deg:
            return None

        return CropState(
            cx=float(detected.cx),
            cy=float(detected.cy),
            width=float(previous_state.width),
            height=float(previous_state.height),
            angle_deg=float(angle_deg),
        )

    def build_detection_debug_bundle(
        self,
        proxy_8bit: np.ndarray,
        target_aspect: float,
        previous_state: Optional[CropState] = None,
        min_area_ratio: float = 0.05,
    ) -> DetectionDebugBundle:
        """Create debug images showing what CV detected on the proxy image."""
        if target_aspect <= 0:
            raise ValueError("target_aspect must be > 0")

        gray = self._to_gray(proxy_8bit)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        edge_mask = cv2.Canny(blur, 40, 120)

        contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = gray.shape[:2]
        min_area = img_h * img_w * float(min_area_ratio)

        overlay = self._to_rgb(proxy_8bit)
        cv2.drawContours(overlay, contours, -1, (80, 170, 255), 1)

        candidates: list[tuple[float, tuple]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), angle = rect
            if rw <= 1 or rh <= 1:
                continue
            box = cv2.boxPoints(rect).astype(np.int32)
            candidates.append((float(area), rect))
            cv2.polylines(overlay, [box], True, (0, 255, 255), 2)

        chosen_state: Optional[CropState] = None
        if candidates:
            _, rect = max(candidates, key=lambda item: item[0])
            chosen_state = self._rect_to_crop_state(
                rect=rect,
                target_aspect=target_aspect,
                image_width=img_w,
                image_height=img_h,
            )
            self._draw_crop_state(overlay, chosen_state, (0, 255, 0), 3)

        if previous_state is not None:
            self._draw_crop_state(overlay, previous_state, (255, 120, 0), 2)

        return DetectionDebugBundle(
            overlay_rgb=overlay,
            threshold_mask=threshold_mask,
            edge_mask=edge_mask,
            chosen_state=chosen_state,
        )

    def build_film_base_debug_bundle(
        self,
        proxy_8bit: np.ndarray,
        target_aspect: float,
        sample_rect: Optional[tuple[int, int, int, int]] = None,
        base_color_model: Optional[FilmBaseColorModel] = None,
        film_type: FilmType = FilmType.NEGATIVE,
        previous_state: Optional[CropState] = None,
        min_area_ratio: float = 0.05,
    ) -> FilmBaseDebugBundle:
        """Visualize a film-base-guided mask based on sampled base color."""
        if target_aspect <= 0:
            raise ValueError("target_aspect must be > 0")

        rgb = self._to_rgb(proxy_8bit)
        gray = self._to_gray(proxy_8bit)
        img_h, img_w = gray.shape[:2]

        if sample_rect is None and base_color_model is None:
            sample_rect = self._suggest_film_base_sample_rect(
                gray,
                min_area_ratio=min_area_ratio,
                film_type=film_type,
            )
        sample_preview = rgb.copy()
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        sanitized_rect: Optional[tuple[int, int, int, int]] = None
        if base_color_model is None:
            if sample_rect is None:
                raise ValueError("sample_rect is required when base_color_model is not provided")
            x, y, w, h = self._sanitize_rect(sample_rect, img_w, img_h)
            sanitized_rect = (x, y, w, h)
            cv2.rectangle(sample_preview, (x, y), (x + w - 1, y + h - 1), (255, 0, 255), 3)
            base_color_model = self.sample_film_base_color_model(proxy_8bit, sanitized_rect)

        base_lab_mean = np.asarray(base_color_model.lab_mean, dtype=np.float32)
        base_lab_std = np.asarray(base_color_model.lab_std, dtype=np.float32)
        base_rgb_mean = tuple(float(v) for v in base_color_model.rgb_mean)

        if film_type == FilmType.REVERSAL:
            light_separation = np.clip(lab[..., 0] - base_lab_mean[0], 0.0, None)
        else:
            light_separation = np.clip(base_lab_mean[0] - lab[..., 0], 0.0, None)
        color_distance = np.linalg.norm(lab[..., 1:3] - base_lab_mean[1:3], axis=2)
        combined_score = light_separation * 1.3 + color_distance * 0.9

        light_thresh = max(6.0, float(base_lab_std[0]) * 2.2)
        color_thresh = max(8.0, float(np.mean(base_lab_std[1:])) * 2.6)
        combined_thresh = light_thresh * 1.2 + color_thresh * 0.8

        non_base_mask = np.where(
            ((light_separation >= light_thresh) & (combined_score >= combined_thresh))
            | (color_distance >= color_thresh * 1.35),
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        non_base_mask = cv2.morphologyEx(non_base_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        non_base_mask = cv2.morphologyEx(non_base_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        dist_norm = cv2.normalize(combined_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        distance_heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        distance_heatmap = cv2.cvtColor(distance_heatmap, cv2.COLOR_BGR2RGB)

        overlay = rgb.copy()
        if sanitized_rect is not None:
            x, y, w, h = sanitized_rect
            cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (255, 0, 255), 2)

        chosen_state, projection = self._detect_crop_by_projection(
            non_base_mask=non_base_mask,
            support_score=self._normalize_projection_score(light_separation, scale=max(light_thresh * 2.5, 1.0)),
            target_aspect=target_aspect,
            image_width=img_w,
            image_height=img_h,
            previous_state=previous_state,
        )
        if projection is not None:
            self._draw_projection_guides(overlay, projection, img_w, img_h)
        if chosen_state is not None:
            self._draw_crop_state(overlay, chosen_state, (0, 255, 0), 3)

        return FilmBaseDebugBundle(
            overlay_rgb=overlay,
            sample_preview_rgb=sample_preview,
            distance_heatmap_rgb=distance_heatmap,
            non_base_mask=non_base_mask,
            sample_rect=sanitized_rect,
            chosen_state=chosen_state,
            base_rgb_mean=base_rgb_mean,
        )

    def detect_crop_by_film_base(
        self,
        proxy_8bit: np.ndarray,
        target_aspect: float,
        sample_rect: Optional[tuple[int, int, int, int]] = None,
        base_color_model: Optional[FilmBaseColorModel] = None,
        film_type: FilmType = FilmType.NEGATIVE,
        previous_state: Optional[CropState] = None,
        min_area_ratio: float = 0.05,
    ) -> Optional[CropState]:
        """Detect a crop rectangle using sampled film-base color as the reference."""
        debug_bundle = self.build_film_base_debug_bundle(
            proxy_8bit=proxy_8bit,
            target_aspect=target_aspect,
            sample_rect=sample_rect,
            base_color_model=base_color_model,
            film_type=film_type,
            previous_state=previous_state,
            min_area_ratio=min_area_ratio,
        )
        return debug_bundle.chosen_state

    def sample_film_base_color_model(
        self,
        proxy_8bit: np.ndarray,
        sample_rect: tuple[int, int, int, int],
    ) -> FilmBaseColorModel:
        """Estimate a stable film-base color model from a small sampled patch."""
        rgb = self._to_rgb(proxy_8bit)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        img_h, img_w = rgb.shape[:2]
        x, y, w, h = self._sanitize_rect(sample_rect, img_w, img_h)

        sample_rgb = rgb[y : y + h, x : x + w].reshape(-1, 3).astype(np.float32)
        sample_lab = lab[y : y + h, x : x + w].reshape(-1, 3)

        rgb_mean = tuple(float(v) for v in sample_rgb.mean(axis=0))
        lab_mean = tuple(float(v) for v in sample_lab.mean(axis=0))
        lab_std = tuple(float(v) for v in sample_lab.std(axis=0))
        return FilmBaseColorModel(
            rgb_mean=rgb_mean,
            lab_mean=lab_mean,
            lab_std=lab_std,
        )

    def export_cropped_image(
        self,
        original_img_16bit: np.ndarray,
        crop_state: CropState,
        scale_factor: float,
        output_path: str | Path,
        inset_ratio_per_side: float = 0.0,
    ) -> None:
        """Warp a rotated crop from proxy coordinates back to the 16-bit original."""
        if original_img_16bit.dtype != np.uint16:
            raise ValueError("export_cropped_image expects uint16 input")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be > 0")
        if not 0.0 <= inset_ratio_per_side <= 0.05:
            raise ValueError("inset_ratio_per_side must be between 0.0 and 0.05")

        cx = float(crop_state.cx) * scale_factor
        cy = float(crop_state.cy) * scale_factor
        width = max(1.0, float(crop_state.width) * scale_factor)
        height = max(1.0, float(crop_state.height) * scale_factor)

        # Apply a uniform inset on all four sides only at export time.
        width *= max(0.01, 1.0 - 2.0 * inset_ratio_per_side)
        height *= max(0.01, 1.0 - 2.0 * inset_ratio_per_side)

        dest_w = max(1, int(round(width)))
        dest_h = max(1, int(round(height)))

        src_box = cv2.boxPoints(((cx, cy), (width, height), float(crop_state.angle_deg)))
        src_pts = self._order_points(src_box.astype(np.float32))
        dst_pts = np.array(
            [
                [0.0, 0.0],
                [dest_w - 1.0, 0.0],
                [dest_w - 1.0, dest_h - 1.0],
                [0.0, dest_h - 1.0],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(
            original_img_16bit,
            transform,
            (dest_w, dest_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if cropped.dtype != np.uint16:
            raise ValueError(f"warpPerspective changed dtype unexpectedly: {cropped.dtype}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output), cropped)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            # TIFF data loaded via tifffile is typically RGB-like channel order.
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Unsupported image shape: {img.shape}")

    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 3:
            return img.copy()
        raise ValueError(f"Unsupported image shape: {img.shape}")

    def _rect_to_crop_state(
        self,
        rect: tuple,
        target_aspect: float,
        image_width: int,
        image_height: int,
    ) -> CropState:
        (cx, cy), (rw, rh), angle = rect
        angle = self._normalize_rect_angle(rw, rh, angle, target_aspect)

        box_aspect = rw / rh if rh != 0 else 1.0
        if box_aspect >= target_aspect:
            height = rh
            width = rh * target_aspect
        else:
            width = rw
            height = rw / target_aspect

        width = min(width, image_width)
        height = min(height, image_height)
        cx = float(np.clip(cx, width / 2.0, image_width - width / 2.0))
        cy = float(np.clip(cy, height / 2.0, image_height - height / 2.0))
        return CropState(
            cx=float(cx),
            cy=float(cy),
            width=float(width),
            height=float(height),
            angle_deg=float(angle),
        )

    def _detect_crop_by_projection(
        self,
        non_base_mask: np.ndarray,
        support_score: np.ndarray,
        target_aspect: float,
        image_width: int,
        image_height: int,
        previous_state: Optional[CropState] = None,
        min_edge_confidence: float = 0.30,
        max_size_deviation_ratio: float = 0.12,
        max_edge_shift_ratio: float = 0.10,
        min_edge_shift_px: float = 36.0,
    ) -> tuple[Optional[CropState], Optional[ProjectionDetectionResult]]:
        row_fraction = non_base_mask.mean(axis=1).astype(np.float32) / 255.0
        col_fraction = non_base_mask.mean(axis=0).astype(np.float32) / 255.0
        row_support = support_score.mean(axis=1).astype(np.float32)
        col_support = support_score.mean(axis=0).astype(np.float32)

        row_fraction = self._smooth_projection(row_fraction)
        col_fraction = self._smooth_projection(col_fraction)
        row_support = self._smooth_projection(row_support)
        col_support = self._smooth_projection(col_support)

        left = right = top = bottom = None
        left_conf = right_conf = top_conf = bottom_conf = 0.0

        if previous_state is None:
            expected_center_x = image_width / 2.0
            expected_center_y = image_height / 2.0

            left_right = self._find_projection_span(
                col_fraction,
                threshold=0.58,
                expected_center=expected_center_x,
                support_values=col_support,
            )
            top_bottom = self._find_projection_span(
                row_fraction,
                threshold=0.58,
                expected_center=expected_center_y,
                support_values=row_support,
            )

            if left_right is not None:
                left, right = left_right
                left_conf = self._compute_edge_confidence(
                    col_fraction,
                    left,
                    side="start",
                    support_values=col_support,
                )
                right_conf = self._compute_edge_confidence(
                    col_fraction,
                    right,
                    side="end",
                    support_values=col_support,
                )
            if top_bottom is not None:
                top, bottom = top_bottom
                top_conf = self._compute_edge_confidence(
                    row_fraction,
                    top,
                    side="start",
                    support_values=row_support,
                )
                bottom_conf = self._compute_edge_confidence(
                    row_fraction,
                    bottom,
                    side="end",
                    support_values=row_support,
                )
        else:
            expected_left = int(round(previous_state.cx - previous_state.width / 2.0))
            expected_right = int(round(previous_state.cx + previous_state.width / 2.0))
            expected_top = int(round(previous_state.cy - previous_state.height / 2.0))
            expected_bottom = int(round(previous_state.cy + previous_state.height / 2.0))
            max_x_shift = max(float(min_edge_shift_px), float(previous_state.width) * max_edge_shift_ratio)
            max_y_shift = max(float(min_edge_shift_px), float(previous_state.height) * max_edge_shift_ratio)

            left, left_conf = self._find_edge_candidate(
                col_fraction,
                threshold=0.58,
                expected_index=expected_left,
                side="start",
                support_values=col_support,
                max_offset=max_x_shift,
            )
            right, right_conf = self._find_edge_candidate(
                col_fraction,
                threshold=0.58,
                expected_index=expected_right,
                side="end",
                support_values=col_support,
                max_offset=max_x_shift,
            )
            top, top_conf = self._find_edge_candidate(
                row_fraction,
                threshold=0.58,
                expected_index=expected_top,
                side="start",
                support_values=row_support,
                max_offset=max_y_shift,
            )
            bottom, bottom_conf = self._find_edge_candidate(
                row_fraction,
                threshold=0.58,
                expected_index=expected_bottom,
                side="end",
                support_values=row_support,
                max_offset=max_y_shift,
            )

        projection = ProjectionDetectionResult(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            left_conf=float(left_conf),
            right_conf=float(right_conf),
            top_conf=float(top_conf),
            bottom_conf=float(bottom_conf),
        )

        if previous_state is None:
            use_left = left is not None and left_conf >= min_edge_confidence
            use_right = right is not None and right_conf >= min_edge_confidence
            use_top = top is not None and top_conf >= min_edge_confidence
            use_bottom = bottom is not None and bottom_conf >= min_edge_confidence

            initial_state = self._build_initial_crop_from_edges(
                image_width=image_width,
                image_height=image_height,
                target_aspect=target_aspect,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                use_left=use_left,
                use_right=use_right,
                use_top=use_top,
                use_bottom=use_bottom,
            )
            return initial_state, projection

        width = float(previous_state.width)
        height = float(previous_state.height)
        use_left = left is not None and left_conf >= min_edge_confidence
        use_right = right is not None and right_conf >= min_edge_confidence
        use_top = top is not None and top_conf >= min_edge_confidence
        use_bottom = bottom is not None and bottom_conf >= min_edge_confidence

        if use_left and use_right:
            measured_width = float(right - left + 1)
            width_deviation = abs(measured_width - width) / max(width, 1.0)
            if width_deviation > max_size_deviation_ratio:
                if left_conf >= right_conf:
                    use_right = False
                else:
                    use_left = False

        if use_top and use_bottom:
            measured_height = float(bottom - top + 1)
            height_deviation = abs(measured_height - height) / max(height, 1.0)
            if height_deviation > max_size_deviation_ratio:
                if top_conf >= bottom_conf:
                    use_bottom = False
                else:
                    use_top = False

        accepted_edges = sum((use_left, use_right, use_top, use_bottom))
        if accepted_edges >= 3:
            inferred_state = self._build_initial_crop_from_edges(
                image_width=image_width,
                image_height=image_height,
                target_aspect=target_aspect,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                use_left=use_left,
                use_right=use_right,
                use_top=use_top,
                use_bottom=use_bottom,
            )
            if inferred_state is not None:
                cx = float(np.clip(inferred_state.cx, width / 2.0, image_width - width / 2.0))
                cy = float(np.clip(inferred_state.cy, height / 2.0, image_height - height / 2.0))
                return CropState(cx=cx, cy=cy, width=width, height=height, angle_deg=0.0), projection

        x_candidates: list[float] = []
        y_candidates: list[float] = []
        accepted_edges = 0

        if use_left:
            x_candidates.append(left + width / 2.0)
            accepted_edges += 1
        if use_right:
            x_candidates.append(right - width / 2.0)
            accepted_edges += 1
        if use_left and use_right:
            x_candidates.append((left + right) / 2.0)

        if use_top:
            y_candidates.append(top + height / 2.0)
            accepted_edges += 1
        if use_bottom:
            y_candidates.append(bottom - height / 2.0)
            accepted_edges += 1
        if use_top and use_bottom:
            y_candidates.append((top + bottom) / 2.0)

        if accepted_edges < 2:
            return None, projection

        cx = float(np.mean(x_candidates)) if x_candidates else float(previous_state.cx)
        cy = float(np.mean(y_candidates)) if y_candidates else float(previous_state.cy)
        cx = float(np.clip(cx, width / 2.0, image_width - width / 2.0))
        cy = float(np.clip(cy, height / 2.0, image_height - height / 2.0))
        return CropState(cx=cx, cy=cy, width=width, height=height, angle_deg=0.0), projection

    @staticmethod
    def _smooth_projection(values: np.ndarray, kernel_size: int = 31) -> np.ndarray:
        kernel_size = max(3, kernel_size | 1)
        values_2d = values.reshape(-1, 1)
        smoothed = cv2.GaussianBlur(values_2d, (1, kernel_size), 0)
        return smoothed.reshape(-1)

    @staticmethod
    def _find_projection_span(
        values: np.ndarray,
        threshold: float,
        expected_center: Optional[float] = None,
        support_values: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int]]:
        runs = ImageCore._collect_projection_runs(values, threshold)
        if not runs:
            return None

        if expected_center is None:
            expected_center = len(values) / 2.0

        best_run = None
        best_score = None
        axis_len = max(float(len(values)), 1.0)
        for run_start, run_end in runs:
            run_center = (run_start + run_end) / 2.0
            run_length = run_end - run_start + 1
            run_support = 0.0
            if support_values is not None:
                run_support = float(np.mean(support_values[run_start : run_end + 1]))
            score = (
                (run_length / axis_len) * 1.35
                + run_support * 0.95
                - abs(run_center - expected_center) / axis_len
            )
            if best_score is None or score > best_score:
                best_score = score
                best_run = (run_start, run_end)

        return best_run

    def _find_edge_candidate(
        self,
        values: np.ndarray,
        threshold: float,
        expected_index: int,
        side: str,
        support_values: Optional[np.ndarray],
        max_offset: float,
    ) -> tuple[Optional[int], float]:
        runs = self._collect_projection_runs(values, threshold)
        if not runs:
            return None, 0.0

        best_index = None
        best_confidence = 0.0
        best_score = None
        max_offset = max(float(max_offset), 1.0)

        for run_start, run_end in runs:
            candidate = run_start if side == "start" else run_end
            distance = abs(float(candidate) - float(expected_index))
            if distance > max_offset:
                continue

            base_confidence = self._compute_edge_confidence(
                values,
                candidate,
                side=side,
                support_values=support_values,
            )
            proximity = 1.0 - distance / max_offset
            effective_confidence = base_confidence * (0.40 + 0.60 * proximity)
            score = effective_confidence + 0.08 * proximity

            if best_score is None or score > best_score:
                best_score = score
                best_index = int(candidate)
                best_confidence = float(effective_confidence)

        return best_index, best_confidence

    @staticmethod
    def _build_initial_crop_from_edges(
        image_width: int,
        image_height: int,
        target_aspect: float,
        left: Optional[int],
        right: Optional[int],
        top: Optional[int],
        bottom: Optional[int],
        use_left: bool,
        use_right: bool,
        use_top: bool,
        use_bottom: bool,
        min_size: float = 20.0,
    ) -> Optional[CropState]:
        accepted_edges = sum((use_left, use_right, use_top, use_bottom))
        if accepted_edges < 3:
            return None

        width: Optional[float] = None
        height: Optional[float] = None
        cx_candidates: list[float] = []
        cy_candidates: list[float] = []

        if use_left and use_right and left is not None and right is not None:
            width = float(right - left + 1)
            if width < min_size:
                return None
            height = width / target_aspect
            if height < min_size:
                return None
            cx_candidates.append((left + right) / 2.0)
            if use_top and top is not None:
                cy_candidates.append(top + height / 2.0)
            if use_bottom and bottom is not None:
                cy_candidates.append(bottom - height / 2.0)

        if use_top and use_bottom and top is not None and bottom is not None:
            candidate_height = float(bottom - top + 1)
            if candidate_height < min_size:
                return None
            candidate_width = candidate_height * target_aspect
            if candidate_width < min_size:
                return None

            if height is None:
                height = candidate_height
                width = candidate_width
            else:
                # When all four edges are present, keep the largest aspect-locked crop that fits.
                assert width is not None
                width = min(width, candidate_width)
                height = width / target_aspect

            cy_candidates.append((top + bottom) / 2.0)
            if use_left and left is not None and width is not None:
                cx_candidates.append(left + width / 2.0)
            if use_right and right is not None and width is not None:
                cx_candidates.append(right - width / 2.0)

        if width is None or height is None or width < min_size or height < min_size:
            return None

        if not cx_candidates or not cy_candidates:
            return None

        width = min(width, float(image_width))
        height = min(height, float(image_height))
        cx = float(np.clip(np.mean(cx_candidates), width / 2.0, image_width - width / 2.0))
        cy = float(np.clip(np.mean(cy_candidates), height / 2.0, image_height - height / 2.0))
        return CropState(cx=cx, cy=cy, width=width, height=height, angle_deg=0.0)

    @staticmethod
    def _collect_projection_runs(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        mask = values >= threshold
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        runs: list[tuple[int, int]] = []
        start = int(indices[0])
        end = start
        for idx in indices[1:]:
            idx = int(idx)
            if idx == end + 1:
                end = idx
                continue
            runs.append((start, end))
            start = idx
            end = idx
        runs.append((start, end))
        return runs

    @staticmethod
    def _compute_edge_confidence(
        values: np.ndarray,
        index: int,
        side: str,
        support_values: Optional[np.ndarray] = None,
        window: int = 24,
    ) -> float:
        if side == "start":
            outside = values[max(0, index - window) : index]
            inside = values[index : min(len(values), index + window)]
            support_outside = (
                support_values[max(0, index - window) : index] if support_values is not None else np.empty(0, dtype=np.float32)
            )
            support_inside = (
                support_values[index : min(len(values), index + window)] if support_values is not None else np.empty(0, dtype=np.float32)
            )
        else:
            outside = values[index + 1 : min(len(values), index + 1 + window)]
            inside = values[max(0, index - window + 1) : index + 1]
            support_outside = (
                support_values[index + 1 : min(len(values), index + 1 + window)]
                if support_values is not None
                else np.empty(0, dtype=np.float32)
            )
            support_inside = (
                support_values[max(0, index - window + 1) : index + 1]
                if support_values is not None
                else np.empty(0, dtype=np.float32)
            )

        inside_mean = float(np.mean(inside)) if inside.size > 0 else float(values[index])
        outside_mean = float(np.mean(outside)) if outside.size > 0 else inside_mean
        contrast = max(0.0, inside_mean - outside_mean)

        support_inside_mean = float(np.mean(support_inside)) if support_inside.size > 0 else 0.0
        support_outside_mean = float(np.mean(support_outside)) if support_outside.size > 0 else support_inside_mean
        support_contrast = max(0.0, support_inside_mean - support_outside_mean)

        outside_coverage = min(1.0, outside.size / max(1.0, window))
        support_outside_coverage = min(1.0, support_outside.size / max(1.0, window))
        context_coverage = min(
            outside_coverage,
            support_outside_coverage if support_values is not None else outside_coverage,
        )

        confidence = (
            0.34 * np.clip(inside_mean, 0.0, 1.0)
            + 0.24 * np.clip(contrast / 0.55, 0.0, 1.0)
            + 0.22 * np.clip(support_inside_mean, 0.0, 1.0)
            + 0.20 * np.clip(support_contrast / 0.22, 0.0, 1.0)
        )
        confidence *= 0.35 + 0.65 * context_coverage
        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _normalize_projection_score(values: np.ndarray, scale: float) -> np.ndarray:
        if scale <= 0:
            scale = 1.0
        return np.clip(values.astype(np.float32) / float(scale), 0.0, 1.0)

    def _suggest_film_base_sample_rect(
        self,
        gray: np.ndarray,
        min_area_ratio: float,
        film_type: FilmType = FilmType.NEGATIVE,
    ) -> tuple[int, int, int, int]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = gray.shape[:2]
        min_area = img_h * img_w * float(min_area_ratio)

        if not contours:
            return self._default_sample_rect(img_w, img_h)

        valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if not valid:
            return self._default_sample_rect(img_w, img_h)

        cnt = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        band = max(16, int(round(min(w, h) * 0.035)))

        candidate_rects = [
            (x + band, y + band, max(20, w - 2 * band), band),
            (x + band, y + h - 2 * band, max(20, w - 2 * band), band),
            (x + band, y + band, band, max(20, h - 2 * band)),
            (x + w - 2 * band, y + band, band, max(20, h - 2 * band)),
        ]

        best_rect = None
        best_score = None
        for rect in candidate_rects:
            rx, ry, rw, rh = self._sanitize_rect(rect, img_w, img_h)
            patch = gray[ry : ry + rh, rx : rx + rw]
            if patch.size == 0:
                continue
            mean_value = float(np.mean(patch))
            brightness_score = mean_value if film_type == FilmType.NEGATIVE else -mean_value
            score = (brightness_score, -float(np.std(patch)))
            if best_score is None or score > best_score:
                best_score = score
                best_rect = (rx, ry, rw, rh)

        return best_rect or self._default_sample_rect(img_w, img_h)

    @staticmethod
    def _default_sample_rect(image_width: int, image_height: int) -> tuple[int, int, int, int]:
        w = max(30, int(round(image_width * 0.12)))
        h = max(30, int(round(image_height * 0.06)))
        x = max(0, (image_width - w) // 2)
        y = max(0, int(round(image_height * 0.03)))
        return x, y, w, h

    @staticmethod
    def _sanitize_rect(rect: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x, y, w, h = rect
        x = max(0, min(int(x), image_width - 1))
        y = max(0, min(int(y), image_height - 1))
        w = max(1, min(int(w), image_width - x))
        h = max(1, min(int(h), image_height - y))
        return x, y, w, h

    @staticmethod
    def _draw_crop_state(image_rgb: np.ndarray, state: CropState, color: tuple[int, int, int], thickness: int) -> None:
        rect = ((float(state.cx), float(state.cy)), (float(state.width), float(state.height)), float(state.angle_deg))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(image_rgb, [box], True, color, thickness)

    @staticmethod
    def _draw_projection_guides(
        image_rgb: np.ndarray,
        projection: ProjectionDetectionResult,
        image_width: int,
        image_height: int,
    ) -> None:
        guide_specs = [
            ("L", projection.left, projection.left_conf, True, (0, 190, 255)),
            ("R", projection.right, projection.right_conf, True, (0, 190, 255)),
            ("T", projection.top, projection.top_conf, False, (255, 120, 0)),
            ("B", projection.bottom, projection.bottom_conf, False, (255, 120, 0)),
        ]

        for label, position, confidence, vertical, color in guide_specs:
            if position is None:
                continue
            if vertical:
                cv2.line(image_rgb, (int(position), 0), (int(position), image_height - 1), color, 1)
                text_origin = (min(image_width - 80, int(position) + 6), 24)
            else:
                cv2.line(image_rgb, (0, int(position)), (image_width - 1, int(position)), color, 1)
                text_origin = (8, min(image_height - 8, int(position) - 6 if position > 24 else int(position) + 18))
            cv2.putText(
                image_rgb,
                f"{label}:{confidence:.2f}",
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )


    @staticmethod
    def _normalize_rect_angle(rect_w: float, rect_h: float, angle_deg: float, target_aspect: float) -> float:
        width_prefers_landscape = target_aspect >= 1.0
        rect_is_landscape = rect_w >= rect_h
        if width_prefers_landscape != rect_is_landscape:
            angle_deg += 90.0

        while angle_deg <= -45.0:
            angle_deg += 90.0
        while angle_deg > 45.0:
            angle_deg -= 90.0
        return float(angle_deg)

    @staticmethod
    def _normalize_angle_near(candidate_deg: float, reference_deg: float) -> float:
        best = candidate_deg
        best_delta = abs(candidate_deg - reference_deg)
        for offset in (-180.0, -90.0, 0.0, 90.0, 180.0):
            value = candidate_deg + offset
            delta = abs(value - reference_deg)
            if delta < best_delta:
                best = value
                best_delta = delta
        return float(best)

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered
