"""ROI localization extracted from the original VLMtest.py UI.

This keeps perception (YOLO/contour crop) separate from the decision pipeline.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # allows contour-only operation
    YOLO = None

_YOLO_MODEL = None


def _get_yolo_model(weights: str = str(Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt")):
    global _YOLO_MODEL
    if YOLO is None:
        return None
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(weights)
    return _YOLO_MODEL


def get_foreground_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = img.shape[:2]
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    if valid:
        all_points = np.vstack(valid)
        x, y, bw, bh = cv2.boundingRect(all_points)
        return x, y, x + bw, y + bh, 0.95
    return int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95), 0.50


def localize_and_crop_object(
    image_path: str,
    crop_path: str = "temp_roi_crop.jpg",
    use_yolo: bool = True,
    yolo_weights: str = str(Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt"),
) -> tuple[str, dict]:
    """Localize the visible workpiece and save a cropped ROI."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    x1 = y1 = x2 = y2 = None
    conf = 0.0
    status = "Contour fallback"

    model = _get_yolo_model(yolo_weights) if use_yolo else None
    if model is not None:
        results = model(img, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            all_xyxy = boxes.xyxy.cpu().numpy()
            x1 = int(np.min(all_xyxy[:, 0]))
            y1 = int(np.min(all_xyxy[:, 1]))
            x2 = int(np.max(all_xyxy[:, 2]))
            y2 = int(np.max(all_xyxy[:, 3]))
            conf = float(boxes.conf.max().cpu().numpy())
            status = f"YOLO detected {len(boxes)} object(s)"

    if x1 is None:
        x1, y1, x2, y2, conf = get_foreground_bounding_box(img)

    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = img[y1:y2, x1:x2]
    out = Path(crop_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), cropped):
        raise OSError(f"Could not write ROI crop: {out}")

    return str(out), {
        "bbox": [x1, y1, x2, y2],
        "confidence": round(conf, 3),
        "status": status,
    }
