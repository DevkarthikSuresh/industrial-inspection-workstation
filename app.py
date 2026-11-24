import os
from pathlib import Path
import cv2
import base64
import asyncio
import json
import numpy as np

from nicegui import ui

from industrial_inspection.decision_engine import (
    run_inspection_pipeline,
    continue_with_new_evidence,
    request_operator_override,
)
from industrial_inspection.learn_engine import LearnEngine
from industrial_inspection.vlm_client import MODEL_NAME
from industrial_inspection import industrial_memory as memory

# ============================================================
# MODEL / CONFIGURATION
# ============================================================

RUNTIME_DIR = str(Path(__file__).resolve().parent / "runtime")
os.makedirs(RUNTIME_DIR, exist_ok=True)

# Browser uploads are normalized into this JPEG working frame.
IMAGE_PATH = os.path.join(RUNTIME_DIR, "current_input.jpg")
CROP_PATH = os.path.join(RUNTIME_DIR, "temp_roi_crop.jpg")
DEFAULT_PART_NAME = "bolt"

# ============================================================
# LOAD YOLO ONCE
# ============================================================

try:
    from ultralytics import YOLO

    HAS_YOLO = True
    yolo_model = YOLO(str(Path(__file__).resolve().parent / "models" / "yolov8n.pt"))

except ImportError:
    HAS_YOLO = False
    yolo_model = None

print("Initializing YOLO + VLM Inspection Pipeline...")

if not os.path.exists(IMAGE_PATH):
    print("No input image loaded yet. Upload one from the browser UI.")


# ============================================================
# CORE UTILITIES
# ============================================================

def encode_image_to_base64(img_bgr):
    """Encode OpenCV BGR image as a JPEG data URI."""

    rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    _, buffer = cv2.imencode(
        ".jpg",
        rgb
    )

    b64_str = base64.b64encode(
        buffer
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_str}"


# ============================================================
# CONTOUR FALLBACK
# ============================================================

def get_foreground_bounding_box(img):
    """
    Finds and merges all disconnected fragments of the
    workpiece into one enclosing bounding box.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = img.shape[:2]

    valid_contours = [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) > 200
    ]

    if valid_contours:
        all_points = np.vstack(
            valid_contours
        )

        x, y, bw, bh = cv2.boundingRect(
            all_points
        )

        return (
            x,
            y,
            x + bw,
            y + bh,
            0.95
        )

    return (
        int(w * 0.05),
        int(h * 0.05),
        int(w * 0.95),
        int(h * 0.95),
        0.50
    )


# ============================================================
# YOLO + ROI LOCALIZATION
# ============================================================

def localize_and_crop_object(image_path):
    """
    Localizes the workpiece using YOLO or contour analysis,
    then creates a tight ROI crop.
    """

    img = cv2.imread(image_path)

    if img is None:
        return (
            image_path,
            {
                "bbox": None,
                "confidence": 0.0,
                "status": "Failed to read image"
            }
        )

    h, w = img.shape[:2]

    status = "Contour Fusion"
    conf = 0.95

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    if HAS_YOLO and yolo_model is not None:

        results = yolo_model(
            img,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) > 0:

            all_xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            x1 = int(
                np.min(
                    all_xyxy[:, 0]
                )
            )

            y1 = int(
                np.min(
                    all_xyxy[:, 1]
                )
            )

            x2 = int(
                np.max(
                    all_xyxy[:, 2]
                )
            )

            y2 = int(
                np.max(
                    all_xyxy[:, 3]
                )
            )

            conf = float(
                boxes.conf.max()
                .cpu()
                .numpy()
            )

            status = (
                f"YOLO detected "
                f"{len(boxes)} object(s)"
            )

        else:

            (
                x1,
                y1,
                x2,
                y2,
                conf
            ) = get_foreground_bounding_box(
                img
            )

            status = (
                "Contour fallback"
            )

    # --------------------------------------------------------
    # CONTOUR FALLBACK
    # --------------------------------------------------------

    else:

        (
            x1,
            y1,
            x2,
            y2,
            conf
        ) = get_foreground_bounding_box(
            img
        )

        status = (
            "Contour Fusion"
        )

    # --------------------------------------------------------
    # PADDING
    # --------------------------------------------------------

    pad_x = int(
        (x2 - x1) * 0.05
    )

    pad_y = int(
        (y2 - y1) * 0.05
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        w,
        x2 + pad_x
    )

    y2 = min(
        h,
        y2 + pad_y
    )

    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    cropped = img[
              y1:y2,
              x1:x2
              ]

    cv2.imwrite(
        CROP_PATH,
        cropped
    )

    metadata = {
        "bbox": [
            x1,
            y1,
            x2,
            y2
        ],
        "confidence": round(
            conf,
            2
        ),
        "status": status
    }

    return (
        CROP_PATH,
        metadata
    )


# ============================================================
# PAGE / CLIENT
# ============================================================

@ui.page("/")
def inspection_page():
    # Per-client orchestration state for the modular backend.
    session_state = {
        "part_name": DEFAULT_PART_NAME,
        "awaiting_inspection_id": None,
        "last_inspection_id": None,
    }

    async def handle_image_upload(e):
        """Save a browser upload as the current frame and localize it immediately."""
        temp_upload_path = None
        try:
            suffix = os.path.splitext(e.file.name)[1].lower() or ".img"
            temp_upload_path = os.path.join(RUNTIME_DIR, f"_incoming{suffix}")
            await e.file.save(temp_upload_path)

            # Decode whatever image format the operator selected, then normalize
            # it to JPEG so the rest of the pipeline always sees one stable path.
            raw_mat = cv2.imread(temp_upload_path)
            if raw_mat is None:
                raise ValueError("OpenCV could not decode the uploaded image.")
            if not cv2.imwrite(IMAGE_PATH, raw_mat):
                raise OSError(f"Could not write working image to {IMAGE_PATH}")

            raw_image_display.set_source(encode_image_to_base64(raw_mat))
            raw_image_display.set_visibility(True)
            no_sensor_label.set_visibility(False)

            crop_path, meta = localize_and_crop_object(IMAGE_PATH)
            cropped_mat = cv2.imread(crop_path)
            if cropped_mat is not None:
                encoded = encode_image_to_base64(cropped_mat)
                roi_image_display.set_source(encoded)
                inspection_image.set_source(encoded)

            yolo_status_label.set_text(meta["status"])
            yolo_bbox_label.set_text(str(meta["bbox"]))
            yolo_conf_label.set_text(f'{meta["confidence"]:.2f}')
            system_detection_label.set_text(meta["status"])
            bbox_display_label.set_text(str(meta["bbox"]))
            conf_display_label.set_text(f'{int(meta["confidence"] * 100)}%')
            confidence_fill.style(f"width: {meta['confidence'] * 100:.1f}%;")

            current_state_label.set_text("IMAGE READY")
            current_state_description.set_text(
                "Image uploaded and ROI localized. Submit an inspection query."
            )

            ui.notify(f"Loaded: {e.file.name}", type="positive")
            image_uploader.reset()

        except Exception as exc:
            current_state_label.set_text("UPLOAD ERROR")
            current_state_description.set_text(str(exc))
            ui.notify(f"Upload failed: {exc}", type="negative")
        finally:
            if temp_upload_path and os.path.exists(temp_upload_path):
                try:
                    os.remove(temp_upload_path)
                except OSError:
                    pass

    # ========================================================
    # CSS
    # ========================================================

    ui.add_head_html(
        """
        <style>

        :root {
            --bg: #f3f5f7;
            --panel: #ffffff;
            --panel-soft: #f8f9fa;

            --border: #d9dee4;
            --border-strong: #c7cdd4;

            --text: #1f2933;
            --text-secondary: #59636e;
            --text-muted: #7a838d;

            --blue: #0068b5;
            --blue-soft: #eaf3fa;
            --blue-border: #bdd6e8;

            --green: #188038;
            --red: #bd2c2c;

            --shadow:
                0 1px 2px rgba(16, 24, 40, 0.04),
                0 4px 14px rgba(16, 24, 40, 0.04);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0 !important;
            padding: 0 !important;

            background: var(--bg) !important;
            color: var(--text) !important;

            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
        }

        .mono {
            font-family:
                "IBM Plex Mono",
                "SFMono-Regular",
                Consolas,
                "Liberation Mono",
                monospace !important;
        }

        /* ====================================================
           APP
        ==================================================== */

        .app-shell {
            width: 100%;
            height: 100vh;

            overflow: hidden;

            background: var(--bg);
        }

        /* ====================================================
           TOP BAR
        ==================================================== */

        .topbar {
            height: 58px;
            min-height: 58px;

            background: #ffffff;

            border-bottom:
                1px solid var(--border);

            padding: 0 22px;
        }

        .brand-mark {
            width: 30px;
            height: 30px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: #f7f8f9;

            border:
                1px solid #bfc7cf;
        }

        .top-title {
            font-size: 13px;
            font-weight: 700;

            letter-spacing: 0.04em;

            color: var(--text);
        }

        .top-subtitle {
            font-size: 11px;

            color: var(--text-muted);
        }

        .system-state {
            display: flex;
            align-items: center;

            gap: 8px;

            font-size: 10px;
            font-weight: 600;

            color: var(--text-secondary);

            letter-spacing: 0.08em;
        }

        .system-dot {
            width: 7px;
            height: 7px;

            border-radius: 50%;

            background: var(--green);
        }

        /* ====================================================
           LEFT PANEL
        ==================================================== */

        .left-panel {
            width: 290px;
            min-width: 290px;

            height: 100%;

            background: #f8f9fa;

            border-right:
                1px solid var(--border);

            padding: 18px;

            overflow-y: auto;
        }

        .section-label {
            margin-bottom: 8px;

            font-size: 10px;
            font-weight: 700;

            letter-spacing: 0.12em;

            text-transform: uppercase;

            color: var(--text-muted);
        }

        .panel-box {
            background: #ffffff;

            border:
                1px solid var(--border);

            border-radius: 4px;

            box-shadow: var(--shadow);
        }

        .raw-image {
            width: 100%;

            display: block;

            object-fit: contain;
        }

        .roi-image {
            width: 100%;

            min-height: 145px;

            display: block;

            object-fit: contain;

            background: #eef1f4;
        }

        /* ====================================================
           DATA
        ==================================================== */

        .data-row {
            display: flex;

            align-items: center;
            justify-content: space-between;

            padding: 8px 10px;

            border-bottom:
                1px solid #edf0f2;
        }

        .data-row:last-child {
            border-bottom: none;
        }

        .data-key {
            font-size: 10px;

            text-transform: uppercase;

            letter-spacing: 0.06em;

            color: var(--text-muted);
        }

        .data-value {
            font-family:
                "IBM Plex Mono",
                "SFMono-Regular",
                Consolas,
                monospace;

            font-size: 10px;

            color: var(--text);
        }

        /* ====================================================
           CENTER
        ==================================================== */

        .center-panel {
            flex: 1;

            min-width: 0;
            min-height: 0;

            height: 100%;

            position: relative;

            display: flex;
            flex-direction: column;

            background: var(--bg);
        }

        .workspace-header {
            padding: 17px 22px 13px;

            background: #ffffff;

            border-bottom:
                1px solid var(--border);
        }

        .workspace-title {
            font-size: 16px;

            font-weight: 650;

            color: var(--text);
        }

        .workspace-caption {
            margin-top: 2px;

            font-size: 11px;

            color: var(--text-muted);
        }

        /* ====================================================
           WORKSPACE
        ==================================================== */

        .workspace-scroll {
            flex: 1;

            min-height: 0;

            overflow-y: auto;

            padding: 22px;

            padding-bottom: 155px;
        }

        .inspection-card {
            width: 100%;

            background: #ffffff;

            border:
                1px solid var(--border);

            border-radius: 5px;

            box-shadow: var(--shadow);
        }

        .inspection-card-header {
            height: 42px;

            padding: 0 14px;

            display: flex;
            align-items: center;
            justify-content: space-between;

            border-bottom:
                1px solid var(--border);
        }

        .inspection-card-title {
            font-size: 10px;

            font-weight: 700;

            letter-spacing: 0.1em;

            text-transform: uppercase;

            color: var(--text-secondary);
        }

        .inspection-card-meta {
            font-family:
                "IBM Plex Mono",
                "SFMono-Regular",
                Consolas,
                monospace;

            font-size: 9px;

            color: var(--text-muted);
        }

        .inspection-image-wrap {
            width: 100%;

            min-height: 370px;

            padding: 22px;

            background: #eef1f4;

            display: flex;
            align-items: center;
            justify-content: center;
        }

        .inspection-image {
            max-width: 100%;
            max-height: 52vh;

            object-fit: contain;

            display: block;
        }

        /* ====================================================
           PIPELINE
        ==================================================== */

        .pipeline-strip {
            width: 100%;

            display: flex;
            align-items: center;

            margin-top: 14px;

            padding: 12px 14px;

            background: #ffffff;

            border:
                1px solid var(--border);
        }

        .pipeline-item {
            flex: 1;
        }

        .pipeline-label {
            font-size: 9px;

            font-weight: 700;

            text-transform: uppercase;

            letter-spacing: 0.08em;

            color: var(--text-muted);
        }

        .pipeline-status {
            display: flex;
            align-items: center;

            gap: 6px;

            margin-top: 2px;

            font-family:
                "IBM Plex Mono",
                "SFMono-Regular",
                Consolas,
                monospace;

            font-size: 9px;

            color: var(--text-secondary);
        }

        .pipeline-check {
            width: 6px;
            height: 6px;

            border-radius: 50%;

            background: var(--green);
        }

        .pipeline-separator {
            width: 24px;
            height: 1px;

            margin: 0 9px;

            background: #d5dbe0;
        }

        /* ====================================================
           CHAT
        ==================================================== */

        .chat-area {
            width: 100%;

            margin-top: 18px;

            display: flex;
            flex-direction: column;

            gap: 14px;
        }

        .chat-entry {
            width: 100%;
        }

        .chat-label {
            margin-bottom: 5px;

            font-size: 9px;

            font-weight: 700;

            letter-spacing: 0.1em;

            text-transform: uppercase;

            color: var(--text-muted);
        }

        /* ====================================================
           RIGHT PANEL
        ==================================================== */

        .right-panel {
            width: 320px;
            min-width: 320px;

            height: 100%;

            background: #ffffff;

            border-left:
                1px solid var(--border);

            overflow-y: auto;
        }

        .right-section {
            padding: 18px;

            border-bottom:
                1px solid var(--border);
        }

        .right-section-title {
            margin-bottom: 12px;

            font-size: 10px;

            font-weight: 700;

            letter-spacing: 0.11em;

            text-transform: uppercase;

            color: var(--text-muted);
        }

        .status-card {
            padding: 14px;

            background: #f8f9fa;

            border:
                1px solid var(--border);
        }

        .status-value {
            font-size: 17px;

            font-weight: 700;
        }

        .status-description {
            margin-top: 6px;

            font-size: 11px;

            line-height: 1.55;

            color: var(--text-secondary);
        }

        /* ====================================================
           CONFIDENCE
        ==================================================== */

        .metric-header {
            display: flex;

            align-items: center;
            justify-content: space-between;
        }

        .metric-name {
            font-size: 10px;

            text-transform: uppercase;

            letter-spacing: 0.06em;

            color: var(--text-muted);
        }

        .metric-value {
            font-family:
                "IBM Plex Mono",
                "SFMono-Regular",
                Consolas,
                monospace;

            font-size: 12px;
            font-weight: 600;

            color: var(--text);
        }

        .metric-bar {
            width: 100%;

            height: 5px;

            margin-top: 8px;

            background: #e7eaed;
        }

        .metric-fill {
            height: 100%;

            width: 0%;

            background: var(--blue);

            transition: width 0.3s ease;
        }

        /* ====================================================
           OPERATOR INPUT
        ==================================================== */

        .input-area {
            position: absolute;

            left: 0;
            right: 0;
            bottom: 0;

            padding:
                18px 22px 20px;

            background:
                linear-gradient(
                    to bottom,
                    rgba(243,245,247,0),
                    rgba(243,245,247,0.94) 35%,
                    rgba(243,245,247,1) 65%
                );
        }

        .input-label {
            width: 100%;
            max-width: 860px;

            margin:
                0 auto 7px;

            font-size: 9px;

            font-weight: 700;

            letter-spacing: 0.1em;

            text-transform: uppercase;

            color: var(--text-muted);
        }

        .operator-input-shell {
            width: 100%;
            max-width: 860px;

            min-height: 48px;

            margin: 0 auto;

            display: flex;
            align-items: center;

            background: #ffffff;

            border:
                1px solid var(--border-strong);

            border-radius: 4px;

            box-shadow:
                0 2px 7px rgba(
                    16,24,40,0.04
                );
        }

        .operator-input-shell:focus-within {
            border-color: var(--blue);

            box-shadow:
                0 0 0 2px rgba(
                    0,104,181,0.08
                );
        }

        .operator-input-shell
        .q-field__control {
            min-height: 46px !important;
        }

        .operator-input-shell
        .q-field__native {
            color: var(--text) !important;

            font-size: 13px !important;
        }

        .operator-input-shell
        .q-field__native::placeholder {
            color: #9aa2aa !important;
        }

        .submit-button {
            width: 38px;
            height: 38px;

            min-width: 38px;

            margin-right: 5px;

            background:
                var(--blue) !important;

            color: white !important;

            border-radius:
                3px !important;
        }

        .submit-button:hover {
            background:
                #005a9e !important;
        }

        /* ====================================================
           SCROLLBARS
        ==================================================== */

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: #cdd3d9;

            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #b8c0c8;
        }

        /* ====================================================
           MOBILE FALLBACK
        ==================================================== */

        @media (max-width: 1100px) {

            .left-panel {
                width: 240px;
                min-width: 240px;
            }

            .right-panel {
                width: 270px;
                min-width: 270px;
            }
        }

        @media (max-width: 850px) {

            .left-panel,
            .right-panel {
                display: none;
            }
        }

        </style>
        """
    )

    # ========================================================
    # MAIN APP
    # ========================================================

    with ui.column().classes(
            "app-shell gap-0"
    ):

        # ====================================================
        # HEADER
        # ====================================================

        with ui.row().classes(
                "topbar w-full items-center "
                "justify-between no-wrap"
        ):

            with ui.row().classes(
                    "items-center gap-3"
            ):
                with ui.element(
                        "div"
                ).classes(
                    "brand-mark"
                ):
                    ui.icon(
                        "precision_manufacturing",
                        size="18px"
                    )

                with ui.column().classes(
                        "gap-0"
                ):
                    ui.label(
                        "INDUSTRIAL INSPECTION"
                    ).classes(
                        "top-title"
                    )

                    ui.label(
                        "Hybrid Vision Workstation"
                    ).classes(
                        "top-subtitle"
                    )

            with ui.row().classes(
                    "items-center gap-6"
            ):
                with ui.column().classes(
                        "items-end gap-0"
                ):
                    ui.label(
                        "SESSION / GEMINI API"
                    ).classes(
                        "mono text-[10px] "
                        "text-[#59636e]"
                    )

                    ui.label(
                        "INPUT · BROWSER UPLOAD"
                    ).classes(
                        "mono text-[9px] "
                        "text-[#7a838d]"
                    )

                with ui.row().classes(
                        "system-state"
                ):
                    ui.element(
                        "div"
                    ).classes(
                        "system-dot"
                    )

                    ui.label(
                        "SYSTEM READY"
                    )

        # ====================================================
        # THREE COLUMN WORKSPACE
        # ====================================================

        with ui.row().classes(
                "flex-1 w-full min-h-0 "
                "flex-nowrap gap-0"
        ):

            # =================================================
            # LEFT PANEL
            # =================================================

            with ui.column().classes(
                    "left-panel gap-0"
            ):

                ui.label(
                    "SENSOR INPUT"
                ).classes(
                    "section-label"
                )

                # ---------------------------------------------
                # RAW IMAGE
                # ---------------------------------------------

                with ui.column().classes(
                        "panel-box w-full gap-0"
                ):

                    with ui.row().classes(
                            "w-full items-center "
                            "justify-between px-3 py-2 "
                            "border-b border-[#d9dee4]"
                    ):

                        ui.label(
                            "Raw frame"
                        ).classes(
                            "text-[11px] "
                            "font-semibold "
                            "text-[#1f2933]"
                        )

                        ui.label(
                            "SOURCE"
                        ).classes(
                            "mono text-[9px] "
                            "text-[#7a838d]"
                        )

                    raw_image_display = ui.image().classes(
                        "raw-image"
                    )

                    no_sensor_label = ui.label(
                        "NO SENSOR FEED"
                    ).classes(
                        "mono text-[10px] text-[#bd2c2c] "
                        "w-full text-center py-16"
                    )

                    if os.path.exists(IMAGE_PATH):

                        raw_mat = cv2.imread(IMAGE_PATH)

                        if raw_mat is not None:
                            raw_image_display.set_source(
                                encode_image_to_base64(raw_mat)
                            )

                            no_sensor_label.set_visibility(False)

                        else:
                            raw_image_display.set_visibility(False)

                    else:
                        raw_image_display.set_visibility(False)

                image_uploader = ui.upload(
                    label="Upload inspection image",
                    on_upload=handle_image_upload,
                    auto_upload=True,
                    max_files=1,
                ).props(
                    'accept="image/*"'
                ).classes(
                    "w-full mt-3"
                )

                # ---------------------------------------------
                # ROI
                # ---------------------------------------------

                ui.label(
                    "LOCALIZED WORKPIECE"
                ).classes(
                    "section-label mt-5"
                )

                with ui.column().classes(
                        "panel-box w-full gap-0"
                ):

                    with ui.row().classes(
                            "w-full items-center "
                            "justify-between px-3 py-2 "
                            "border-b border-[#d9dee4]"
                    ):
                        ui.label(
                            "VLM input"
                        ).classes(
                            "text-[11px] "
                            "font-semibold "
                            "text-[#1f2933]"
                        )

                        ui.label(
                            "ROI"
                        ).classes(
                            "mono text-[9px] "
                            "text-[#0068b5]"
                        )

                    roi_image_display = ui.image().classes(
                        "roi-image"
                    )

                # ---------------------------------------------
                # LOCALIZATION TELEMETRY
                # ---------------------------------------------

                ui.label(
                    "LOCALIZATION TELEMETRY"
                ).classes(
                    "section-label mt-5"
                )

                with ui.column().classes(
                        "panel-box w-full gap-0"
                ):

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Detection"
                        ).classes(
                            "data-key"
                        )

                        yolo_status_label = ui.label(
                            "Ready"
                        ).classes(
                            "data-value "
                            "text-right"
                        )

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Engine"
                        ).classes(
                            "data-key"
                        )

                        ui.label(
                            "YOLOv8n"
                            if HAS_YOLO
                            else "Contour Fusion"
                        ).classes(
                            "data-value"
                        )

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Confidence"
                        ).classes(
                            "data-key"
                        )

                        yolo_conf_label = ui.label(
                            "—"
                        ).classes(
                            "data-value"
                        )

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Bounding box"
                        ).classes(
                            "data-key"
                        )

                        yolo_bbox_label = ui.label(
                            "None"
                        ).classes(
                            "data-value "
                            "text-right"
                        )

                # ---------------------------------------------
                # EDGE RUNTIME
                # ---------------------------------------------

                ui.label(
                    "EDGE RUNTIME"
                ).classes(
                    "section-label mt-5"
                )

                with ui.column().classes(
                        "panel-box w-full gap-0"
                ):

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Vision"
                        ).classes(
                            "data-key"
                        )

                        ui.label(
                            "YOLOv8n"
                            if HAS_YOLO
                            else "Contour Fusion"
                        ).classes(
                            "data-value"
                        )

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "VLM"
                        ).classes(
                            "data-key"
                        )

                        ui.label(
                            MODEL_NAME
                        ).classes(
                            "data-value "
                            "text-right"
                        )

                    with ui.row().classes(
                            "data-row"
                    ):
                        ui.label(
                            "Runtime"
                        ).classes(
                            "data-key"
                        )

                        ui.label(
                            "GEMINI API / CLOUD"
                        ).classes(
                            "data-value"
                        )

            # =================================================
            # CENTER PANEL
            # =================================================

            with ui.column().classes(
                    "center-panel gap-0"
            ):

                # ---------------------------------------------
                # HEADER
                # ---------------------------------------------

                with ui.column().classes(
                        "workspace-header gap-0"
                ):
                    with ui.row().classes(
                            "items-center "
                            "justify-between"
                    ):
                        with ui.column().classes(
                                "gap-0"
                        ):
                            ui.label(
                                "WORKPIECE INSPECTION"
                            ).classes(
                                "workspace-title"
                            )

                            ui.label(
                                "Visual evidence review "
                                "and operator interaction"
                            ).classes(
                                "workspace-caption"
                            )

                        ui.label(
                            "ASSEMBLY · BOLT"
                        ).classes(
                            "mono text-[9px] "
                            "text-[#59636e]"
                        )

                # ---------------------------------------------
                # SCROLLABLE CONTENT
                # ---------------------------------------------

                with ui.column().classes(
                        "workspace-scroll gap-0"
                ):
                    # =========================================
                    # INSPECTION IMAGE
                    # =========================================

                    with ui.column().classes(
                            "inspection-card "
                            "w-full gap-0"
                    ):
                        with ui.row().classes(
                                "inspection-card-header"
                        ):
                            ui.label(
                                "LOCALIZED INSPECTION VIEW"
                            ).classes(
                                "inspection-card-title"
                            )

                            ui.label(
                                "ROI → VLM"
                            ).classes(
                                "inspection-card-meta"
                            )

                        with ui.element(
                                "div"
                        ).classes(
                            "inspection-image-wrap"
                        ):
                            inspection_image = ui.image().classes(
                                "inspection-image"
                            )

                    # =========================================
                    # PIPELINE
                    # =========================================

                    with ui.row().classes(
                            "pipeline-strip"
                    ):
                        with ui.column().classes(
                                "pipeline-item gap-1"
                        ):
                            ui.label(
                                "Sensor"
                            ).classes(
                                "pipeline-label"
                            )

                            with ui.row().classes(
                                    "pipeline-status"
                            ):
                                ui.element(
                                    "div"
                                ).classes(
                                    "pipeline-check"
                                )

                                ui.label(
                                    "Frame received"
                                )

                        ui.element(
                            "div"
                        ).classes(
                            "pipeline-separator"
                        )

                        with ui.column().classes(
                                "pipeline-item gap-1"
                        ):
                            ui.label(
                                "Localization"
                            ).classes(
                                "pipeline-label"
                            )

                            with ui.row().classes(
                                    "pipeline-status"
                            ):
                                ui.element(
                                    "div"
                                ).classes(
                                    "pipeline-check"
                                )

                                ui.label(
                                    "Object localized"
                                )

                        ui.element(
                            "div"
                        ).classes(
                            "pipeline-separator"
                        )

                        with ui.column().classes(
                                "pipeline-item gap-1"
                        ):
                            ui.label(
                                "VLM"
                            ).classes(
                                "pipeline-label"
                            )

                            with ui.row().classes(
                                    "pipeline-status"
                            ):
                                ui.element(
                                    "div"
                                ).classes(
                                    "pipeline-check"
                                )

                                ui.label(
                                    "Evidence analysis"
                                )

                        ui.element(
                            "div"
                        ).classes(
                            "pipeline-separator"
                        )

                        with ui.column().classes(
                                "pipeline-item gap-1"
                        ):
                            ui.label(
                                "Assessment"
                            ).classes(
                                "pipeline-label"
                            )

                            with ui.row().classes(
                                    "pipeline-status"
                            ):
                                ui.element(
                                    "div"
                                ).classes(
                                    "pipeline-check"
                                )

                                ui.label(
                                    "Operator review"
                                )

                    # =========================================
                    # CHAT / INSPECTION LOG
                    # =========================================

                    ui.label(
                        "INSPECTION LOG"
                    ).classes(
                        "section-label mt-6"
                    )

                    chat_area = ui.column().classes(
                        "chat-area gap-4"
                    )

                    with chat_area:
                        with ui.column().classes(
                                "chat-entry gap-0"
                        ):
                            ui.label(
                                "SYSTEM"
                            ).classes(
                                "chat-label"
                            )

                            ui.chat_message(
                                (
                                    "Evidence-gated inspection pipeline active. "
                                    "Upload an image from the Sensor Input panel, "
                                    "then submit an inspection query. Use /teach <part> "
                                    "for a known-good reference, /evidence after "
                                    "uploading a requested follow-up view, and /newbatch "
                                    "to begin a fresh batch scope."
                                ),
                                name="",
                                sent=False
                            ).classes(
                                "w-full text-[13px] "
                                "leading-[1.6] "
                                "text-[#1f2933] "
                                "bg-white "
                                "border border-[#d9dee4] "
                                "rounded-[4px] "
                                "shadow-sm"
                            )

                # ---------------------------------------------
                # INPUT
                # ---------------------------------------------

                with ui.column().classes(
                        "input-area gap-0"
                ):
                    ui.label(
                        "OPERATOR QUERY"
                    ).classes(
                        "input-label"
                    )

                    with ui.row().classes(
                            "operator-input-shell"
                    ):
                        chat_input = ui.input(
                            placeholder=(
                                "Inspect bolt health, "
                                "check thread integrity, "
                                "assess visible damage…"
                            )
                        ).props(
                            "borderless"
                        ).classes(
                            "flex-grow px-3 "
                            "bg-transparent"
                        )

                        submit_button = ui.button(
                            icon="arrow_forward"
                        ).props(
                            "flat"
                        ).classes(
                            "submit-button"
                        )

            # =================================================
            # RIGHT PANEL
            # =================================================

            with ui.column().classes(
                    "right-panel gap-0"
            ):

                # ---------------------------------------------
                # STATE
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "CURRENT STATE"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "status-card gap-0"
                    ):
                        current_state_label = ui.label(
                            "PENDING INSPECTION"
                        ).classes(
                            "status-value "
                            "text-[#59636e]"
                        )

                        current_state_description = ui.label(
                            "No operator assessment has "
                            "been generated for the "
                            "current frame."
                        ).classes(
                            "status-description"
                        )

                # ---------------------------------------------
                # LOCALIZATION
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "LOCALIZATION"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "panel-box w-full gap-0"
                    ):
                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "Status"
                            ).classes(
                                "data-key"
                            )

                            system_detection_label = ui.label(
                                "Pending"
                            ).classes(
                                "data-value"
                            )

                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "Bounding box"
                            ).classes(
                                "data-key"
                            )

                            bbox_display_label = ui.label(
                                "—"
                            ).classes(
                                "data-value "
                                "text-right"
                            )

                # ---------------------------------------------
                # CONFIDENCE
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "DETECTION CONFIDENCE"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "gap-0"
                    ):
                        with ui.row().classes(
                                "metric-header"
                        ):
                            ui.label(
                                "YOLO localization"
                            ).classes(
                                "metric-name"
                            )

                            conf_display_label = ui.label(
                                "—"
                            ).classes(
                                "metric-value"
                            )

                        with ui.element(
                                "div"
                        ).classes(
                            "metric-bar"
                        ):
                            confidence_fill = ui.element(
                                "div"
                            ).classes(
                                "metric-fill"
                            )

                # ---------------------------------------------
                # PROCESSING CHAIN
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "PROCESSING CHAIN"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "panel-box w-full gap-0"
                    ):
                        stages = [
                            (
                                "01",
                                "Image acquisition"
                            ),
                            (
                                "02",
                                "YOLO localization"
                            ),
                            (
                                "03",
                                "ROI isolation"
                            ),
                            (
                                "04",
                                "VLM inspection"
                            ),
                            (
                                "05",
                                "Evidence-gated response"
                            ),
                        ]

                        for number, label in stages:
                            with ui.row().classes(
                                    "data-row "
                                    "items-center"
                            ):
                                ui.label(
                                    number
                                ).classes(
                                    "mono text-[9px] "
                                    "text-[#0068b5] "
                                    "w-6"
                                )

                                ui.label(
                                    label
                                ).classes(
                                    "text-[11px] "
                                    "text-[#1f2933]"
                                )

                # ---------------------------------------------
                # INDUSTRIAL MEMORY
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "INDUSTRIAL MEMORY"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "panel-box w-full gap-0"
                    ):
                        with ui.row().classes("data-row"):
                            ui.label("Batch").classes("data-key")
                            batch_id_label = ui.label(
                                memory.get_active_batch_id(DEFAULT_PART_NAME) or "AUTO"
                            ).classes("data-value text-right")

                        with ui.row().classes("data-row"):
                            ui.label("Episodes").classes("data-key")
                            memory_episode_label = ui.label("0").classes("data-value")

                        with ui.row().classes("data-row"):
                            ui.label("Visual drift").classes("data-key")
                            visual_drift_label = ui.label("—").classes("data-value")

                        with ui.row().classes("data-row"):
                            ui.label("Semantic drift").classes("data-key")
                            semantic_drift_label = ui.label("—").classes("data-value")

                        with ui.row().classes("data-row"):
                            ui.label("Cross-modal").classes("data-key")
                            concordance_label = ui.label("—").classes("data-value")

                    new_batch_button = ui.button(
                        "New Batch", icon="playlist_add"
                    ).props(
                        "outline no-caps"
                    ).classes(
                        "w-full mt-3"
                    )

                    wipe_memory_button = ui.button(
                        "Wipe All Memory", icon="delete_forever"
                    ).props(
                        "outline no-caps color=negative"
                    ).classes(
                        "w-full mt-2"
                    )

                    with ui.dialog() as wipe_dialog, ui.card().classes("w-[430px]"):
                        ui.label("Wipe all Industrial Memory?").classes(
                            "text-[15px] font-semibold text-[#1f2933]"
                        )
                        ui.label(
                            "This permanently deletes taught references, immutable inspection "
                            "episodes, archived good/bad images, embeddings, batches, and override audits."
                        ).classes("text-[12px] text-[#59636e]")
                        with ui.row().classes("w-full justify-end gap-2 mt-3"):
                            ui.button("Cancel", on_click=wipe_dialog.close).props("flat no-caps")
                            confirm_wipe_button = ui.button(
                                "Confirm Wipe", icon="delete_forever"
                            ).props("color=negative no-caps")

                # ---------------------------------------------
                # MODEL RUNTIME
                # ---------------------------------------------

                with ui.column().classes(
                        "right-section gap-0"
                ):
                    ui.label(
                        "MODEL RUNTIME"
                    ).classes(
                        "right-section-title"
                    )

                    with ui.column().classes(
                            "panel-box w-full gap-0"
                    ):
                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "Vision model"
                            ).classes(
                                "data-key"
                            )

                            ui.label(
                                "YOLOv8n"
                                if HAS_YOLO
                                else "Contour"
                            ).classes(
                                "data-value"
                            )

                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "VLM"
                            ).classes(
                                "data-key"
                            )

                            ui.label(
                                "Gemini 3.5 Flash-Lite"
                            ).classes(
                                "data-value"
                            )

                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "Context"
                            ).classes(
                                "data-key"
                            )

                            ui.label(
                                "4096 tokens"
                            ).classes(
                                "data-value"
                            )

                        with ui.row().classes(
                                "data-row"
                        ):
                            ui.label(
                                "Temperature"
                            ).classes(
                                "data-key"
                            )

                            ui.label(
                                "0.0"
                            ).classes(
                                "data-value"
                            )

    # ========================================================
    # SEND MESSAGE - MODULAR PIPELINE INTEGRATION
    # ========================================================

    def _render_pipeline_result(result: dict) -> str:
        decision = result.get("decision", "UNKNOWN")
        lines = [f"Decision: {decision}"]
        if result.get("confidence") is not None:
            lines.append(f"Confidence: {result['confidence']}%")
        if result.get("reasoning"):
            lines.append(f"Visual evidence: {result['reasoning']}")
        if result.get("recommended_action"):
            lines.append(f"Action: {result['recommended_action']}")
        if result.get("requested_evidence"):
            lines.append(f"Requested evidence: {result['requested_evidence']}")
        if result.get("message"):
            lines.append(result["message"])
        if result.get("override_status"):
            lines.append(f"Override status: {result['override_status']}")
        if result.get("original_claim"):
            lines.append(f"Original claim: {result['original_claim']}")
        if result.get("adjudication"):
            lines.append(f"Claim adjudication: {result['adjudication']}")
        adequacy = result.get("evidence_adequacy") or {}
        if adequacy:
            lines.append(
                "Evidence adequacy: "
                + ("ADEQUATE" if adequacy.get("evidence_adequate") else "INADEQUATE")
            )
            if adequacy.get("adequacy_reason"):
                lines.append(f"Adequacy reason: {adequacy['adequacy_reason']}")
        analytics = result.get("memory_analytics") or {}
        if analytics:
            if analytics.get("batch_id"):
                lines.append(f"Batch: {analytics['batch_id']}")
            if analytics.get("visual_drift") is not None:
                lines.append(f"Visual drift: {analytics['visual_drift']:.3f}")
            if analytics.get("semantic_drift") is not None:
                lines.append(f"Semantic drift: {analytics['semantic_drift']:.3f}")
            if analytics.get("concordance"):
                lines.append(f"Cross-modal: {analytics['concordance']}")
        return "\n".join(lines)

    def _metric_text(value):
        return "—" if value is None else f"{float(value):.3f}"

    def refresh_memory_panel():
        snapshot = memory.get_memory_snapshot(session_state["part_name"])
        batch_id_label.set_text(snapshot.get("active_batch_id") or "AUTO")
        memory_episode_label.set_text(str(snapshot.get("batch_episodes", 0)))
        metrics = snapshot.get("latest_metrics") or {}
        visual_drift_label.set_text(_metric_text(metrics.get("visual_drift")))
        semantic_drift_label.set_text(_metric_text(metrics.get("semantic_drift")))
        concordance_label.set_text(metrics.get("concordance") or "—")

    def start_new_batch_ui():
        batch = memory.start_new_batch(session_state["part_name"])
        session_state["awaiting_inspection_id"] = None
        session_state["last_inspection_id"] = None
        refresh_memory_panel()
        current_state_label.set_text("NEW BATCH")
        current_state_description.set_text(
            f"Started {batch['batch_id']} for {session_state['part_name']}."
        )
        ui.notify(f"New batch started: {batch['batch_id']}", type="positive")

    def confirmed_wipe_memory():
        memory.wipe_all_memory()
        session_state["awaiting_inspection_id"] = None
        session_state["last_inspection_id"] = None
        wipe_dialog.close()
        refresh_memory_panel()
        current_state_label.set_text("MEMORY WIPED")
        current_state_description.set_text(
            "All learned references, episodes, archives, embeddings, batches and override audits were deleted."
        )
        ui.notify("Industrial Memory wiped.", type="warning")

    async def send_message():
        user_text = (chat_input.value or "").strip()
        if not user_text:
            return
        chat_input.value = ""

        with chat_area:
            with ui.column().classes("chat-entry gap-0"):
                ui.label("OPERATOR").classes("chat-label")
                ui.chat_message(
                    user_text, name="", sent=True
                ).classes(
                    "w-fit max-w-[85%] ml-auto text-[13px] "
                    "text-[#1f2933] bg-[#eaf3fa] border "
                    "border-[#bdd6e8] rounded-[4px] shadow-none"
                )

        # /part bolt changes the active part without running inference.
        if user_text.lower().startswith("/part "):
            session_state["part_name"] = user_text.split(maxsplit=1)[1].strip()
            refresh_memory_panel()
            response_text = f"Active part set to: {session_state['part_name']}"
            with chat_area:
                ui.chat_message(response_text, name="", sent=False)
            return

        if user_text.lower() == "/newbatch":
            start_new_batch_ui()
            with chat_area:
                ui.chat_message(
                    f"Started a new batch for {session_state['part_name']}.",
                    name="", sent=False
                )
            return

        if not os.path.exists(IMAGE_PATH):
            with chat_area:
                ui.chat_message(
                    "No image loaded. Upload an inspection image first.", name="", sent=False
                )
            return

        # Localize every submitted frame. For /evidence, upload the newly
        # captured angle in the browser before submitting the command.
        crop_path, meta = localize_and_crop_object(IMAGE_PATH)
        cropped_mat = cv2.imread(crop_path)
        if cropped_mat is not None:
            encoded = encode_image_to_base64(cropped_mat)
            roi_image_display.set_source(encoded)
            inspection_image.set_source(encoded)

        yolo_status_label.set_text(meta["status"])
        yolo_bbox_label.set_text(str(meta["bbox"]))
        yolo_conf_label.set_text(f'{meta["confidence"]:.2f}')
        system_detection_label.set_text(meta["status"])
        bbox_display_label.set_text(str(meta["bbox"]))
        conf_display_label.set_text(f'{int(meta["confidence"] * 100)}%')
        confidence_fill.style(f"width: {meta['confidence'] * 100:.1f}%;")
        current_state_label.set_text("ANALYZING")
        current_state_description.set_text(
            "ROI localized; modular inspection pipeline is running."
        )

        with chat_area:
            with ui.column().classes("chat-entry gap-0"):
                ui.label("INSPECTION ANALYSIS").classes("chat-label")
                typing_indicator = ui.chat_message(
                    "Processing inspection evidence…", name="", sent=False
                )

        loop = asyncio.get_running_loop()
        part_name = session_state["part_name"]

        try:
            lower = user_text.lower()

            if lower.startswith("/teach"):
                tokens = user_text.split(maxsplit=1)
                if len(tokens) == 2 and tokens[1].strip():
                    part_name = tokens[1].strip()
                    session_state["part_name"] = part_name
                teaching = await loop.run_in_executor(
                    None,
                    lambda: LearnEngine().teach_new_part(
                        crop_path, part_name, "Known-good sample supplied from UI."
                    ),
                )
                response_text = (
                    f"Teaching status: {teaching['status']}\n"
                    f"Part: {part_name}\n"
                    f"{json.dumps(teaching['detail'], indent=2)}"
                )
                current_state_label.set_text(teaching["status"])

            elif lower.startswith("/evidence"):
                pending_id = session_state.get("awaiting_inspection_id")
                if not pending_id:
                    response_text = "No inspection is currently waiting for extra evidence."
                else:
                    result = await loop.run_in_executor(
                        None,
                        continue_with_new_evidence,
                        pending_id,
                        crop_path,
                    )
                    response_text = _render_pipeline_result(result)
                    session_state["last_inspection_id"] = result.get(
                        "inspection_id", pending_id
                    )
                    session_state["awaiting_inspection_id"] = (
                        result.get("inspection_id")
                        if result.get("decision") == "AWAITING_OPERATOR_EVIDENCE"
                        else None
                    )
                    current_state_label.set_text(result.get("decision", "RESULT"))

            elif lower.startswith("/override "):
                last_id = session_state.get("last_inspection_id")
                if not last_id:
                    response_text = "No prior inspection is available to override."
                else:
                    justification = user_text.split(maxsplit=1)[1].strip()
                    result = await loop.run_in_executor(
                        None,
                        lambda: request_operator_override(
                            last_id,
                            "local_operator",
                            justification,
                            crop_path,
                        ),
                    )
                    response_text = _render_pipeline_result(result)
                    current_state_label.set_text(result.get("decision", "RESULT"))

            else:
                result = await loop.run_in_executor(
                    None,
                    run_inspection_pipeline,
                    crop_path,
                    part_name,
                    user_text,
                )
                response_text = _render_pipeline_result(result)
                session_state["last_inspection_id"] = result.get("inspection_id")
                session_state["awaiting_inspection_id"] = (
                    result.get("inspection_id")
                    if result.get("decision") == "AWAITING_OPERATOR_EVIDENCE"
                    else None
                )
                current_state_label.set_text(result.get("decision", "RESULT"))

            refresh_memory_panel()
            typing_indicator.set_visibility(False)
            current_state_description.set_text(response_text.split("\n", 1)[0])
            with chat_area:
                with ui.column().classes("chat-entry gap-0"):
                    ui.label("INSPECTION ANALYSIS").classes("chat-label")
                    ui.chat_message(
                        response_text, name="", sent=False
                    ).classes(
                        "w-full text-[13px] leading-[1.6] text-[#1f2933] "
                        "bg-white border border-[#d9dee4] rounded-[4px] shadow-sm"
                    )

        except Exception as e:
            try:
                typing_indicator.set_visibility(False)
            except Exception:
                pass
            current_state_label.set_text("ERROR")
            current_state_description.set_text(str(e))
            with chat_area:
                ui.chat_message(
                    f"Pipeline Error: {e}", name="", sent=False
                ).classes("text-[#bd2c2c]")

        ui.run_javascript(
            """
            const el = document.querySelector('.workspace-scroll');
            if (el) el.scrollTop = el.scrollHeight;
            """
        )

    # ========================================================
    # CONNECT INPUT EVENTS
    # ========================================================

    refresh_memory_panel()
    new_batch_button.on("click", start_new_batch_ui)
    wipe_memory_button.on("click", wipe_dialog.open)
    confirm_wipe_button.on("click", confirmed_wipe_memory)

    chat_input.on(
        "keydown.enter",
        send_message
    )

    submit_button.on(
        "click",
        send_message
    )


# ============================================================
# RUN
# ============================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Industrial Inspection Workstation",
        port=8880,
        show=True,
        dark=False
    )
