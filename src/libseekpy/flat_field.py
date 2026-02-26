"""
Create flat-field calibration image for Seek Compact XR.
Warmup frames, then average N frames and save as 16-bit PNG.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .seek_compact_xr import SeekCompactXR
from .constants import THERMAL_HEIGHT, THERMAL_WIDTH


def create_flat_field(
    output_path: str = "flat_field.png",
    warmup: int = 10,
    smoothing: int = 100,
    ffc_path: Optional[str] = None,
) -> None:
    """
    Open Seek Compact XR, discard warmup frames, average smoothing frames
    after retrieve(), save as 16-bit PNG for use as additional FFC.
    Cover the lens with uniform temperature before running.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = SeekCompactXR(ffc_filename=ffc_path)
    if not cam.open():
        raise RuntimeError("Failed to open Seek Compact XR camera")

    try:
        for _ in range(warmup):
            frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
            if not cam.grab():
                raise RuntimeError("Frame acquisition failed during warmup")
            cam.retrieve(frame)

        accum: Optional[np.ndarray] = None
        for _ in range(smoothing):
            frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
            if not cam.grab():
                raise RuntimeError("Frame acquisition failed during averaging")
            cam.retrieve(frame)
            f = frame.astype(np.float64)
            if accum is None:
                accum = f.copy()
            else:
                accum += f

        if accum is None:
            raise RuntimeError("No frames averaged")
        accum /= smoothing
        out = np.clip(accum, 0, 65535).astype(np.uint16)

        out_min, out_max = int(out.min()), int(out.max())
        out_mean = float(out.mean())
        if out_max == 0:
            raise RuntimeError(
                "Flat field is all zeros - check camera connection and that the lens is covered with uniform temperature"
            )
        print(f"Flat field range: min={out_min} max={out_max} mean={out_mean:.0f}")

        out_contiguous = np.ascontiguousarray(out)
        try:
            import cv2
            if not cv2.imwrite(str(path), out_contiguous):
                raise RuntimeError("cv2.imwrite failed for 16-bit PNG")
            preview_path = path.parent / (path.stem + "_preview.png")
            preview = np.clip((out_contiguous.astype(np.float64) - out_min) * 255.0 / max(out_max - out_min, 1), 0, 255).astype(np.uint8)
            cv2.imwrite(str(preview_path), preview)
            print(f"Saved 16-bit (for FFC): {path}")
            print(f"Saved 8-bit preview (to verify data): {preview_path}")
        except ImportError:
            from PIL import Image
            Image.fromarray(out_contiguous).save(str(path))
            preview_path = path.parent / (path.stem + "_preview.png")
            preview = np.clip((out_contiguous.astype(np.float64) - out_min) * 255.0 / max(out_max - out_min, 1), 0, 255).astype(np.uint8)
            Image.fromarray(preview).save(str(preview_path))
            print(f"Saved 16-bit (for FFC): {path}")
            print(f"Saved 8-bit preview (to verify data): {preview_path}")
    finally:
        cam.close()
