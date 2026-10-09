"""Images for the agent: frame sheets from the recorded video and labelled dwell heatmaps."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import db as D

EXPORT_DIR = Path(os.environ.get("SA_EXPORT_DIR") or (D.PKG_DIR.parent / "frame_exports"))


def _out(name: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR / name


def read_frame(video: str, t_sec: float, fps: float):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None
    real_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t_sec * real_fps))))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def frame_sheet(run: dict, times: list[float], zones: list[dict], detections: dict, crop=None,
                overlay: bool = True, tag: str = "sheet") -> tuple[str | None, list[str]]:
    """Grid of frames (max 9) with the wall-clock time burned in. detections: {t: [bbox,...]}.
    crop: (x1,y1,x2,y2) pixel box. Returns (path or None, notes)."""
    import cv2
    notes, tiles = [], []
    fps = float(run["config"].get("fps") or 25)
    for t in times[:9]:
        img = read_frame(run["source_path"], t, fps)
        if img is None:
            notes.append(f"{D.fmt_t(run, t)}: frame not readable (video missing or offline)")
            img = np.full((360, 640, 3), 30, np.uint8)
            cv2.putText(img, "NO FRAME", (220, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        elif overlay:
            for z in zones:
                pts = np.array(z["points"], np.int32)
                col = (0, 200, 255) if z["kind"] == "floor_area" else (255, 200, 0)
                cv2.polylines(img, [pts], True, col, 1)
                cv2.putText(img, z["name"], (int(z["box"][0]) + 3, int(z["box"][1]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            for i, b in enumerate(detections.get(t, [])):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"#{i + 1}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if crop and img.shape[0] > 400:
            x1, y1, x2, y2 = [int(v) for v in crop]
            h, w = img.shape[:2]
            img = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        tw = 1280 if len(times) == 1 else 640
        img = cv2.resize(img, (tw, int(tw * img.shape[0] / max(img.shape[1], 1))))
        cv2.rectangle(img, (0, 0), (250, 26), (0, 0, 0), -1)
        cv2.putText(img, D.fmt_t(run, t, with_date=False), (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        return None, notes
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for t in tiles]
    cols = 1 if len(tiles) == 1 else (2 if len(tiles) <= 4 else 3)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    path = _out(f"{run['camera']}_{tag}_{int(times[0])}_{len(times)}.jpg")
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path), notes


def heatmap(run: dict, cells: dict, cell_px: int, zones: list[dict], hotspots: list, W: int, H: int, tag: str) -> str:
    """Labelled dwell heatmap over the reference frame (or a plain map)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(12, 12 * H / W), dpi=100)
    bg = read_frame(run["source_path"], run["start_sec"] + 1, float(run["config"].get("fps") or 25))
    if bg is not None:
        ax.imshow(bg[:, :, ::-1], alpha=0.55)
    for z in zones:
        ax.add_patch(Polygon(z["points"], closed=True, fill=False, lw=0.8,
                             ec="#1f5aa8" if z["kind"] == "floor_area" else "#555"))
        ax.text(z["box"][0] + 4, z["box"][1] + 14, z["name"], fontsize=7, color="#111")
    mx = max(cells.values()) if cells else 1
    for (cx, cy), v in cells.items():
        f = (v / mx) ** 0.5
        ax.add_patch(Rectangle((cx * cell_px, cy * cell_px), cell_px, cell_px, color=(0.85, 0.15, 0.1, 0.12 + 0.75 * f), lw=0))
    for i, (sec, x, y, _) in enumerate(hotspots):
        ax.text(x + 10, y - 10, f"H{i + 1}: {sec / 60:.1f} min", fontsize=9, fontweight="bold",
                bbox=dict(fc="white", ec="#333", pad=1.5))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"{run['camera']} dwell heatmap (all people, no time order). Boxes = hotspot minutes (match the text list).", fontsize=9, loc="left")
    path = _out(f"{run['camera']}_heat_{tag}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return str(path)
