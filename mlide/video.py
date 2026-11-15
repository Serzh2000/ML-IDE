"""Video helpers: probing and frame extraction (pure logic, no Qt).

Non-ASCII Windows paths: cv2.VideoCapture goes through the ANSI API on many
builds. Strategy (no full copy unless unavoidable):
  1. cv2.VideoCapture(path) directly.
  2. Windows 8.3 short path (GetShortPathNameW) - free, works when 8.3 names exist.
  3. NTFS hard link in the temp dir (free, same volume only).
  4. Plain copy to a temp file (last resort; removed when done).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from . import imgio


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _short_path(path: str):
    if os.name != "nt":
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)
        if 0 < n < 1024:
            return buf.value
    except Exception:
        pass
    return None


@contextmanager
def _open_capture(path):
    """Yield an opened cv2.VideoCapture or raise ValueError. Cleans up temp files."""
    path = str(path)
    if not os.path.isfile(path):
        raise ValueError(f"Видеофайл не найден: {path}")
    tmp = None
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened() and not _is_ascii(path):
            cap.release()
            sp = _short_path(path)
            if sp and _is_ascii(sp):
                cap = cv2.VideoCapture(sp)
        if not cap.isOpened() and not _is_ascii(path):
            cap.release()
            fd, tmp = tempfile.mkstemp(suffix=Path(path).suffix or ".mp4", prefix="mlide_vid_")
            os.close(fd)
            os.remove(tmp)
            try:
                os.link(path, tmp)
            except OSError:
                shutil.copyfile(path, tmp)
            cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео (неподдерживаемый формат или файл повреждён): {path}")
        yield cap
    finally:
        if cap is not None:
            cap.release()
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def video_info(path) -> dict:
    """dict(frames, fps, width, height, duration_sec). frames may be an estimate (container metadata)."""
    with _open_capture(path) as cap:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frames < 0:
        frames = 0
    if not (0 < fps < 1000):
        fps = 0.0
    dur = frames / fps if fps > 0 and frames > 0 else 0.0
    return {"frames": frames, "fps": fps, "width": w, "height": h, "duration_sec": dur}


def _safe_prefix(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or "video"


_SCENE_SIZE = (64, 36)


def _scene_sig(frame) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, _SCENE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def extract_frames(video_path, out_dir, every_n=None, every_sec=None, scene_threshold=None,
                   max_frames=None, prefix=None, quality=95, progress=None,
                   start_sec=0, end_sec=None) -> list[Path]:
    """Extract frames to `out_dir` as JPEG `<prefix>_<frameindex:06d>.jpg`; returns written paths.

    Exactly one mode may be given (default every_sec=1.0):
      every_n         keep frame indices start, start+n, ...
      every_sec       keep one frame per `every_sec` seconds (needs fps; falls back to 25)
      scene_threshold keep a frame when the mean absolute difference (0..1) between its
                      64x36 grayscale and the LAST KEPT frame's exceeds the threshold.
                      0..1 = mean(|a-b|)/255. Typical: 0.05 sensitive, 0.10-0.15 hard cuts
                      only. The first frame is always kept.
    progress(done, total) -> False cancels (already written files stay; they are returned).
    `total` is the number of frames in the range (0 if unknown); `done` frames processed.
    start_sec/end_sec limit the range; frame indices in names are absolute video indices.
    """
    modes = [m for m in (every_n, every_sec, scene_threshold) if m is not None]
    if len(modes) > 1:
        raise ValueError("Укажите только один режим: every_n, every_sec или scene_threshold.")
    if not modes:
        every_sec = 1.0
    if every_n is not None and int(every_n) < 1:
        raise ValueError("every_n должен быть >= 1.")
    if every_sec is not None and every_sec <= 0:
        raise ValueError("every_sec должен быть > 0.")
    if scene_threshold is not None and not (0 <= scene_threshold <= 1):
        raise ValueError("scene_threshold должен быть в диапазоне 0..1.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(prefix if prefix is not None else Path(video_path).stem)
    written: list[Path] = []

    with _open_capture(video_path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0 < fps < 1000):
            fps = 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0, int(round(start_sec * fps))) if start_sec else 0
        end = int(round(end_sec * fps)) if end_sec is not None else None  # exclusive
        if end is None and n_total > 0:
            end = n_total
        total = max(0, end - start) if end is not None else 0
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            # trust the real position (some codecs seek to a keyframe)
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            idx = pos if pos >= 0 else start
            if idx < start:  # inexact seek: skip forward
                while idx < start and cap.grab():
                    idx += 1
        else:
            idx = 0

        step = float(every_n) if every_n is not None else (every_sec * fps if every_sec is not None else 0.0)
        next_target = float(idx)
        last_sig = None
        done = 0

        def report(force=False):
            if progress is not None and (force or done % 8 == 0):
                return progress(done, total) is not False
            return True

        while True:
            if end is not None and idx >= end:
                break
            if max_frames is not None and len(written) >= max_frames:
                break
            if not cap.grab():
                break
            keep = False
            frame = None
            if scene_threshold is not None:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                sig = _scene_sig(frame)
                if last_sig is None or float(np.abs(sig - last_sig).mean()) / 255.0 > scene_threshold:
                    keep = True
                    last_sig = sig
            elif idx >= next_target - 1e-6:
                keep = True
                next_target += step
                ok, frame = cap.retrieve()
                if not ok:
                    break
            if keep:
                p = out_dir / f"{prefix}_{idx:06d}.jpg"
                imgio.imwrite(p, frame, quality)
                written.append(p)
            idx += 1
            done += 1
            if not report(force=keep):
                break
        if progress is not None and done and done % 8 != 0:
            progress(done, total)
    return written
