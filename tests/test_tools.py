"""Tests for mlide.video and mlide.dedupe. Run: python tests/test_tools.py (or pytest)."""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from mlide import dedupe, imgio, video

FPS = 10
SCENE_LEN = 24


def _scene_frame(scene, k):
    img = np.zeros((90, 160, 3), np.uint8)
    if scene == 0:
        img[:] = (200, 60, 40)
    elif scene == 1:
        img[:] = (30, 200, 60)
        img[:, 80:] = (240, 240, 240)
    else:
        img[:] = (20, 20, 20)
        img[30:60, :] = (40, 60, 230)
    x = 5 + k * 3
    cv2.rectangle(img, (x, 5), (x + 8, 13), (128, 128, 128), -1)  # small motion
    return img


def make_video(dst_dir):
    """3 scenes x 24 frames @10fps (7.2 s) -> returns path inside dst_dir (non-ASCII ok)."""
    tmp = Path(tempfile.mkdtemp())
    ascii_path = tmp / "v.avi"
    vw = cv2.VideoWriter(str(ascii_path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 90))
    assert vw.isOpened(), "VideoWriter MJPG/avi unavailable"
    for s in range(3):
        for k in range(SCENE_LEN):
            vw.write(_scene_frame(s, k))
    vw.release()
    dst = Path(dst_dir) / "видео_тест.avi"
    dst.write_bytes(ascii_path.read_bytes())
    return dst


def test_video(root):
    vdir = Path(root) / "Папка видео"
    vdir.mkdir()
    v = make_video(vdir)
    info = video.video_info(v)
    assert info["frames"] == 72 and info["width"] == 160 and info["height"] == 90, info
    assert abs(info["fps"] - FPS) < 0.01 and abs(info["duration_sec"] - 7.2) < 0.1, info
    try:
        video.video_info(vdir / "нет.avi")
        assert False
    except ValueError:
        pass
    bad = vdir / "битый.avi"
    bad.write_bytes(b"not a video")
    try:
        video.video_info(bad)
        assert False
    except ValueError:
        pass

    out = Path(root) / "Кадры"  # unicode output dir
    r = video.extract_frames(v, out, every_n=10)
    assert len(r) == 8, len(r)  # 0,10,...,70
    assert r[0].name == "видео_тест_000000.jpg" and r[1].name == "видео_тест_000010.jpg", r[:2]
    assert imgio.imread(r[0]).shape == (90, 160, 3)

    r = video.extract_frames(v, Path(root) / "s", every_sec=1.0)
    assert len(r) == 8, len(r)
    r = video.extract_frames(v, Path(root) / "s2")  # default 1 s
    assert len(r) == 8
    r = video.extract_frames(v, Path(root) / "s3", every_sec=2.0, start_sec=1.0, end_sec=6.0, prefix="a b")
    assert [p.name for p in r] == ["a_b_000010.jpg", "a_b_000030.jpg", "a_b_000050.jpg"], [p.name for p in r]

    r = video.extract_frames(v, Path(root) / "sc", scene_threshold=0.1)
    assert len(r) == 3, [p.name for p in r]
    assert [p.stem.split("_")[-1] for p in r] == ["000000", "000024", "000048"], r

    r = video.extract_frames(v, Path(root) / "mx", every_n=1, max_frames=5)
    assert len(r) == 5

    calls = []

    def prog(d, t):
        calls.append((d, t))
        return d < 20

    r = video.extract_frames(v, Path(root) / "cx", every_n=1, progress=prog)
    assert 15 <= len(r) <= 25 and len(r) == len(list((Path(root) / "cx").glob("*.jpg"))), len(r)
    assert calls[-1][1] == 72

    try:
        video.extract_frames(v, Path(root) / "x", every_n=2, every_sec=1)
        assert False
    except ValueError:
        pass


def _textured(seed, w=240, h=180):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (9, 12, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def test_dedupe(root):
    d = Path(root) / "Картинки"
    d.mkdir()
    base = {i: _textured(i) for i in range(6)}
    paths = {}
    for i, im in base.items():
        p = d / f"orig{i}x.jpg"
        imgio.imwrite(p, im, 95)
        paths[f"o{i}"] = str(p)
    # near-duplicates of 0, 1 and 2
    rng = np.random.default_rng(99)
    noisy = np.clip(base[0].astype(int) + rng.normal(0, 3, base[0].shape), 0, 255).astype(np.uint8)
    imgio.imwrite(d / "n0.jpg", noisy, 90)
    imgio.imwrite(d / "b1.png", np.clip(base[1].astype(int) + 12, 0, 255).astype(np.uint8))
    imgio.imwrite(d / "r2.jpg", cv2.resize(base[2], (120, 90), interpolation=cv2.INTER_AREA), 60)
    imgio.imwrite(d / "j2.jpg", base[2], 30)
    allp = sorted(str(p) for p in d.iterdir())
    groups = dedupe.find_near_duplicates(allp)
    gs = sorted(sorted(Path(x).name for x in g) for g in groups)
    assert gs == [["b1.png", "orig1x.jpg"], ["j2.jpg", "orig2x.jpg", "r2.jpg"], ["n0.jpg", "orig0x.jpg"]], gs

    assert dedupe.hamming(0b1011, 0b0001) == 2
    assert 0 <= dedupe.dhash(base[0]) < 2 ** 64

    # source_key
    k = dedupe.source_key
    assert k("/x/v/clip_000123.jpg") == k("/x/v/clip-123.png") == "v/clip"
    assert k("/x/v/frame_000045.jpg") == "v/frame"
    assert k("/x/a/video1_f0003.jpg") == "a/video1" != k("/x/a/video2_f0003.jpg")
    assert k("/x/a/IMG_0001.jpg") == k("/x/a/IMG_0002.jpg") != k("/x/b/IMG_0001.jpg")
    assert k("/x/a/cat.jpg") == "a/cat"

    # build_groups: names + hashes
    extra = [str(d / "orig3x.jpg")]
    seq = []
    for i in range(3):
        p = d / f"clip_{i:04d}.jpg"
        imgio.imwrite(p, _textured(100 + i), 90)
        seq.append(str(p))
    ps = allp + seq
    g = dedupe.build_groups(ps)
    assert g[str(d / "n0.jpg")] == g[str(d / "orig0x.jpg")]
    assert len({g[s] for s in seq}) == 1
    assert g[extra[0]] not in {g[s] for s in seq} and g[extra[0]] != g[str(d / "orig4x.jpg")]
    g2 = dedupe.build_groups(ps, use_hash=False, use_names=False)
    assert len(set(g2.values())) == len(ps)
    g3 = dedupe.build_groups(ps, use_hash=False)
    assert len({g3[s] for s in seq}) == 1 and g3[str(d / "n0.jpg")] != g3[str(d / "orig0x.jpg")]

    # cancel
    assert dedupe.find_near_duplicates(allp, progress=lambda a, b: False) == []


def test_split():
    items = [(i, f"g{i // 3 if i < 150 else i}") for i in range(200)]  # 50 groups of 3 + 50 singles
    fr = {"train": 0.8, "val": 0.1, "test": 0.1}
    r = dedupe.split_groups(items, lambda it: it[1], fr, seed=5)
    where = {}
    for s, v in r.items():
        for it in v:
            assert it[1] not in where or where[it[1]] == s, "group split!"
            where[it[1]] = s
    assert sum(len(v) for v in r.values()) == 200
    assert abs(len(r["train"]) - 160) <= 3 and abs(len(r["val"]) - 20) <= 3 and abs(len(r["test"]) - 20) <= 3, \
        {k: len(v) for k, v in r.items()}
    assert r == dedupe.split_groups(items, lambda it: it[1], fr, seed=5)
    assert r != dedupe.split_groups(items, lambda it: it[1], fr, seed=6)
    r2 = dedupe.split_groups(items, lambda it: it[1], {"train": 0.8, "val": 0.2, "test": 0.0}, 1)
    assert set(r2) == {"train", "val"}
    assert dedupe.split_groups([], lambda x: x, fr, 0) == {}


def test_speed():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2 ** 63, 20000, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, 20000, dtype=np.uint64)
    h[5000:5010] = h[5000]
    h[7000] = h[100] ^ np.uint64(0b101)
    t = time.perf_counter()
    lab = dedupe.cluster_hashes(h, 6)
    dt = time.perf_counter() - t
    print(f"cluster_hashes 20000 hashes: {dt:.2f}s")
    assert lab[7000] == lab[100] and len(set(lab[5000:5010])) == 1 and lab[0] != lab[1]
    assert dt < 15


def main():
    with tempfile.TemporaryDirectory() as r1:
        test_video(r1)
        print("video ok")
    with tempfile.TemporaryDirectory() as r2:
        test_dedupe(r2)
        print("dedupe ok")
    test_split()
    test_speed()
    # hashing throughput on files
    with tempfile.TemporaryDirectory() as r3:
        ps = []
        for i in range(300):
            p = Path(r3) / f"i{i}.jpg"
            imgio.imwrite(p, _textured(i, 1280, 960), 90)
            ps.append(str(p))
        t = time.perf_counter()
        dedupe.hash_files(ps)
        print(f"hash 300 files 1280x960: {time.perf_counter() - t:.2f}s")
    print("ALL OK")


if __name__ == "__main__":
    main()
