"""Готовые наборы ключевых точек (названия, скелет, пары «левая-правая»)."""

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
              "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
              "right_ankle"]
_i = {n: k for k, n in enumerate(COCO_NAMES)}
COCO_SKELETON = [[_i[a], _i[b]] for a, b in (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("nose", "left_eye"), ("nose", "right_eye"))]
COCO_FLIP = [[_i[f"left_{p}"], _i[f"right_{p}"]] for p in ("eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle")]


def names_text(names):
    return "\n".join(names)


def pairs_text(pairs, names):
    return "\n".join(f"{names[i]}-{names[j]}" for i, j in pairs if i < len(names) and j < len(names))
