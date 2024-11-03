import json
from typing import Dict, List

import numpy as np

# ref: https://github.com/robertklee/COCO-Human-Pose/blob/main/README.md
KEYPOINT_TO_DESCRIPTION_MAP = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}


def load_poses(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_keypoints_from_pose(pose: dict):
    assert len(pose["instances"]) == 1, "Expected to receive a single instance"
    return np.array(pose["instances"][0]["keypoints"])


def get_keypoints(poses: List[Dict]):
    return np.stack([get_keypoints_from_pose(p) for p in poses])
