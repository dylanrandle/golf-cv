import argparse
import logging

from mmpose.apis import MMPoseInferencer


logger = logging.getLogger(__name__)


MODES = {"2d", "3d"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, choices=MODES)
    return parser.parse_args()


def main():
    args = parse_args()

    inference_mode = args.mode.strip().lower()

    if inference_mode == "3d":
        inferencer = MMPoseInferencer(pose3d="human3d")
    elif inference_mode == "2d":
        inferencer = MMPoseInferencer(pose2d="human")
    else:
        raise ValueError(
            f"Unrecognized inference mode: {inference_mode} (valid types are: {MODES})"
        )

    results = inferencer(
        args.video_path,
        show=False,
        draw_bbox=True,
        out_dir=args.output_path,
        radius=6,
        thickness=2,
        skeleton_style="openpose",
    )
    results = [result for result in results]

    return results


if __name__ == "__main__":
    main()
