import argparse
import logging

from mmpose.apis import MMPoseInferencer


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    inferencer = MMPoseInferencer(pose3d="human3d")

    results = inferencer(
        args.video_path, show=False, draw_bbox=True, out_dir=args.output_path
    )
    results = [result for result in results]

    return results


if __name__ == "__main__":
    main()
