"""Run the preprocessor over a video file.

Usage:
    python examples/from_video_file.py path/to/clip.mp4

Prints the shape of each motion image the moment it is produced.

Reading video is the caller's job: the package takes any iterable of
``uint8`` BGR frames and never opens a file or camera itself. ``frames_from``
below is the few lines that bridge the two. Keep its ``try``/``finally``:
without it the capture handle leaks whenever the loop is left early.
"""

import argparse
import sys

import cv2

from amprep import AdaptiveMotionPreprocessor


def frames_from(path):
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="video file to read")
    args = parser.parse_args()

    preprocessor = AdaptiveMotionPreprocessor()
    try:
        for number, image in enumerate(preprocessor.process(frames_from(args.path))):
            print(f"motion image {number}: shape {image.data.shape}", flush=True)
    except OSError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
