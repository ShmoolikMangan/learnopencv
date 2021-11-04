import sys
from argparse import ArgumentParser

import cv2

from algorithms.dense_optical_flow import dense_optical_flow
from algorithms.lucas_kanade import lucas_kanade_method


def main(inpArg=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        default="farneback",
        choices=["farneback", "lucaskanade", "lucaskanade_dense", "rlof"],
        # required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default="videos/people.mp4",
        help="Path to the video",
    )

    args = parser.parse_args(inpArg)
    video_path = args.video_path
    if args.algorithm == "lucaskanade":
        lucas_kanade_method(video_path)
    elif args.algorithm == "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dense_optical_flow(method, video_path, to_gray=True)
    elif args.algorithm == "farneback":
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
        params = [
                    0.5,    #0.5,   double  pyr_scale,
                    3,      #3,     int     levels,
                    9,     #15,    int     winsize,
                    1,      #3,     int     iterations,
                    5,      #5,     int     poly_n,
                    1.2,    #1.2,   double  poly_sigma,
                    0,      #0,     int     flags
                  ]
        dense_optical_flow(method, video_path, params, to_gray=True)
    elif args.algorithm == "rlof":
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dense_optical_flow(method, video_path)


if __name__ == "__main__":
    movPath = r"C:/Shmoolik/Data/Slow_motion_960/VID_20190528_182440.mp4"
    #
    # movPath = "videos/people.mp4"
    algp = 'farneback'  # ["farneback", "lucaskanade", "lucaskanade_dense", "rlof"],
    inpArg = [f'--video_path', f'{movPath}', f'--algorithm', f'{algp}']
    # sys.argv.append(f'--algorithm farneback')
    # sys.argv.append(f'--video_path {movPath}')
    main(inpArg)
