import os
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt

def dense_optical_flow(method, video_path, params=[], to_gray=False):

    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        return

    # some inits
    frmNum = 0
    execTime = 0
    if method ==  cv2.calcOpticalFlowFarneback:
        margins = round(params[2] * params[0] ** -params[1])
    else:
        margins = 16

    # Open stream to write video
    h, w, _ = old_frame.shape
    w -= 2* margins
    h -= 2* margins
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps=5
    pn, fn = os.path.split(video_path)
    outFile = os.path.join(pn, f'Motion_{fn}')
    outVid = cv2.VideoWriter(outFile, fourcc, fps, (w, h))
    # outVid = cv2.VideoWriter(outFile, fourcc, 20.0, old_frame[margins:-margins,margins:-margins].shape())

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        if not ret:
            break
        frame_copy = new_frame[margins:-margins,margins:-margins]
        frmNum += 1

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        t1 = time.time()
        flow = method(old_frame, new_frame, None, *params)
        dt = time.time() - t1
        execTime += dt

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = 0 # 180 / np.pi / 2 # ang * 180 / np.pi / 2
        hsv[..., 1] = 1
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv[margins:-margins,margins:-margins], cv2.COLOR_HSV2BGR)
        # imgplot = plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # imgplot = plt.imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))


        titleStr = f'Frame - {frmNum=}, execTime={execTime / frmNum:.3f}'
        bgr = cv2.putText(bgr, titleStr, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame_copy)
        cv2.setWindowTitle('frame', titleStr)
        cv2.imshow("optical flow", bgr)
        cv2.setWindowTitle('optical flow', titleStr)

        # imgplot = plt.imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        imgplot = plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # write the motion frame
        outVid.write(bgr)

        # Quit if ESC was pressed
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame


    # When everything done, release the video capture object
    cap.release()
    outVid and outVid.release()

    # Closes all the frames
    cv2.destroyAllWindows()
