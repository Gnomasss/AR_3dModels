import cv2
from cv2 import aruco

import numpy as np


calibration_path = "realsense_d435.npz"
dictionary = aruco.DICT_6X6_250

mtx = None
dist = None
with np.load(calibration_path) as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()
    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

# Check communication
cap = cv2.VideoCapture(1)
scaling_factor = 0.5

while True:
    frames = get_frame(cap, scaling_factor)
    img_rgb = frames
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    imsize = frames.shape[:1]
    res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(dictionary))

    for i in range(len(res[0])):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
        cv2.ovis.addResourceLocation("packs/Sinbad.zip")
        # create an Ogre window for rendering
        win = cv2.ovis.createWindow("OgreWindow", imsize, flags=cv2.ovis.WINDOW_SCENE_AA)
        win.setBackground(frames)
        # make Ogre renderings match your camera images
        win.setCameraIntrinsics(mtx, imsize)
        # create the virtual scene, consisting of Sinbad and a light
        win.createEntity("figure", "Sinbad.mesh", tvec=(0, 0, 5), rot=(1.57, 0, 0))
        win.createLightEntity("sun", tvec=(0, 0, 100))
        # position the camera according to the first marker detected
        win.setCameraPose(tvecs[0].ravel(), rvecs[0].ravel(), invert=True)
    cv2.imshow('frame', frames)
    if cv2.waitKey(100) == 27:
        print("Application closed")
        break
cv2.destroyAllWindows()