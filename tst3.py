import cv2
from cv2 import aruco
from objloader_simple import *
import numpy as np
import os

calibration_path = "realsense_d435.npz"
dictionary = aruco.DICT_6X6_250
DEFAULT_COLOR = (0, 0, 0)
mtx = None
dist = None
with np.load(calibration_path) as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
camera_parameters = np.array(mtx)
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()
    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = 200, 200

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img



# Check communication
cap = cv2.VideoCapture(1)
scaling_factor = 1
dir_name = os.getcwd()
obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
while True:
    frames = get_frame(cap, scaling_factor)
    frames2 = frames.copy()
    img_rgb = frames
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    imsize = frames.shape[:1]
    res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(dictionary))

    for i in range(len(res[0])):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
        R_T, _ = cv2.Rodrigues(rvecs)
        #print(tvecs[0])
        #trans2 = glyp(rvecs, tvecs)
        #trans2 = trans2.T
        #trans2 = trans2[:3]
        #print(trans2)
        #frames2 = render(frames, obj, trans2, False)
        R_T = np.append(R_T, tvecs[0], axis=0)
        R_T = R_T.T
        print(R_T)
        #R_T = R_T * 4
        transformation = camera_parameters.dot(R_T)
        frames = render(frames, obj, transformation, False)
    #frame_markers = aruco.drawDetectedMarkers(frames.copy(), res[0], res[1])
    cv2.imshow("res", frames)
    #cv2.imshow('res2', frames2)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
