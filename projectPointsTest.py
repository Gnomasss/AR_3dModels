import cv2
import cv2.aruco as aruco
import numpy as np


# Defines the path of the calibration file and the dictonary used
calibration_path = "realsense_d435.npz"
dictionary = aruco.DICT_6X6_250  # aruco.DICT_6X6_250

# Load calibration from file
mtx = None
dist = None
with np.load(calibration_path) as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Initialize communication with intel realsense
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()
    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

# Check communication
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

def createMAtrix(step, count):
    a = []
    for i in range(-count, count, step):
        for j in range(-count, count, step):
            a.append([i / 10, j / 10, 0])
    return np.float64([a])

print("Press [ESC] to close the application")
while True:
    # Get frame from realsense and convert to grayscale image
    frames = get_frame(cap, scaling_factor)
    img_rgb = frames
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    h, w = frames.shape[:2]
    # Detect markers on the gray image
    res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(dictionary))
    # Draw each marker
    #print(len(res[0]))
    for i in range(len(res[0])):
        #print(res[0][i])
        # Estimate pose of the respective marker, with matrix size 1x1
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
        # Define the ar cube
        # Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is also defined with a size of 1x1x1
        # It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin
        axis = np.float32([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0],
                           [-0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1], [0.5, -0.5, 1]])
        ''''axis = np.float32([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0],
                           [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])'''

        axis2 = createMAtrix(4, 50)

        '''for x in range(0, w, 2):
            for y in range(0, h, 2):
                axis2.append([x, y, 0])
        axis2 = np.float32([axis2])'''

        #print(tvecs)
        #print(mtx)
        #cv2.circle(img_rgb, (130, 188), 10, (0,0,255), 10)
        tvecs[0][0][0] = 0
        tvecs[0][0][1] = 0
        mtx[0][2] = w // 2
        mtx[1][2] = h // 2
        '''for j in range(len(axis2)):
            axis2[j] = rvecs[0][0] @ axis2[j]
        for j in axis2:
            cv2.circle(img_rgb, (j[0], j[1]), 1, (0, 255, 0), 10)'''
        # Now we transform the cube to the marker position and project the resulting points into 2d
        imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)
        #print(jac)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        #print(imgpts)
        print(imgpts)
        for j in imgpts:
            cv2.circle(img_rgb, (j[0], j[1]), 1, (255, 0, 0), 10)
            #print(j)



    cv2.imshow("AR-Example", img_rgb)

    # If [ESC] pressed, close the application
    if cv2.waitKey(1) == 27:
        print("Application closed")
        break
# Close all cv2 windows
cv2.destroyAllWindows()