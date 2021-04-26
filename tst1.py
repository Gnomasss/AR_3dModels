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

print("Press [ESC] to close the application")
while True:
    # Get frame from realsense and convert to grayscale image
    frames = get_frame(cap, scaling_factor)
    img_rgb = frames
    img_gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)

    # Detect markers on the gray image
    res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(dictionary))
    # Draw each marker
    print(len(res[0]))
    for i in range(len(res[0])):
        print(res[0][i])
        # Estimate pose of the respective marker, with matrix size 1x1
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
        # Define the ar cube
        # Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is also defined with a size of 1x1x1
        # It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin
        axis = np.float32([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0],
                           [-0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1], [0.5, -0.5, 1]])
        '''axis = np.float32([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0],
                           [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])'''
        # Now we transform the cube to the marker position and project the resulting points into 2d
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        print(jac)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        print(imgpts)
        # Now comes the drawing.
        # In this example, I would like to draw the cube so that the walls also get a painted
        # First create six copies of the original picture (for each side of the cube one)
        side1 = img_rgb.copy()
        side2 = img_rgb.copy()
        side3 = img_rgb.copy()
        side4 = img_rgb.copy()
        side5 = img_rgb.copy()
        side6 = img_rgb.copy()

        # Draw the bottom side (over the marker)
        side1 = cv2.drawContours(side1, [imgpts[:4]], -1, (255, 0, 0), -2)
        # Draw the top side (opposite of the marker)
        side2 = cv2.drawContours(side2, [imgpts[4:]], -1, (255, 0, 0), -2)
        # Draw the right side vertical to the marker
        side3 = cv2.drawContours(side3, [np.array(
            [imgpts[0], imgpts[1], imgpts[5],
             imgpts[4]])], -1, (255, 0, 0), -2)
        # Draw the left side vertical to the marker
        side4 = cv2.drawContours(side4, [np.array(
            [imgpts[2], imgpts[3], imgpts[7],
             imgpts[6]])], -1, (255, 0, 0), -2)
        # Draw the front side vertical to the marker
        side5 = cv2.drawContours(side5, [np.array(
            [imgpts[1], imgpts[2], imgpts[6],
             imgpts[5]])], -1, (255, 0, 0), -2)
        # Draw the back side vertical to the marker
        side6 = cv2.drawContours(side6, [np.array(
            [imgpts[0], imgpts[3], imgpts[7],
             imgpts[4]])], -1, (255, 0, 0), -2)

        # Until here the walls of the cube are drawn in and can be merged
        img_rgb = cv2.addWeighted(side1, 0.1, img_rgb, 0.9, 0)
        img_rgb = cv2.addWeighted(side2, 0.1, img_rgb, 0.9, 0)
        img_rgb = cv2.addWeighted(side3, 0.1, img_rgb, 0.9, 0)
        img_rgb = cv2.addWeighted(side4, 0.1, img_rgb, 0.9, 0)
        img_rgb = cv2.addWeighted(side5, 0.1, img_rgb, 0.9, 0)
        img_rgb = cv2.addWeighted(side6, 0.1, img_rgb, 0.9, 0)

        # Now the edges of the cube are drawn thicker and stronger
        img_rgb = cv2.drawContours(img_rgb, [imgpts[:4]], -1, (255, 0, 0), 2)
        for i, j in zip(range(4), range(4, 8)):
            img_rgb = cv2.line(img_rgb, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
        img_rgb = cv2.drawContours(img_rgb, [imgpts[4:]], -1, (255, 0, 0), 2)

    # Display the result
    cv2.imshow("AR-Example", img_rgb)

    # If [ESC] pressed, close the application
    if cv2.waitKey(100) == 27:
        print("Application closed")
        break
# Close all cv2 windows
cv2.destroyAllWindows()