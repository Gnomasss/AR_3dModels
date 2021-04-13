import cv2
from cv2 import aruco
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()
    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    frame = get_frame(cap, scaling_factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(corners)

    if corners:
        '''for i in corners[0][0]:
            cv2.circle(frame, (i[0], i[1]), 1, (255, 0, 0), thickness=10)'''
        cv2.circle(frame, (corners[0][0][0][0], corners[0][0][0][1]), 1, (255, 0, 0), thickness=10)
        cv2.circle(frame, (corners[0][0][1][0], corners[0][0][1][1]), 1, (0, 255, 0), thickness=10)
        cv2.circle(frame, (corners[0][0][2][0], corners[0][0][2][1]), 1, (0, 0, 255), thickness=10)


    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    cv2.imshow("res", frame_markers)


    if cv2.waitKey(1) == 27:

        break

cv2.destroyAllWindows()