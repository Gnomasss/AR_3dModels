import cv2
import numpy as np
def nothing(*arg):
    pass

cv2.namedWindow( "settings" )
cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    lower2 = np.array((h1, s1, v1), np.uint8)
    upper2 = np.array((h2, s2, v2), np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower2, upper2)

    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        print(h1, s1, v1)
        print(h2, s2, v2)
        print('---------------')
    if key == 27:
        break

cv2.destroyAllWindows()
