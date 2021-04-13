import cv2
import numpy as np
from cv2 import aruco

dic = aruco.getPredefinedDictionary(aruco.DICT_8X8_64)

for i in range(5):
    marker = np.zeros((200, 200), dtype=np.uint8)
    marker = cv2.aruco.drawMarker(dic, i, 200, marker, 1)

    cv2.imwrite("marker_8x8_"+ str(i) + ".png", marker)
