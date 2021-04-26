import numpy as np
import cv2

class ProjMatr:

    def createMatrix(self, step, count):
        a = []
        for i in range(-count, count, step):
            for j in range(-count, count, step):
                a.append([i / 10, j / 10, 0])
        return np.float64([a])

    def ProjectPoints(self, frame, step, count, tvecs, rvecs, mtx, dist):
        axis = self.createMatrix(step, count)

        h, w = frame.shape[:2]

        tvecs[0][0][0] = 0
        tvecs[0][0][1] = 0
        mtx[0][2] = w // 2
        mtx[1][2] = h // 2

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        imgpts = np.int32(imgpts).reshape(-1, 2)
        for j in imgpts:
            cv2.circle(frame, (j[0], j[1]), 1, (255, 0, 0), 10)

        return imgpts





