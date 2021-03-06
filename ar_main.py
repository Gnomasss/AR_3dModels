




import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *
from cv2 import aruco

from createProjectionMatrix import ProjMatr

Mat = ProjMatr()


MIN_MATCHES = 10
DEFAULT_COLOR = (0, 0, 0)
flFirst = True
calibration_path = "realsense_d435.npz"
with np.load(calibration_path) as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
print(mtx)
def main():
    global flFirst
    """
    This functions loads the target surface image,
    """
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array(mtx)

    dir_name = os.getcwd()

    obj2 = OBJ(os.path.join(dir_name, 'models1/rat.obj'), swapyz=True)
    obj = OBJ(os.path.join(dir_name, 'models1/fox.obj'), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        frame2 = np.copy(frame)
        if not ret:
            print("Unable to capture video")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        #res = aruco.detectMarkers(gray, aruco.getPredefinedDictionary(aruco.DICT_6X6_250))
        '''for i in range(len(res[0])):
            print(res[0][i])
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
            #print(i, rvecs, tvecs)
            #sm = moveModel(frame, (135, 150, 110), corners)
            sm = moveModel2(frame, (135, 156, 118), corners, tvecs, rvecs)
        #frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)'''

        if corners:
            '''for i in range(len(ids)):
                rvecs, tvecs, _ 1.27= aruco.estimatePoseSingleMarkers(np.array([corners[0][i]]), 1, mtx, dist)
                #print('----------------------')
                H = mtx @ (rvecs[0].T @ rvecs[0])
                #print(mtx @ (tvecs[0].T @ rvecs[0]))
                #print(rvecs, tvecs, sep='\n')
                v = glyp(rvecs, tvecs)
                #print(v)'''


            src_pts = np.array([[0,0], [0, 200], [200, 200], [200, 0]]).reshape(-1,1,2)
            dst_pts = []
            avCorn = [[0] * 2 for i in range(4)]
            k = 0
            #print(corners[0][0])
            for i in range(len(corners)):
                k += 1
                for j in range(4):
                    avCorn[j][0] += corners[i][0][j][0]
                    avCorn[j][1] += corners[i][0][j][1]
                    #cv2.circle(frame, (corners[i][0][j][0], corners[i][0][j][1]), 1, (255 * (j % 3 == 0),255 * (j % 2 == 0),255), 10)
            for i in range(len(avCorn)):
                avCorn[i][0] = avCorn[i][0] // k
                avCorn[i][1] = avCorn[i][1] // k

            '''for i in avCorn:
                
                cv2.circle(frame, (i[0], i[1]), 1, (255, 0, 125), 10)'''
            '''for i in corners[0][0]:
                dst_pts.append([i[0] + sm[0], i[1] + sm[1]])'''
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(np.array([avCorn]), 1, mtx, dist)

            rvecsMat, _ = cv2.Rodrigues(rvecs)
            print('-----------')
            print(rvecs, tvecs)
            print(rvecsMat)
            for i in range(len(avCorn)):
                avCorn[i][0] = int(avCorn[i][0])
                avCorn[i][1] = int(avCorn[i][1])

            if flFirst:
                avCornOld = avCorn.copy()
                flFirst = False
            else:
                for i in range(len(avCorn)):
                    for j in range(2):
                        if abs(avCorn[i][j] - avCornOld[i][j]) <= 5:
                            avCorn[i][j] = avCornOld[i][j]
                avCornOld = avCorn.copy()

            sm = moveModel2(frame, (135, 156, 118), np.array(avCorn), tvecs, rvecs)
            #print(np.array([avCorn]))
            #print('-----------------')

            for i in avCorn:
                dst_pts.append([i[0] + sm[0], i[1] + sm[1]])
            dst_pts = np.array(dst_pts).reshape(-1,1,2)
            homography, mask = cv2.findHomography(src_pts, dst_pts)
            #print(homography)
            R_T = get_extended_RT(camera_parameters, homography)
            #print('aaaa',R_T)
            transformation = camera_parameters.dot(R_T)
            print(R_T)
            frame = render(frame, obj, transformation, False)



        #cv2.imshow('frame2', frame2)
        #cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    return 0




def get_extended_RT(A, H):
    # finds r3 and appends
    # A is the intrinsic mat, and H is the homography estimated
    H = np.float64(H)  # for better precision
    A = np.float64(A)
    R_12_T = np.linalg.inv(A).dot(H)
    #print(R_12_T)
    r1 = np.float64(R_12_T[:, 0])  # col1
    r2 = np.float64(R_12_T[:, 1])  # col2
    T = R_12_T[:, 2]  # translation
    #print(T)
    # ideally |r1| and |r2| should be same
    # since there is always some error we take square_root(|r1||r2|) as the normalization factor
    norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))

    r3 = np.cross(r1, r2) / (norm)
    #print(r1, r2, r3)
    R_T = np.zeros((3, 4))
    R_T[:, 0] = r1
    R_T[:, 1] = r2
    R_T[:, 2] = r3
    R_T[:, 3] = T
    return R_T

def moveModel(frame, color, corners):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([i - 50 for i in color])
    up = np.array([i + 50 for i in color])
    mask = cv2.inRange(hsv, low, up)
    edged, cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edged = sorted(edged, key=cv2.contourArea, reverse=True)

    if len(edged) > 0 and len(corners) > 0:
        rect = cv2.minAreaRect(edged[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box.T
        x0 = sum(box[0]) // 4
        y0 = sum(box[1]) // 4
        cv2.circle(frame, (x0, y0), 10, (255, 0, 0), 10)
        x1 = int(sum(corners[0][0][:, 0]) // 4)
        y1 = int(sum(corners[0][0][:, 1]) // 4)
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), 10)
        ny = y0 - y1
        nx = x0 - x1
    else:
        ny, nx = 0, 0
    return (nx, ny)

def moveModel2(frame, color, corners, tvecs, rvecs):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #low = np.array([i - 50 for i in color])
    #up = np.array([i + 50 for i in color])
    low = np.array([51, 88, 77])
    up = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, low, up)
    cv2.imshow('mask', mask)
    h, w = frame.shape[:2]
    edged, cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edged = sorted(edged, key=cv2.contourArea, reverse=True)
    points = Mat.ProjectPoints(frame, 10, 50, tvecs, rvecs, mtx, dist)
    #print('tut1')
    if len(edged) > 0 and len(corners) > 0:
        rect = cv2.minAreaRect(edged[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box.T
        xp = sum(box[0]) // 4
        yp = sum(box[1]) // 4
        mn = int(100000000)
        x0 = 0
        y0 = 0
        #print(len(points))
        for i in points:
            #print(i)
            if (i[0] - xp) ** 2 + (i[1] - yp) ** 2 < mn:
                mn = (i[0] - xp) ** 2 + (i[1] - yp) ** 2
                x0 = i[0]
                y0 = i[1]
        #print('tut2', x0, y0, w, h)
        if abs(x0) < w and abs(y0) < h and x0 > -200000:
            #print('wtf', abs(x0) < w, abs(y0) < h, abs(x0))
            cv2.circle(frame, (x0, y0), 10, (0, 255, 0), 10)
            x1 = int(sum(corners[:, 0]) // 4)
            y1 = int(sum(corners[:, 1]) // 4)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), 10)
            ny = y0 - y1
            nx = x0 - x1
        else:
            ny = 0
            nx = 0
        #print(nx, ny)
    else:
        ny, nx = 0, 0
    return (nx, ny)

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
        #points = np.array([[p[0] + sm[0],  p[1] + sm[1], p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
