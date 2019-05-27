

import numpy as np
import cv2 as cv
import cv2.aruco as aruco


def center(image):

    # Camera parameters
    calibrationFile = "logitech9000.yml"
    calibrationParams = cv.FileStorage(calibrationFile, cv.FILE_STORAGE_READ)
    camera_matrix = calibrationParams.getNode("camera_matrix").mat()
    dist_coeffs = calibrationParams.getNode("distortion_coefficients").mat()
    frame = image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters =  aruco.DetectorParameters_create()
        
        # Marker detection
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids != None: # ce je aruco marker zaznan
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.10, camera_matrix, dist_coeffs)
    else: # If marker is not detected
        arucoFrame = frame
        rvecs = [[[0, 0, 0]]]
        tvecs = [[[0, 0, 0]]]
        ids = [[3,3]]
        
    x1 = corners[0][0][0][0]
    x2 = corners[0][0][1][0]
    x3 = corners[0][0][2][0]
    x4 = corners[0][0][3][0]
    y1 = corners[0][0][0][1]
    y2 = corners[0][0][1][1]
    y3 = corners[0][0][2][1]
    y4 = corners[0][0][3][1]
    a = (x1+x2)/2
    b= (y1+y4)/2
    center =[a,b]
    x_raz1 = (x2 - x1)/10
    x_raz2 = (x3 - x4)/10
    y_raz1 = (y3 - y1)/10
    y_raz2 = (y4 - y2)/10

    return(center, x_raz1, x_raz2, y_raz1, y_raz2)
