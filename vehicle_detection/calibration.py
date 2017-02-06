import numpy as np
import glob
import cv2
import pickle

calibration = None
path = "camera_cal/calibration.p"
try:
    calibration = pickle.load(open(path, "rb"))
except(Exception) as e:
    pass

def calibrate(images):

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # not able to find all corners for all images
    # try finding as many corners as possible
    sizes = [(9,6), (8,6), (9,5), (7,6)]

    # prepare object points for varying sizes, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objps = [np.zeros((size[0] * size[1],3), np.float32) for size in sizes]
    for idx, objp in enumerate(objps):
        size = sizes[idx]
        objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2)
        objps[idx] = objp

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        for objp_idx, objp in enumerate(objps):
            size = sizes[objp_idx]

            ret, corners = cv2.findChessboardCorners(gray, size, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                break

    img_size = (gray.shape[1], gray.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save Calibration Results
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(path, "wb"))

def get_calibration():
    assert calibration is not None, 'Cannot undistort.  Run camera calibration, and try again'
    return calibration

def undistort(image):
    cal = get_calibration()
    return cv2.undistort(image, cal.get('mtx'), cal.get('dist'))
