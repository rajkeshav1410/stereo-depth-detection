import cv2
import json
import glob
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

def individual_calibrate(path):
	# global imgLeft, imgRight, obj_pts, img_ptsL, img_ptsR, new_mtxL, new_mtxR, distL, distR

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((9*6,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
	img_ptsL, img_ptsR, obj_pts = [], [], []
	n_file = len(glob.glob(f'{path}/*.png')) >> 1

	for i in range(n_file):
		imgLeft = cv2.imread(f'{path}/left_{(i + 1):02}.png', 0)
		imgRight = cv2.imread(f'{path}/right_{(i + 1):02}.png', 0)
		retL, cornersL = cv2.findChessboardCorners(imgLeft, (9, 6), None)
		retR = None
		if retL:
			retR, cornersR = cv2.findChessboardCorners(imgRight, (9, 6), None)

		if retR and retL:
			obj_pts.append(objp)
			cv2.cornerSubPix(imgLeft, cornersL, (11, 11), (-1, -1), criteria)
			cv2.cornerSubPix(imgRight, cornersR, (11, 11), (-1, -1), criteria)
			img_ptsL.append(cornersL)
			img_ptsR.append(cornersR)
			cv2.drawChessboardCorners(imgLeft, (9, 6), cornersL, retL)
			cv2.drawChessboardCorners(imgRight, (9, 6), cornersR, retR)
			cv2.imshow("left", imgLeft)
			cv2.imshow("right", imgRight)
		cv2.waitKey(0)
	retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgLeft.shape[::-1], None, None)
	retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgRight.shape[::-1], None, None)
	hL, wL = imgLeft.shape[:2]
	hR, wR = imgRight.shape[:2]
	print(hL, wL, hR, wR)
	print(f'Camera matrix L:\n{mtxL}')
	print(f'Camera matrix R:\n{mtxR}')
	print(f'Distortion matrix')
	new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))
	new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

# def stereo_calibrate():
# 	global retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat

	flags = 0
	flags |= cv2.CALIB_FIX_INTRINSIC
	criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgLeft.shape[::-1], criteria_stereo, flags)

# def stereo_rectify():
# 	global rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR

	rectify_scale = 1
	rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR  = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgLeft.shape[::-1], Rot, Trns, rectify_scale, (0, 0))

# def undistort():
	Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, imgLeft.shape[::-1], cv2.CV_16SC2)
	Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, imgRight.shape[::-1], cv2.CV_16SC2)

	print("Saving paraeters ......")
	cv_file = cv2.FileStorage("params.xml", cv2.FILE_STORAGE_WRITE)
	cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
	cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
	cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
	cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
	cv_file.release()

def calibrate(path):
	global image_size
	n_file = len(glob.glob(f'{path}/*.png')) >> 1
	img = cv2.imread(f'{path}/left_01.png', 0)
	image_size = (img.shape[1], img.shape[0])
	calibrator = StereoCalibrator(rows, cols, square_size, image_size)

	for i in range(n_file):
		imgLeft = cv2.imread(f'{path}/left_{(i + 1):02}.png', 1)
		imgRight = cv2.imread(f'{path}/right_{(i + 1):02}.png', 1)
		try:
			calibrator._get_corners(imgLeft)
			calibrator._get_corners(imgRight)
		except ChessboardNotFoundError as error:
			print(error)
			print("Pair No "+ str(i + 1) + " ignored")
		else:
			calibrator.add_corners((imgLeft, imgRight), True)
	
	print ('Starting calibration... It can take several minutes!')
	calibration = calibrator.calibrate_cameras()
	calibration.export('calib_res')
	print ('Calibration complete!')

if __name__ == '__main__':
	global rows, cols, square_size
	rows, cols, square_size = 6, 9, 1
	calibrate('./n_pairs')
	# individual_calibrate('./n_pairs')
	# stereo_calibrate()
	# stereo_rectify()
	# undistort()