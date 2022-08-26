from email.mime import image
import cv2
import json
import glob
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# def individual_calibrate(path):
# 	global imgLeft, imgRight, obj_pts, img_ptsL, img_ptsR, new_mtxL, new_mtxR, distL, distR

# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 	objp = np.zeros((9*6, 3), np.float32)
# 	objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# 	img_ptsL, img_ptsR, obj_pts = [], [], []
# 	n_file = len(glob.glob(f'{path}/*.png')) >> 1

# 	for i in range(n_file):
# 		imgLeft = cv2.imread(f'{path}/left_{(i + 1):02}.png', 0)
# 		imgRight = cv2.imread(f'{path}/right_{(i + 1):02}.png', 0)
# 		retL, cornersL = cv2.findChessboardCorners(imgLeft, (9, 6), None)
# 		retR = None
# 		if retL:
# 			retR, cornersR = cv2.findChessboardCorners(imgRight, (9, 6), None)

# 		if retR and retL:
# 			obj_pts.append(objp)
# 			cv2.cornerSubPix(imgLeft, cornersL, (11, 11), (-1, -1), criteria)
# 			cv2.cornerSubPix(imgRight, cornersR, (11, 11), (-1, -1), criteria)
# 			img_ptsL.append(cornersL)
# 			img_ptsR.append(cornersR)
# 			cv2.drawChessboardCorners(imgLeft, (9, 6), cornersL, retL)
# 			cv2.drawChessboardCorners(imgRight, (9, 6), cornersR, retR)
# 			cv2.imshow("left", imgLeft)
# 			cv2.imshow("right", imgRight)
# 		cv2.waitKey(0)
# 	retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
# 		obj_pts, img_ptsL, imgLeft.shape[::-1], None, None)
# 	retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
# 		obj_pts, img_ptsR, imgRight.shape[::-1], None, None)
# 	hL, wL = imgLeft.shape[:2]
# 	hR, wR = imgRight.shape[:2]
# 	print(hL, wL, hR, wR)
# 	print(f'Camera matrix L:\n{mtxL}')
# 	print(f'Camera matrix R:\n{mtxR}')
# 	print(f'Distortion matrix')
# 	new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(
# 		mtxL, distL, (wL, hL), 1, (wL, hL))
# 	new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(
# 		mtxR, distR, (wR, hR), 1, (wR, hR))

# def stereo_calibrate():
# 	global retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat

# 	flags = 0
# 	flags |= cv2.CALIB_FIX_INTRINSIC
# 	criteria_stereo = (cv2.TERM_CRITERIA_EPS +
# 					   cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 	retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
# 		obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgLeft.shape[::-1], criteria_stereo, flags)

# def stereo_rectify():
# 	global rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR

# 	rectify_scale = 1
# 	rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
# 		new_mtxL, distL, new_mtxR, distR, imgLeft.shape[::-1], Rot, Trns, rectify_scale, (0, 0))

# # def undistort():
# 	Left_Stereo_Map = cv2.initUndistortRectifyMap(
# 		new_mtxL, distL, rect_l, proj_mat_l, imgLeft.shape[::-1], cv2.CV_16SC2)
# 	Right_Stereo_Map = cv2.initUndistortRectifyMap(
# 		new_mtxR, distR, rect_r, proj_mat_r, imgRight.shape[::-1], cv2.CV_16SC2)

# 	print("Saving paraeters ......")
# 	cv_file = cv2.FileStorage("params.xml", cv2.FILE_STORAGE_WRITE)
# 	cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
# 	cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
# 	cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
# 	cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
# 	cv_file.release()


# def simple_calibrate(path):
# 	global image_size
# 	n_file = len(glob.glob(f'{path}/*.png')) >> 1
# 	img = cv2.imread(f'{path}/left_01.png', 0)
# 	image_size = (img.shape[1], img.shape[0])
# 	calibrator = StereoCalibrator(rows, cols, square_size, image_size)

# 	for i in range(n_file):
# 		imgLeft = cv2.imread(f'{path}/left_{(i + 1):02}.png', 1)
# 		imgRight = cv2.imread(f'{path}/right_{(i + 1):02}.png', 1)
# 		try:
# 			calibrator._get_corners(imgLeft)
# 			calibrator._get_corners(imgRight)
# 		except ChessboardNotFoundError as error:
# 			print(error)
# 			print("Pair No " + str(i + 1) + " ignored")
# 		else:
# 			calibrator.add_corners((imgLeft, imgRight), True)

# 	print('Starting calibration... It can take several minutes!')
# 	calibration = calibrator.calibrate_cameras()
# 	calibration.export('calib_res')
# 	print('Calibration complete!')


criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
criteria_stereo_cal = (cv2.TERM_CRITERIA_EPS +
					   cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


def calibrate_camera(path, size=0.0254, width=9, height=6):
	global img_size
	o_points = np.zeros((height*width, 3), np.float32)
	o_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
	o_points = o_points * size

	obj_points = []
	img_points_l = []
	img_points_r = []

	l_images = glob.glob(f'./{path}/left_*.png')
	r_images = glob.glob(f'./{path}/right_*.png')

	img_size = cv2.imread(l_images[0]).shape[:2]

	l_images.sort()
	r_images.sort()

	for i, _ in enumerate(l_images):
		l_img = cv2.imread(l_images[i])
		r_img = cv2.imread(r_images[i])
		gray_l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
		gray_r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

		l_ret, l_corners = cv2.findChessboardCorners(
			gray_l_img, (width, height), None)
		r_ret, r_corners = cv2.findChessboardCorners(
			gray_r_img, (width, height), None)

		if l_ret is True and r_ret is True:
			obj_points.append(o_points)

			corners_l = cv2.cornerSubPix(
				gray_l_img, l_corners, (11, 11), (-1, -1), criteria_cal)
			img_points_l.append(corners_l)

			l_img = cv2.drawChessboardCorners(
				l_img, (width, height), l_corners, l_ret)

		# if r_ret is True:
			corners_r = cv2.cornerSubPix(
				gray_r_img, r_corners, (11, 11), (-1, -1), criteria_cal)
			img_points_r.append(corners_r)

			r_img = cv2.drawChessboardCorners(
				r_img, (width, height), r_corners, r_ret)
				
	print('lens', len(obj_points), len(img_points_l), len(img_points_r))
	print('calibrating left')
	ret, matrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
		obj_points, img_points_l, gray_l_img.shape[::-1], None, None)

	print('calibrating right')
	ret, matrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
		obj_points, img_points_r, gray_r_img.shape[::-1], None, None)

	return [obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r]


def stereo_calibrate(obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r):
	flag = 0
	flag |= cv2.CALIB_FIX_INTRINSIC
	retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = cv2.stereoCalibrate(
		obj_points, img_points_l, img_points_r, matrix_l, dist_l, matrix_r, dist_r, img_size, criteria=criteria_stereo_cal)
	return [retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F]


def rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T):
	rotation_1, rotation_2, pose_1, pose_2, Q, roi_left, roi_right = cv2.stereoRectify(
		matrix_1, dist_1, matrix_2, dist_2, (9, 6), R, T)
	return [rotation_1, rotation_2, pose_1, pose_2, Q, roi_left, roi_right]


def generate_undistored_rectified_image(img_left, img_right):
	img_left_size = (img_left.shape[1], img_left.shape[0])
	img_right_size = (img_right.shape[1], img_right.shape[0])

	map_w, map_h = cv2.initUndistortRectifyMap(
		matrix_1, dist_1, rot_1, pose_1, img_left_size, cv2.CV_32FC1)
	rectified_left = cv2.remap(img_left, map_w, map_h, cv2.INTER_LINEAR)

	map_w, map_h = cv2.initUndistortRectifyMap(
		matrix_1, dist_1, rot_1, pose_1, img_right_size, cv2.CV_32FC1)
	rectified_right = cv2.remap(img_right, map_w, map_h, cv2.INTER_LINEAR)

	return rectified_left, rectified_right


def save_parameters(matrix_l, matrix_1, dist_l, dist_1, rot_1, pose_1, matrix_r, matrix_2, dist_r, dist_2, rot_2, pose_2):
	file = cv2.FileStorage('./parameters.yml', cv2.FILE_STORAGE_WRITE)
	file.write('matrix_l', matrix_l)
	file.write('matrix_1', matrix_1)
	file.write('dist_l', dist_l)
	file.write('dist_1', dist_1)
	file.write('rot_1', rot_1)
	file.write('pose_1', pose_1)
	file.write('matrix_r', matrix_r)
	file.write('matrix_2', matrix_2)
	file.write('dist_r', dist_r)
	file.write('dist_2', dist_2)
	file.write('rot_2', rot_2)
	file.write('pose_2', pose_2)
	file.release()


def load_parameters():
	global matrix_l, matrix_1, dist_l, dist_1, rot_1, pose_1, matrix_r, matrix_2, dist_r, dist_2, rot_2, pose_2
	file = cv2.FileStorage('./parameters.yml', cv2.FILE_STORAGE_READ)
	matrix_l = file.getNode('matrix_l').mat()
	matrix_1 = file.getNode('matrix_1').mat()
	dist_l = file.getNode('dist_l').mat()
	dist_1 = file.getNode('dist_1').mat()
	rot_1 = file.getNode('rot_1').mat()
	pose_1 = file.getNode('pose_1').mat()
	matrix_r = file.getNode('matrix_r').mat()
	matrix_2 = file.getNode('matrix_2').mat()
	dist_r = file.getNode('dist_r').mat()
	dist_2 = file.getNode('dist_2').mat()
	rot_2 = file.getNode('rot_2').mat()
	pose_2 = file.getNode('pose_2').mat()
	file.release()


def caliberate():
	global img_size
	obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r = calibrate_camera(
		'n_pairs')
	print('Camera calibrated')
	retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = stereo_calibrate(
		obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r)
	print('Stereo calibration done')
	rot_1, rot_2, pose_1, pose_2, Q, roi_left, roi_right = rectify_stereo_camera(
		matrix_1, dist_1, matrix_2, dist_2, R, T)
	print('Stereo camera rectified')
	save_parameters(matrix_l, matrix_1, dist_l, dist_1, rot_1,
				 pose_1, matrix_r, matrix_2, dist_r, dist_2, rot_2, pose_2)
	print('Camera Parameters saved')


if __name__ == '__main__':
	# global rows, cols, square_size, img_size
	# rows, cols, square_size = 6, 9, 1
	# simple_calibrate('./n_pairs')

	# individual_calibrate('./n_pairs')
	# stereo_calibrate()
	# stereo_rectify()
	# undistort()

	# caliberate()
	load_parameters()
	img_left, img_right = cv2.imread('./rectified/left_01.png'), cv2.imread('./rectified/right_01.png')
	img_left, img_right = generate_undistored_rectified_image(img_left, img_right)
	cv2.imwrite('rleft_01.png', img_left)
	cv2.imwrite('rright_01.png', img_right)
