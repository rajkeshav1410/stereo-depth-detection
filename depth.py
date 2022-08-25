import cv2
import json
import glob
import blinker
import numpy as np
from arducam_camera import MyCamera
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration

f = open("camera_params.json", "r+")
camera_params = json.load(f)

scale = camera_params['scale']
mode, path = camera_params['mode'], camera_params['path']
marking = False

camera = MyCamera()
camera.open_camera()
fmt = camera.get_framesize()
cam_width, cam_height = camera.get_framesize()
img_width, img_height = int(cam_width * scale), int(cam_height * scale)

light = blinker.Blinker(leds=3, delay=250, brightness=0.2)

image_area = img_width * img_height
max_area, min_area = 0.95 * image_area, 0.0005 * image_area

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('disp', 600, 600)


def nothing(x):
	pass


cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)
cv2.createTrackbar('winSize', 'disp', 5, 25, nothing)
cv2.createTrackbar('threshold', 'disp', 0, 255, nothing)

cv2.createTrackbar('LEDs', 'disp', 1, 12, nothing)
cv2.createTrackbar('LED Brightness', 'disp', 0, 100, nothing)
cv2.createTrackbar('LED delay', 'disp', 10, 2000, nothing)
cv2.createTrackbar('LED mode', 'disp', 0, 1, nothing)

cv2.setTrackbarPos('numDisparities', 'disp', camera_params['numDisparities'])
cv2.setTrackbarPos('blockSize', 'disp', camera_params['blockSize'])
cv2.setTrackbarPos('preFilterType', 'disp', camera_params['preFilterType'])
cv2.setTrackbarPos('preFilterSize', 'disp', camera_params['preFilterSize'])
cv2.setTrackbarPos('preFilterCap', 'disp', camera_params['preFilterCap'])
cv2.setTrackbarPos('textureThreshold', 'disp',
				   camera_params['textureThreshold'])
cv2.setTrackbarPos('uniquenessRatio', 'disp', camera_params['uniquenessRatio'])
cv2.setTrackbarPos('speckleRange', 'disp', camera_params['speckleRange'])
cv2.setTrackbarPos('speckleWindowSize', 'disp',
				   camera_params['speckleWindowSize'])
cv2.setTrackbarPos('disp12MaxDiff', 'disp', camera_params['disp12MaxDiff'])
cv2.setTrackbarPos('minDisparity', 'disp', camera_params['minDisparity'])
cv2.setTrackbarPos('winSize', 'disp', camera_params['winSize'])
cv2.setTrackbarPos('threshold', 'disp', camera_params['threshold'])

cv2.setTrackbarPos('LEDs', 'disp', camera_params['leds'])
cv2.setTrackbarPos('LED Brightness', 'disp', camera_params['led_bright'])
cv2.setTrackbarPos('LED delay', 'disp', camera_params['led_delay'])
cv2.setTrackbarPos('LED mode', 'disp', camera_params['led_mode'])

# cv2.createButton("Mode", nothing, None, cv2.QT_PUSH_BUTTON, 1)

stereo = cv2.StereoBM_create()
# stereo = cv2.StereoSGBM_create()
backSub = cv2.createBackgroundSubtractorKNN()

# cv_file = cv2.FileStorage("./params.xml", cv2.FILE_STORAGE_READ)
# Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
# Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
# Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
# Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
# cv_file.release()

calibration = StereoCalibration(input_folder='calib_res')

n_file, i = len(glob.glob(f'./{path}/*.png')) // 2, 0

while True:
	key = cv2.waitKey(1) & 0xFFFF
	if mode == 0:  # video
		frame = camera.get_frame()
		frame = cv2.resize(frame, (img_width, img_height))
		pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		imgLeft = pair_img[0:img_height, 0:int(img_width/2)]
		imgRight = pair_img[0:img_height, int(img_width/2):img_width]

		if key == ord("c"):
			n_file += 1
			cv2.imwrite(f'./{path}/left_{n_file:02}.png', imgLeft)
			cv2.imwrite(f'./{path}/right_{n_file:02}.png', imgRight)

	elif mode == 1:  # static
		if key == ord('n'):
			i = (i + 1) % n_file
		imgLeft = cv2.imread(f'./{path}/left_{(i + 1):02}.png', 0)
		imgRight = cv2.imread(f'./{path}/right_{(i + 1):02}.png', 0)

	if key == ord("m"):
		mode = (mode + 1) % 2
		continue

	elif key == ord("b"):
		marking = not marking
		continue

	elif key == ord("l"):
		light.lighting_toggle()

	elif key == ord("q"):
		break

	cv2.imshow("Left image before rectification", imgLeft)
	cv2.imshow("Right image before rectification", imgRight)

	# Left_nice= cv2.remap(imgLeft, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
	# Right_nice= cv2.remap(imgRight, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

	# cv2.imshow("Left image after rectification", Left_nice)
	# cv2.imshow("Right image after rectification", Right_nice)

	rectified_pair = calibration.rectify((imgLeft, imgRight))
	rectified_pair[0] = rectified_pair[0][:img_height, :int(img_width / 2)]
	rectified_pair[1] = rectified_pair[1][:img_height, :int(img_width / 2)]

	cv2.imshow("Left image after rectification", rectified_pair[0])
	cv2.imshow("Right image after rectification", rectified_pair[1])

	# for line in range(0, int(rectified_pair[0].shape[0] / 20)):
	# 	rectified_pair[0][line * 20, :] = 255
	# 	rectified_pair[1][line * 20, :] = 255
	# cv2.imshow("Left image after rectification", rectified_pair[0])
	# cv2.imshow("Right image after rectification", rectified_pair[1])

	camera_params["numDisparities"] = cv2.getTrackbarPos(
		'numDisparities', 'disp')
	camera_params["blockSize"] = cv2.getTrackbarPos('blockSize', 'disp')
	camera_params["preFilterType"] = cv2.getTrackbarPos(
		'preFilterType', 'disp')
	camera_params["preFilterSize"] = cv2.getTrackbarPos(
		'preFilterSize', 'disp')
	camera_params["preFilterCap"] = cv2.getTrackbarPos('preFilterCap', 'disp')
	camera_params["textureThreshold"] = cv2.getTrackbarPos(
		'textureThreshold', 'disp')
	camera_params["uniquenessRatio"] = cv2.getTrackbarPos(
		'uniquenessRatio', 'disp')
	camera_params["speckleRange"] = cv2.getTrackbarPos('speckleRange', 'disp')
	camera_params["speckleWindowSize"] = cv2.getTrackbarPos(
		'speckleWindowSize', 'disp')
	camera_params["disp12MaxDiff"] = cv2.getTrackbarPos(
		'disp12MaxDiff', 'disp')
	camera_params["minDisparity"] = cv2.getTrackbarPos('minDisparity', 'disp')
	camera_params["threshold"] = cv2.getTrackbarPos('threshold', 'disp')

	camera_params["leds"] = cv2.getTrackbarPos('LEDs', 'disp')
	camera_params["led_bright"] = cv2.getTrackbarPos('LED Brightness', 'disp')
	camera_params["led_delay"] = cv2.getTrackbarPos('LED delay', 'disp')
	camera_params["led_mode"] = cv2.getTrackbarPos('LED mode', 'disp')

	stereo.setNumDisparities(camera_params["numDisparities"] * 16)
	stereo.setBlockSize(camera_params["blockSize"] * 2 + 5)
	stereo.setPreFilterType(camera_params["preFilterType"])
	stereo.setPreFilterSize(camera_params["preFilterSize"] * 2 + 5)
	stereo.setPreFilterCap(camera_params["preFilterCap"])
	stereo.setTextureThreshold(camera_params["textureThreshold"])
	stereo.setUniquenessRatio(camera_params["uniquenessRatio"])
	stereo.setSpeckleRange(camera_params["speckleRange"])
	stereo.setSpeckleWindowSize(camera_params["speckleWindowSize"] * 2)
	stereo.setDisp12MaxDiff(camera_params["disp12MaxDiff"])
	stereo.setMinDisparity(camera_params["minDisparity"])

	light.params(camera_params["leds"], camera_params["led_mode"] ==
				 1, camera_params["led_delay"], camera_params["led_bright"] / 100)

	# for StereoSGBM
	# camera_params["numDisparities"] = cv2.getTrackbarPos('numDisparities', 'disp')
	# camera_params["minDisparity"] = cv2.getTrackbarPos('minDisparity', 'disp')
	# camera_params["blockSize"] = cv2.getTrackbarPos('blockSize', 'disp')
	# camera_params["uniquenessRatio"] = cv2.getTrackbarPos('uniquenessRatio', 'disp')
	# camera_params["speckleWindowSize"] = cv2.getTrackbarPos('speckleWindowSize', 'disp')
	# camera_params["speckleRange"] = cv2.getTrackbarPos('speckleRange', 'disp')
	# camera_params["speckleRange"] = cv2.getTrackbarPos('speckleRange', 'disp')
	# camera_params["winSize"] = cv2.getTrackbarPos('winSize', 'disp')

	# stereo.setNumDisparities(camera_params["numDisparities"] * 16)
	# stereo.setMinDisparity(camera_params["minDisparity"])
	# stereo.setBlockSize(camera_params["blockSize"] * 2 + 5)
	# stereo.setUniquenessRatio(camera_params["uniquenessRatio"])
	# stereo.setSpeckleWindowSize(camera_params["speckleWindowSize"] * 2)
	# stereo.setSpeckleRange(camera_params["speckleRange"])
	# stereo.setDisp12MaxDiff(camera_params["disp12MaxDiff"])
	# stereo.setP1(8 * camera_params["winSize"] ** 2)
	# stereo.setP2(3 * camera_params["winSize"] ** 2)
	if marking:
		# ret, thres = cv2.threshold(imgRight, camera_params["threshold"], 255, cv2.THRESH_BINARY)
		# _, contours, heirarchy = cv2.findContours(image=thres, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
		# img_cpy = imgRight.copy()
		# cv2.drawContours(image=img_cpy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
		# cv2.imshow("Binary image", img_cpy)

		# img_cpy = imgRight.copy()
		# img_blur = cv2.GaussianBlur(img_cpy, (3, 3), 0)
		# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
		# edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
		# cv2.imshow('Sobel', sobelxy)
		# edges = cv2.addWeighted(imgRight, 1, edges, 1, 0)
		# cv2.imshow('Canny', edges)

		# Grab-cut Algo for foreground
		mask = np.zeros(rectified_pair[1].shape[:2], np.uint8)
		bgdModel = np.zeros((1, 65), np.float64)
		fgdModel = np.zeros((1, 65), np.float64)
		rect = (65, 190, 402, 296)
		img = cv2.cvtColor(rectified_pair[1], cv2.COLOR_GRAY2BGR)
		cv2.grabCut(img, mask, rect, bgdModel,
					fgdModel, 1, cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
		img = img * mask2[:, :, np.newaxis]
		cv2.imshow('Grab cut', img)

		# img_cpy = imgRight.copy()
		# mask = np.zeros(img_cpy.shape, dtype=np.uint8)
		# ret, thres = cv2.threshold(imgRight, camera_params["threshold"], 255, cv2.THRESH_BINARY)
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		# opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=3)
		# _, contours, heirarchy = cv2.findContours(image=opening, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
		# for cnt in contours:
		# 	cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
		# close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
		# # close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
		# res = cv2.bitwise_and(img_cpy, img_cpy, mask=close)
		# cv2.imshow('Result', res)

		# img_cpy = imgRight.copy()
		# mask = np.zeros(img_cpy.shape, dtype=np.uint8)
		# ret, thres = cv2.threshold(imgRight, camera_params["threshold"], 255, cv2.THRESH_BINARY)
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		# edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=3)
		# _, contours, heirarchy = cv2.findContours(image=edges, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
		# contour_info = []
		# for cnt in contours:
		# 	contour_info.append((cnt, cv2.contourArea(cnt)))
		# close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
		# # close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
		# res = cv2.bitwise_and(img_cpy, img_cpy, mask=close)
		# cv2.imshow('Result', res)

		# Background substraction method
		# fgMask = backSub.apply(imgRight)
		# cv2.rectangle(imgRight, (10, 2), (100,20), (255,255,255), -1)
		# cv2.putText(imgRight, "str(frame.get(cv2.CAP_PROP_POS_FRAMES))", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
		# # cv2.imshow('Frame', frame)
		# res = cv2.bitwise_and(imgRight, imgRight, mask=fgMask)
		# cv2.imshow('FG Mask', res)

	else:
		disparity = stereo.compute(rectified_pair[0], rectified_pair[1])

		# grayscale disparity map
		# disparity = disparity.astype(np.float32)
		# disparity = (disparity / 16.0 - camera_params["minDisparity"]) / camera_params["numDisparities"]
		# cv2.imshow("disparity", disparity)

		local_max = disparity.max()
		local_min = disparity.min()
		# print((local_max, local_min))
		disparity_grayscale = (disparity-local_min) * \
			(65535.0/(local_max-local_min))
		disparity_fixtype = cv2.convertScaleAbs(
			disparity_grayscale, alpha=(255.0/65535.0))
		disparity_color = cv2.applyColorMap(
			disparity_fixtype, cv2.COLORMAP_JET)
		cv2.imshow("Disparity", disparity_color)

cv2.destroyAllWindows()
camera.close_camera()

light.lighting_stop()

f.seek(0)
camera_params['mode'] = mode
json.dump(camera_params, f, indent=4)
f.truncate()
