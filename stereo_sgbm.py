import cv2
import json
import glob
import blinker
import numpy as np
import calibrate
from Video import Video
from threading import Thread
# from Show import Show
# from Counter import Counter

f = open("sgbm_params.json", "r+")
camera_params = json.load(f)

scale = camera_params['scale']
mode, path = camera_params['mode'], camera_params['path']
marking, running = False, True
rectified_pair, disparity = None, np.zeros((650, 800))

cv2.namedWindow('Control', cv2.WINDOW_NORMAL)
cv2.namedWindow('Rectified Pair', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Rectified Pair', (camera_params['width'], camera_params['height']))
cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

stereo = cv2.StereoSGBM_create(mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
light = blinker.Blinker(leds=3, delay=250, brightness=0.2)
# cps = Counter().start()

def nothing(x):
	pass

cv2.createTrackbar('minDisparity', 'Control', 5, 25, nothing)
cv2.createTrackbar('numDisparities', 'Control', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'Control', 5, 50, nothing)
cv2.createTrackbar('winSize', 'Control', 5, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'Control', 5, 25, nothing)
cv2.createTrackbar('preFilterCap', 'Control', 5, 62, nothing)
cv2.createTrackbar('uniquenessRatio', 'Control', 15, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'Control', 3, 25, nothing)
cv2.createTrackbar('speckleRange', 'Control', 0, 100, nothing)
cv2.createTrackbar('threshold', 'Control', 0, 255, nothing)

cv2.createTrackbar('LEDs', 'Control', 1, 12, nothing)
cv2.createTrackbar('LED Brightness', 'Control', 0, 100, nothing)
cv2.createTrackbar('LED delay', 'Control', 10, 2000, nothing)
cv2.createTrackbar('LED mode', 'Control', 0, 1, nothing)

cv2.setTrackbarPos('numDisparities', 'Control', camera_params['numDisparities'])
cv2.setTrackbarPos('blockSize', 'Control', camera_params['blockSize'])
cv2.setTrackbarPos('preFilterCap', 'Control', camera_params['preFilterCap'])
cv2.setTrackbarPos('uniquenessRatio', 'Control', camera_params['uniquenessRatio'])
cv2.setTrackbarPos('speckleRange', 'Control', camera_params['speckleRange'])
cv2.setTrackbarPos('speckleWindowSize', 'Control', camera_params['speckleWindowSize'])
cv2.setTrackbarPos('disp12MaxDiff', 'Control', camera_params['disp12MaxDiff'])
cv2.setTrackbarPos('minDisparity', 'Control', camera_params['minDisparity'])
cv2.setTrackbarPos('winSize', 'Control', camera_params['winSize'])
cv2.setTrackbarPos('threshold', 'Control', camera_params['threshold'])

cv2.setTrackbarPos('LEDs', 'Control', camera_params['leds'])
cv2.setTrackbarPos('LED Brightness', 'Control', camera_params['led_bright'])
cv2.setTrackbarPos('LED delay', 'Control', camera_params['led_delay'])
cv2.setTrackbarPos('LED mode', 'Control', camera_params['led_mode'])

calibrate.load_parameters()
n_file, i = len(glob.glob(f'./{path}/*.png')) // 2, 0

contr = []
maskLeft, maskRight = None, None

def createMask(event, x, y, flags, params):
	global contr, maskLeft, maskRight
		
	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	contr.append([x, y])
	# 	if len(contr) == 4 and len(imgLeft) > 0:
	# 		pts = np.array(contr)
	# 		maskLeft = np.zeros(rectified_pair[0].shape[:2], dtype="uint8")
	# 		cv2.fillPoly(maskLeft, pts=[pts], color=(255, 255, 255))
			# img = cv2.bitwise_and(rectified_pair[0], rectified_pair[0], mask=img)
			# cv2.imshow('maskedL', img)
			# cv2.imwrite(f'./test/mask/left_{(i + 1)}.png', img)
		# elif len(contr) == 8 and len(imgRight) > 0:
		# 	pts = np.array(contr[4:]) - np.array([stream.img_width // 2, 0])
		# 	contr = []
		# 	maskRight = np.zeros(rectified_pair[1].shape[:2], dtype="uint8")
		# 	cv2.fillPoly(maskRight, pts=[pts], color=(255, 255, 255))
		# 	print('right mask created')
			# img = cv2.bitwise_and(rectified_pair[1], rectified_pair[1], mask=img)
			# cv2.imshow('maskedR', img)
			# cv2.imwrite(f'./test/mask/right_{(i + 1)}.png', img)
	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	contr.append([x, y])
	# 	if len(contr) == 4 and len(imgLeft) > 0:
	# 		pts = np.array(contr)
	# 		contr = []
	# 		img = np.zeros(rectified_pair[0].shape[:2], dtype="uint8")
	# 		cv2.fillPoly(img, pts=[pts], color=(255, 255, 255))
	# 		img = cv2.bitwise_and(rectified_pair[0], rectified_pair[0], mask=img)
	# 		cv2.imshow('maskedL', img)
	# 		cv2.imwrite(f'./test/mask/left_{(i + 1)}.png', img)


cv2.setMouseCallback('Rectified Pair', createMask)
# cv2.setMouseCallback('Disparity', createMask)

def gamma_trans(img, gamma):
	gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return cv2.LUT(img, gamma_table)


def compute_disparity():
	global disparity
	l = 70000
	s = 1.2
	while running:
		stereo.setNumDisparities(camera_params["numDisparities"] * 16)
		stereo.setMinDisparity(camera_params["minDisparity"])
		stereo.setBlockSize(camera_params["blockSize"] * 2 + 5)
		stereo.setUniquenessRatio(camera_params["uniquenessRatio"])
		stereo.setSpeckleWindowSize(camera_params["speckleWindowSize"] * 2)
		stereo.setSpeckleRange(camera_params["speckleRange"])
		stereo.setDisp12MaxDiff(camera_params["disp12MaxDiff"])
		stereo.setP1(8 * 3 * camera_params["winSize"] ** 2)
		stereo.setP2(32 * 3 * camera_params["winSize"] ** 2)

		left_matcher = stereo
		right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
		
		disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
		disparity_filter.setLambda(l)
		disparity_filter.setSigmaColor(s)
		
		if rectified_pair:
			d_l = left_matcher.compute(rectified_pair[0], rectified_pair[1])
			d_r = right_matcher.compute(rectified_pair[1], rectified_pair[0])

			d_l = np.int16(d_l)
			d_r = np.int16(d_r)

			d_filter = disparity_filter.filter(d_l, rectified_pair[0], None, d_r)
			disparity = cv2.normalize(d_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
			
			# disparity = np.uint8(disparity_)
			# grayscale disparity map
			# disparity_ = disparity_.astype(np.float32)
			# disparity = (disparity_ / 16.0 - camera_params["minDisparity"]) / camera_params["numDisparities"]

			# local_max = disparity_.max()
			# local_min = disparity_.min()
			# disparity_grayscale = (disparity_-local_min) * (65535.0/(local_max-local_min))
			# disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
			# disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
			# disparity = disparity_color

			# disparity_ = np.uint8(disparity_)
			# pointCloud = cv2.reprojectImageTo3D(disparity_, disparityToDepthMap) / 420

def rectify():
	global imgLeft, imgRight, rectified_pair
	while running:
		img_left, img_right = calibrate.generate_undistored_rectified_image(imgLeft, imgRight)
		# if stream.running:
		# 	img_left = cv2.bitwise_and(img_left, img_left, mask=maskLeft)
		# 	img_right = cv2.bitwise_and(img_right, img_right, mask=maskRight)
		rectified_pair = (img_left, img_right)


def driver():
	global running, rectified_pair, mode, marking, light, camera_params, i, n_file, imgLeft, imgRight
	while running:
		key = cv2.waitKey(1) & 0xFFFF
		if mode == 0:  # video
			imgLeft, imgRight = stream.get_images()
			# out.display(imgLeft)
			# print(cps.countsPerSec())

			# imgLeft = gamma_trans(imgLeft, 2)
			# imgRight = gamma_trans(imgRight, 2)

			if key == ord("c"):
				n_file += 1
				cv2.imwrite(f'./{path}/left_{n_file}.png', imgLeft)
				cv2.imwrite(f'./{path}/right_{n_file}.png', imgRight)
				# print('saving')
				# cv2.imwrite(f'./test/test2/{n_file}.png', disparity)

		elif mode == 1:  # static
			if key == ord('n'):
				i = (i + 1) % n_file
			imgLeft = cv2.imread(f'./{path}/left_{(i + 1)}.png', 0)
			imgRight = cv2.imread(f'./{path}/right_{(i + 1)}.png', 0)

		if key == ord("m"):
			mode = (mode + 1) % 2
			continue

		elif key == ord("b"):
			marking = not marking
			continue

		elif key == ord("l"):
			light.lighting_toggle()

		elif key == ord("q"):
			running = False
			break

		# cv2.imshow("Left image before rectification", imgLeft)
		# cv2.imshow("Right image before rectification", imgRight)
		# cps.increment()
		# img = np.concatenate((imgLeft, imgRight), axis=1)
		# cv2.imshow("Image pair", img)
		img = np.concatenate(rectified_pair, axis=1)
		cv2.imshow("Rectified Pair", img)

		camera_params["leds"] = cv2.getTrackbarPos('LEDs', 'Control')
		camera_params["led_bright"] = cv2.getTrackbarPos('LED Brightness', 'Control')
		camera_params["led_delay"] = cv2.getTrackbarPos('LED delay', 'Control')
		camera_params["led_mode"] = cv2.getTrackbarPos('LED mode', 'Control')

		light.params(camera_params["leds"], camera_params["led_mode"] ==
				1, camera_params["led_delay"], camera_params["led_bright"] / 100)

		camera_params["numDisparities"] = cv2.getTrackbarPos('numDisparities', 'Control')
		camera_params["minDisparity"] = cv2.getTrackbarPos('minDisparity', 'Control')
		camera_params["blockSize"] = cv2.getTrackbarPos('blockSize', 'Control')
		camera_params["uniquenessRatio"] = cv2.getTrackbarPos('uniquenessRatio', 'Control')
		camera_params["speckleWindowSize"] = cv2.getTrackbarPos('speckleWindowSize', 'Control')
		camera_params["speckleRange"] = cv2.getTrackbarPos('speckleRange', 'Control')
		camera_params["speckleRange"] = cv2.getTrackbarPos('speckleRange', 'Control')
		camera_params["winSize"] = cv2.getTrackbarPos('winSize', 'Control')

		if marking:
			pass

		else:
			cv2.imshow("Disparity", disparity)

	cv2.destroyAllWindows()
	stream.close()
	light.lighting_stop()

	f.seek(0)
	camera_params['mode'] = mode
	json.dump(camera_params, f, indent=4)
	f.truncate()

if __name__ == '__main__':
	stream = Video(scale=scale).start()
	imgLeft, imgRight = stream.get_images()

	rectify_t = Thread(target=rectify)
	print('Starting rectification thread...')
	rectify_t.start()
	print('Rectification thread running')

	disparity_t = Thread(target=compute_disparity)
	print('Starting disparity thread...')
	disparity_t.start()
	print('Disparity thread running')

	print('running driver')
	driver()

	rectify_t.join()
	print('Rectification thread stopped')
	disparity_t.join()
	print('Disparity thread stopped')