import cv2
import numpy as np
import blinker
import glob
import threading
from arducam_camera import MyCamera

scale = 0.4
j = 0
camera = MyCamera()
camera.open_camera()
fmt = camera.get_framesize()
cam_width, cam_height = camera.get_framesize()
img_width, img_height = int(cam_width * scale), int(cam_height * scale)
light = blinker.Blinker(leds=3, delay=6000, brightness=0.2)
t = None

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

def rectify(imgLeft, imgRight):
	global j
	sift = cv2.SIFT_create()
	kp1, des1 = sift.detectAndCompute(imgLeft, None)
	kp2, des2 = sift.detectAndCompute(imgRight, None)
	# imgSift = cv2.drawKeypoints(imgLeft, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imshow("SIFT", imgSift)

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	matchesMask = [[0, 0] for i in range(len(matches))]
	good, pts1, pts2 = [], [], []

	for i, (m, n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			matchesMask[i] = [1, 0]
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	# draw_params = dict(matchColor=(0, 255, 0),
	# 				   singlePointColor=(255, 0, 0),
	# 				   matchesMask=matchesMask[300:500],
	# 				   flags=cv2.DrawMatchesFlags_DEFAULT)

	# keypoint_matches = cv2.drawMatchesKnn(
	# 	imgLeft, kp1, imgRight, kp2, matches[300:500], None, **draw_params)
	# cv2.imshow("Keypoint matches", keypoint_matches)

	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

	pts1 = pts1[inliers.ravel() == 1]
	pts2 = pts2[inliers.ravel() == 1]

	# lines1 = cv2.computeCorrespondEpilines(
    # pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
	# lines1 = lines1.reshape(-1, 3)
	# img5, img6 = drawlines(imgLeft, imgRight, lines1, pts1, pts2)

	# lines2 = cv2.computeCorrespondEpilines(
	# 	pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
	# lines2 = lines2.reshape(-1, 3)
	# img3, img4 = drawlines(imgRight, imgLeft, lines2, pts2, pts1)

	# cv2.imshow('5', img5)
	# cv2.imshow('3', img3)
	h1, w1 = imgLeft.shape
	h2, w2 = imgRight.shape
	_, H1, H2 = cv2.stereoRectifyUncalibrated(
		np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
	)
	img1_rectified = cv2.warpPerspective(imgLeft, H1, (w1, h1))
	img2_rectified = cv2.warpPerspective(imgRight, H2, (w2, h2))
	cv2.imshow("rectified_1.png", img1_rectified)
	cv2.imshow("rectified_2.png", img2_rectified)

	# j += 1
	# cv2.imwrite(f'./{"rectified"}/left_{j:02}.png', img1_rectified)
	# cv2.imwrite(f'./{"rectified"}/right_{j:02}.png', img2_rectified)

	block_size = 1
	min_disp = -128
	max_disp = 128

	num_disp = max_disp - min_disp
	uniquenessRatio = 5
	speckleWindowSize = 200
	speckleRange = 2
	disp12MaxDiff = 0

	stereo = cv2.StereoSGBM_create(
		minDisparity=min_disp,
		numDisparities=num_disp,
		blockSize=block_size,
		uniquenessRatio=uniquenessRatio,
		speckleWindowSize=speckleWindowSize,
		speckleRange=speckleRange,
		disp12MaxDiff=disp12MaxDiff,
		P1=8 * 1 * block_size * block_size,
		P2=32 * 1 * block_size * block_size
	)

	disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)
	disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,beta=0, norm_type=cv2.NORM_MINMAX)
	disparity_SGBM = np.uint8(disparity_SGBM)
	cv2.imshow("Disparity", disparity_SGBM)
	
n_file, i = len(glob.glob(f'./{"n_pairs"}/*.png')) // 2, 0
imgLeft, imgRight = np.zeros((img_height, img_width // 2, 3), np.uint8), np.zeros((img_height, img_width // 2, 3), np.uint8)

while True:
	key = cv2.waitKey(1) & 0xFFFF
	frame = camera.get_frame()
	frame = cv2.resize(frame, (img_width, img_height))

	pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	imgLeft = pair_img[0:img_height, 0:int(img_width / 2)]
	imgRight = pair_img[0:img_height, int(img_width / 2):img_width]

	if key == ord('n'):
		if t:
			t.join()
		t = threading.Thread(target=rectify, args=(imgLeft, imgRight))
		t.start()

	elif key == ord("l"):
		light.lighting_toggle()

	if key == ord("q"):
		break

	cv2.imshow("Left image before rectification", imgLeft)
	cv2.imshow("Right image before rectification", imgRight)

cv2.destroyAllWindows()
camera.close_camera()
light.lighting_stop()
