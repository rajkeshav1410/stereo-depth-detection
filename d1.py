import cv2
import json
from arducam_camera import MyCamera

with open("camera_params.json", "r") as f:
	camera_params = json.load(f)

scale = camera_params['scale']
cam_width = camera_params['width']
cam_height = camera_params['height']
img_width = int(cam_width * scale)
img_height = int(cam_height * scale)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

camera = MyCamera()
camera.open_camera(camera_params['device'], cam_width, cam_height)
fmt = camera.get_framesize()
print("Current resolution: {}x{}".format(fmt[0], fmt[1]))

cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


while True:
	frame = camera.get_frame()
	pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
	imgLeft = pair_img [0:img_height,0:int(img_width/2)]
	imgRight = pair_img [0:img_height,int(img_width/2):img_width]
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		quit()
	cv2.imshow("left", imgLeft)
	cv2.imshow("right", imgRight)

# camera.close_camera()