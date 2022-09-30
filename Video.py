from threading import Thread
import cv2
from arducam_camera import MyCamera

class Video:
	def __init__(self, src=0, scale=0.5):
		# self.camera = MyCamera()
		# self.camera.open_camera()

		# cam_width, cam_height = self.camera.get_framesize()
		# self.img_width, self.img_height = int(cam_width * scale), int(cam_height * scale)

		camera = MyCamera()
		camera.open_camera()

		print('Getting camera dimension')
		cam_width, cam_height = camera.get_framesize()
		self.img_width, self.img_height = int(cam_width * scale), int(cam_height * scale)
		camera.close_camera()

		print('Starting stearm...')
		self.stream = cv2.VideoCapture(src)
		self.get_frame()
		self.running = True

	def start(self):
		self.thread = Thread(target=self.get, args=())
		self.thread.start()
		print('Video stream started')
		return self

	def get_frame(self):
		self.connected, frame = self.stream.read()

		# self.frame = self.camera.get_frame()
		frame = cv2.resize(frame, (self.img_width, self.img_height))
		pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		self.imgLeft = pair_img[0:self.img_height, 0:int(self.img_width/2)]
		self.imgRight = pair_img[0:self.img_height, int(self.img_width/2):self.img_width]

	def get(self):
		while self.running:
			if not self.connected:
				print('Stream closing...')
				self.close()
			else:
				self.get_frame()

	def close(self):
		self.running = False

	def get_images(self):
		return [self.imgLeft, self.imgRight]
