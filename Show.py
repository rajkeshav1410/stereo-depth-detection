from threading import Thread
import cv2
import matplotlib.pyplot as plt

class Show:
	def __init__(self, frame_name):
		self.frame = []
		self.thread = None
		self.frame_name = frame_name
		self.key = ''
		print('\nGot frame name', frame_name)
		self.running = True
		self.panel = None
	
	def display(self, frame):
		# self.frames[frame_name] = frame
		self.frame = frame

	def start(self):
		print('Starting display...')
		self.thread = Thread(target=self.show, args=())
		self.thread.start()
		print('Display thread working')

	def show(self):
		# while self.running:
		#     # for frame_name, frame in self.frames.items():
		#     if len(self.frame) == 0:
		#         continue
		#     cv2.imshow(self.frame_name, self.frame)
		#     print('Redrawn')
		#     self.key = cv2.waitKey(1)
		#     if self.key == ord('q'):
		#         self.running = False
		while self.running:
			if type(self.frame) == type([]):
				continue
			if self.panel == None:
				ax = plt.subplot()
				self.panel = ax.imshow(self.frame)
				plt.ion()
				# plt.show()
			else:
				self.panel.set_data(self.frame)

	def close(self):
		self.running = False
		self.thread.join()
		plt.ioff()
		# cv2.destroyAllWindows()
