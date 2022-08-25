import time
import board
import neopixel
import threading
# import argparse


class Blinker:

	# parser = argparse.ArgumentParser()
	# parser.add_argument("-l", "--leds", required=False, type=int,
	# 					default=1, help="Number of LED's [1-12]to glow. Default is 1")
	# parser.add_argument("-b", "--brightness", required=False, type=float,
	# 					default=1.0, help="Set brightness [0.0-1.0] of LED. Default is 1.0")
	# parser.add_argument("-d", "--delay", required=False, type=int,
	# 					default=100, help="Wait time in millisec")
	# parser.add_argument("-c", "--continuous", action="store_true",
	# 					help="Move in continuous order")
	# args = parser.parse_args()

	def __init__(self, leds=1, continuous=False, delay=100, brightness=1.0):

		self.pixel_pin = board.D18
		self.num_pixels = 12
		self.ORDER = neopixel.GRB
		self.running = False

		self.n = leds
		self.cntnus = continuous
		self.wait = delay / 1000
		self.bright = brightness

	def white_cycle(self):
		pixels = neopixel.NeoPixel(
			self.pixel_pin, self.num_pixels, auto_write=True, pixel_order=self.ORDER
			)
		self.running = True
		while self.running:
			if not self.cntnus:
				for i in range(0, self.num_pixels, self.n):
					for j in range(self.n):
						pixels[(i + j) % self.num_pixels] = (0, 0, 0)
						pixels[(i + self.n + j) % self.num_pixels] = (int(255 * self.bright), int(255 * self.bright), int(255 * self.bright))
					time.sleep(self.wait)
			else:
				for i in range(self.num_pixels):
					for j in range(self.n):
						pixels[(i + j) % self.num_pixels] = (0, 0, 0)
						pixels[(i + self.n + j) % self.num_pixels] = (int(255 * self.bright), int(255 * self.bright), int(255 * self.bright))
					time.sleep(self.wait)
		self.pixels = pixels

	def lighting_start(self):
		self.thread = threading.Thread(target=self.white_cycle)
		self.thread.start()
		# white_cycle(args.leds, args.continuous, args.delay / 1000, args.brightness)

	def lighting_stop(self):
		self.running = False
		try:
			self.thread.join()
			for i in range(self.num_pixels):
				self.pixels[i] = (0, 0, 0)
		except Exception:
			pass

	def lighting_toggle(self):
		if self.running:
			self.lighting_stop()
		else:
			self.lighting_start()

	def params(self, leds, continuous, delay, brightness):
		self.n = leds
		self.cntnus = continuous
		self.wait = delay / 1000
		self.bright = brightness
		
# b = Blinker(leds=3, continuous=True, delay=20, brightness=0.05)
# b.lighting_start()
# try:
# 	while True:
# 		continue
# except KeyboardInterrupt:
# 	b.lighting_stop()