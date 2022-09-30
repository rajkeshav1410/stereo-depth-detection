import cv2
import csv
from glob import glob
from matplotlib import pyplot as plt

path = './test/test2/'
running = True
files, i = glob(f'{path}*.png'), 0
img = cv2.imread(files[0], 0)
data = []

cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

pnts = []
def getArea(event, x, y, flags, params):
    global pnts
    if event == cv2.EVENT_LBUTTONDOWN:
        pnts.append([x, y])
    if len(pnts) == 2:
        cimg = img[pnts[0][1]:pnts[1][1], pnts[0][0]:pnts[1][0]]
        cv2.imshow('clip', cimg)
        # print(cimg.min(), cimg.max())
        hist = cv2.calcHist([cimg], [0], None, [256], [0, 256])
        # plt.plot(hist)
        # plt.show()
        hist = [val[0] for val in hist]
        index = list(range(0, 256))
        s = [(x, y) for y, x in sorted(zip(hist, index), reverse=True)]
        # print(s[0][0])
        data.append([i + 1, cimg.min(), cimg.max(), s[0][0]])
        pnts = []

cv2.setMouseCallback('Disparity', getArea)
while running:
    key = cv2.waitKey(1) & 0xFFFF
    if key == ord('q'):
        running = False
    
    elif key == ord('n'):
        i = (i + 1) % len(files)
        img = cv2.imread(files[i])

    cv2.imshow('Disparity', img)

with open('depth_data.csv', 'w') as f:
    file = csv.writer(f)
    file.writerows(data)