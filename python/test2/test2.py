'''
@Author: Jack
@Date: 2020-04-02 12:55:27
@LastEditors: Jack
@LastEditTime: 2020-04-02 14:07:00
@Description: 把一张图片从歪的变成正的，方便读取信息 https://towardsdatascience.com/document-scanner-using-computer-vision-opencv-and-python-20b87b1cbb06
'''

# import required libraries
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def show_img(opencv_img):
	image = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
	pixels = np.array(image)
	plt.imshow(pixels)
	plt.show() 


# parameter for image to scan/process
args_image = "test3.png"
# read the image
image = cv2.imread(args_image)
orig = image.copy()

# convert image to gray scale. This will remove any color noise
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur the image to remove high frequency noise 
# it helps in finding/detecting contour in gray image
grayImageBlur = cv2.blur(grayImage,(3,3))
# then we performed canny edge detection
edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
# show the gray and edge-detected image
# show_img(grayImage)
# show_img(grayImageBlur)
# show_img(edgedImage)

# find the contours in the edged image, sort area wise 
# keeping only the largest ones 
allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
# descending sort contours area and keep top 1
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
# approximate the contour
perimeter = cv2.arcLength(allContours[0], True) 
ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
# show the contour on image
cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)

# show_img(image)

# reshape coordinates array
ROIdimensions = ROIdimensions.reshape(4,2)
# list to hold ROI coordinates
rect = np.zeros((4,2), dtype="float32")
# top left corner will have the smallest sum, 
# bottom right corner will have the largest sum
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]
# top-right will have smallest difference
# botton left will have largest difference
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]
# top-left, top-right, bottom-right, bottom-left
(tl, tr, br, bl) = rect
# compute width of ROI
widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
maxWidth = max(int(widthA), int(widthB))
# compute height of ROI
heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
maxHeight = max(int(heightA), int(heightB))

# Set of destinations points for "birds eye view"
# dimension of the new image
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")
# compute the perspective transform matrix and then apply it
transformMatrix = cv2.getPerspectiveTransform(rect, dst)
# transform ROI
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
# lets see the wraped document

# show_img(scan)

# convert to gray
scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

# ------------------------------
# convert to black/white with high contrast for documents
from skimage.filters import threshold_local
# increase contrast incase its document
T = threshold_local(scanGray, 9, offset=8, method="gaussian")
scanBW = (scanGray > T).astype("uint8") * 255
# display final high-contrast image

show_img(scanBW)