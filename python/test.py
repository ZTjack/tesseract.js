'''
@Author: Jack
@Date: 2020-03-31 14:27:11
@LastEditors: Jack
@LastEditTime: 2020-04-02 13:05:35
@Description: 
'''

import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from matplotlib import pyplot as plt

# 读取灰度图片
img_gs = cv.imread('rose.jpg', cv.IMREAD_GRAYSCALE)

# 显示图片
def show_img(opencv_img):
      image = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
      pixels = np.array(image)
      plt.imshow(pixels)
      plt.show() 

# Adding salt & pepper noise to an image
# 去噪
def salt_pepper(prob):
      # Extract image dimensions
      row, col = img_gs.shape

      # Declare salt & pepper noise ratio
      s_vs_p = 0.5
      output = np.copy(img_gs)

      # Apply salt noise on each pixel individually
      num_salt = np.ceil(prob * img_gs.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img_gs.shape]
      output[coords] = 1

      # Apply pepper noise on each pixel individually
      num_pepper = np.ceil(prob * img_gs.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img_gs.shape]
      output[coords] = 0
      # show_img(output)

      return output

# 中值滤波
def midpoint(img):
      maxf = maximum_filter(img, (3, 3))
      minf = minimum_filter(img, (3, 3))
      midpoint = (maxf + minf) / 2
      return midpoint
    

# 锐化滤波器
def ruihua(img):

      kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])
      sharpened_img = cv.filter2D(sp_05, -1, kernel_sharpening)
      return sharpened_img

# 逆谐波均值滤波器
def contraharmonic_mean(img, size, Q):
      num = np.power(img, Q + 1)
      denom = np.power(img, Q)
      kernel = np.full(size, 1.0)
      result = cv.filter2D(num, -1, kernel) / cv.filter2D(denom, -1, kernel)
      return result

# Canny算子
def canny(img_gs):
      plt.figure(figsize=(16, 16))
      # Apply canny edge detector algorithm on the image to find edges
      edges = cv.Canny(img_gs, 100,200)
      # Plot the original image against the edges
      plt.subplot(121), plt.imshow(img_gs)
      plt.title('Original Gray Scale Image')
      plt.subplot(122), plt.imshow(edges)
      plt.title('Edge Image')
      # Display the two images
      plt.show()
   


# 去噪
sp_05 = salt_pepper(0.5)
# cv.imwrite('mid_img.jpg', midpoint(sp_05))
# show_img(ruihua(sp_05))
# cv.imwrite('nixiebo.jpg', contraharmonic_mean(sp_05, (3,3), 0.5))
canny(sp_05)















# img = cv.imread('rose.jpg')
# print("- Number of Pixels: " + str(img.size))
# print("- Shape/Dimensions: " + str(img.shape))
# show_img(img)

# img_gs = cv.imread('rose.jpg', cv.IMREAD_GRAYSCALE)    # Convert image to grayscale
# show_img(img_gs)

# blue, green, red = cv.split(img)    # Split the image into its channels
# show_img(red) # Display the red channel in the image
# show_img(blue) # Display the red channel in the image
# show_img(green) # Display the red channel in the image

