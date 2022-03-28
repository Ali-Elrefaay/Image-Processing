import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('a.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(img,100,200)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)


plt.figure()
plt.title('a.png')
plt.imsave('a.png',img , cmap='gray', format='png')
plt.imshow(img, cmap='gray')
#plt.imshow()
plt.figure()
plt.title('a-sobel.png')
plt.imsave('a-sobel.png',img_sobel , cmap='gray', format='png')
plt.imshow(img_sobel, cmap='gray')
#plt.imshow()
plt.figure()
plt.title('a-canny.png')
plt.imsave('a-canny.png',img_canny , cmap='gray', format='png')
plt.imshow(img_canny, cmap='gray')
#plt.imshow()
plt.figure()
plt.title('a-sobelx.png')
plt.imsave('a-sobelx.png',img_sobelx , cmap='gray', format='png')
plt.imshow(img_sobelx, cmap='gray')
#plt.imshow()
plt.figure()
plt.title('a-prewitt.png')
plt.imsave('a-prewitt.png',img_prewittx + img_prewitty , cmap='gray', format='png')
plt.imshow(img_prewittx + img_prewitty, cmap='gray')
#plt.imshow()