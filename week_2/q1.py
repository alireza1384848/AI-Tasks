
import cv2
import numpy as np
def sharpen_image(image, kernel_size=(5,5), gamma=1.5):
  x,y,z = image.shape
  bilateral_blur = cv2.resize(cv2.GaussianBlur(image,kernel_size,10), (y,x))
  diff = cv2.subtract(image , bilateral_blur)
  sharpened = cv2.addWeighted(image, 1.0, diff, gamma, 0)
  abstract_img = cv2.convertScaleAbs(sharpened)
  return (abstract_img,bilateral_blur)
    

image = cv2.imread('./Image/tiger.jpg')
if image is not None:
  sharpened_img, blur = sharpen_image(image)
  cv2.imshow("window",image)
  cv2.imshow("window1",sharpened_img)
  cv2.imshow("window2",blur)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
else:
  print("Error: Could not load image.")