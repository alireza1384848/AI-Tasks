import cv2
import numpy as np
import matplotlib.pyplot as plt



# Read the image
img = cv2.imread(r"D:\techstack2025-ai\week2\assets\bacteria.jpg", cv2.IMREAD_GRAYSCALE)
# For black and white image:
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (2,2))


dilate = cv2.dilate(binary , cv2kernel , iterations=1)

Eroded = cv2.erode(binary,cv2kernel , iterations= 1)

diff = cv2.subtract(dilate,Eroded)

cv2.imshow("Cleaned Edges", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()