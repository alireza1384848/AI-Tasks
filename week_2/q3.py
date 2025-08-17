import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"D:\techstack2025-ai\week2\assets\bacteria.jpg", cv2.IMREAD_GRAYSCALE)
dx,dy= img.shape
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binary = cv2.bitwise_not(binary)

cv2kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (35,35))

Eroide = cv2.erode(binary , cv2kernel , iterations=1)
 
contours, _ = cv2.findContours(Eroide, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"number of Bacteria in Image : {len(contours)-1}")

Eroide = cv2.cvtColor(Eroide , cv2.COLOR_GRAY2BGR)
for cnt in contours:
    # Filter out too small areas (noise)
    area = cv2.contourArea(cnt)

    if area > 0:  # adjust threshold depending on your image
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(Eroide, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(Eroide , "*" ,(x + (w//2) , y + (h//2)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255),1)    



cv2.imshow("Counter",Eroide)
cv2.waitKey(0)
cv2.destroyAllWindows()