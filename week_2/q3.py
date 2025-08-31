# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread(r"./Image/bacteria.jpg", cv2.IMREAD_GRAYSCALE)
# dx,dy= img.shape
# _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# binary = cv2.bitwise_not(binary)

# # cv2kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (35,35))


# dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
# dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

# # Threshold peaks as markers
# _, markers = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
# markers = np.uint8(markers)

# # Connected components as markers
# num_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(markers)
# print(num_labels)

# # Watershed
# colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(colored, markers)


# print("Max marker value:", np.unique(markers))
# # Count unique regions (excluding background/borders)
# count = len(np.unique(markers))

# cv2.imshow("Counter",colored)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("Number of bacteria:", count)


# # Eroide = cv2.erode(binary , cv2kernel , iterations=1)
# # contours, _ = cv2.findContours(Eroide, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Eroide)

# # print(f"number of Bacteria in Image by contours : {len(contours)}")
# # print(f"number of Bacteria in Image by connected components : {num_labels - 1}")

# # Eroide = cv2.cvtColor(Eroide , cv2.COLOR_GRAY2BGR)
# # for cnt in contours:
# #     # Filter out too small areas (noise)
# #     area = cv2.contourArea(cnt)

# #     if area > 0:  # adjust threshold depending on your image
# #         x, y, w, h = cv2.boundingRect(cnt)
# #         cv2.rectangle(Eroide, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         cv2.putText(Eroide , "*" ,(x + (w//2) , y + (h//2)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255),1)    

# # for i in range(1, num_labels):
# #     x, y, w, h = stats[i , :4]
# #     cv2.rectangle(Eroide, (x, y), (x + w, y + h), (255, 0, 0), 1)
# #     cv2.putText(Eroide, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# # cv2.imshow("Counter",Eroide)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
import cv2
import numpy as np

# بارگذاری تصویر خاکستری
img = cv2.imread(r"./Image/bacteria.jpg", cv2.IMREAD_GRAYSCALE)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# باینری با آستانه اوتسو
_, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = cv2.bitwise_not(binary)

cv2.imshow("Binary Image", binary)
# Distance Transform + نرمال‌سازی
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

# Threshold روی Distance Transform
_, markers = cv2.threshold(dist, 0.45, 1.0, cv2.THRESH_BINARY)
markers = np.uint8(markers)   

num_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(markers)
# Watershed
colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(colored, markers)
print(num_labels)
print("Max marker value:", np.unique(markers))
# Count unique regions (excluding background/borders)
count = len(np.unique(markers))

print(np.unique(markers))

# رسم ستاره روی نقاطی که مقدارشان 1 است
for y in range(markers.shape[0]):
    for x in range(markers.shape[1]):
        if markers[y, x] >= 1:  # پیکسل‌های انتخاب شده
            cv2.putText(color_img, "g", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1)  # ستاره قرمز

cv2.imshow("Markers on Original", color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
