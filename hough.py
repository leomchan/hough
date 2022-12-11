# Hough transform to identify circles

import numpy as np
import cv2

# Read image
img = cv2.imread('lego.png')

img_copy = img.copy()
# Smooth it
img_copy = cv2.medianBlur(img_copy, 3)

# Convert to greyscale
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

rows = img_gray.shape[0]

# Apply Hough transform to greyscale image
minDist = rows / 8
param1 = 100
param2 = 30  # smaller value-> more false circles
minRadius = 1
maxRadius = 30
circles = cv2.HoughCircles(
    img_gray, cv2.HOUGH_GRADIENT, 1, minDist,
    param1, param2, minRadius, maxRadius)
circles = np.uint16(np.around(circles))

# Draw the circles
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
