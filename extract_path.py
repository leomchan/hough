import cv2 as cv

img = cv.imread("lego-path.jpeg")
img_copy = img.copy()

# Resize so longest edge is 256
longest_side = max(img.shape[0], img.shape[1])
resize_scale = 256 / longest_side
resized_shape = (round(img.shape[1] * resize_scale), round(img.shape[0] * resize_scale))
img_resized = cv.resize(img_copy, resized_shape, interpolation=cv.INTER_AREA)

# Smoothed
img_blurred = cv.medianBlur(img_resized, 5)

# Resize to single pixel to calculate to mean color
img_mean_color = cv.resize(img_blurred, (1, 1), interpolation=cv.INTER_AREA)
img_mean_color = cv.resize(img_mean_color, resized_shape)


# Convert image to gray and blur it
img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)
img_gray = cv.blur(img_gray, (3, 3))

cv.imshow('Original', img)
cv.imshow('Resized', img_resized)
cv.imshow('Blurred', img_blurred)
cv.imshow('Mean color', img_mean_color)
cv.imshow('Grayscale', img_gray)

cv.waitKey(0)
cv.destroyAllWindows()
