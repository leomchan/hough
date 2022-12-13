import cv2 as cv
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.breadth_first import BreadthFirstFinder

img = cv.imread("lego-path.jpeg")
img_copy = img.copy()

# Resize so longest edge is 256
longest_side = max(img.shape[0], img.shape[1])
resize_scale = 128.0 / longest_side
resized_shape = (round(img.shape[1] * resize_scale), round(img.shape[0] * resize_scale))
img_resized = cv.resize(img_copy, resized_shape, interpolation=cv.INTER_AREA)

# Smoothed
img_blurred = cv.medianBlur(img_resized, 5)

# Convert to HSV
# See https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
img_hls = cv.cvtColor(img_blurred, cv.COLOR_BGR2HLS)

# Resize to single pixel to calculate to mean color
img_mean_color = cv.resize(img_hls, (1, 1), interpolation=cv.INTER_AREA)


def min_elem(x, delta):
    return np.uint8(max(round(x - delta), 0))


def max_elem(x, delta):
    return np.uint8(min(round(x + delta), 255))


def min_hue(x, delta):
    lower = round(x - delta)
    while lower < 0:
        lower = lower + 255
    return np.uint8(lower)


def max_hue(x, delta):
    upper = round(x + delta)
    while upper > 255:
        upper = upper - 255
    return np.uint8(upper)


mean_color = img_mean_color[0][0]
print(mean_color)

# Create base plate mask
lowerb = np.array((min_hue(mean_color[0], 30), min_elem(mean_color[1], 80), min_elem(mean_color[2], 50)), np.uint8)
upperb = np.array((max_hue(mean_color[0], 30), max_elem(mean_color[1], 80), max_elem(mean_color[2], 50)), np.uint8)
print(lowerb, upperb)
img_baseplate_mask = cv.bitwise_not(cv.inRange(img_hls, lowerb, upperb))

print(img_baseplate_mask.shape)

# Find a path from start to finish
# (row, column) of start and finish
START = (112, 54)
FINISH = (14, 70)

grid = Grid(matrix=img_baseplate_mask)
start = grid.node(START[1], START[0])
finish = grid.node(FINISH[1], FINISH[0])

finder = BreadthFirstFinder(diagonal_movement=DiagonalMovement.never)
path, runs = finder.find_path(start, finish, grid)

print('operations:', runs, 'path length:', len(path))
print(grid.grid_str(path=path, start=start, end=finish))

cv.imshow('Original', img)
cv.imshow('Resized', img_resized)
cv.imshow('Blurred', img_blurred)
cv.imshow('HLS', img_hls)

img_mean_color = cv.resize(img_mean_color, resized_shape)
cv.imshow('Mean color', img_mean_color)
cv.imshow('Baseplate mask', img_baseplate_mask)

# cv.imshow('Grayscale', img_gray)


def click_data(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        hls = img_hls[y][x]
        in_range = (
            hls[0] >= lowerb[0] and hls[0] <= upperb[0]
            and hls[1] >= lowerb[1] and hls[1] <= upperb[1]
            and hls[2] >= lowerb[2] and hls[2] <= upperb[2]
        )
        print(x, ',', y, '=', hls, ',in range=', in_range)


cv.setMouseCallback('HLS', click_data)
cv.waitKey(0)
cv.destroyAllWindows()
