from typing import List

import cv2
import numpy as np

MIN_THRESHOLD = 150
SQUARE_THRESHOLD = 10
Y_MARGIN_PERC = 20


def is_square(first: List[int, int], second: List[int, int]):
    return abs(abs(second[0] - first[0]) - abs(second[1] - first[1])) < SQUARE_THRESHOLD


def side_length(first: List[int, int], second: List[int, int]):
    return abs(first[1] - second[1])


def is_located_middle_upper_y(image: np.ndarray, ypoint: List[int, int]):
    heigth = image.shape[0]
    margin = heigth * Y_MARGIN_PERC / 100

    return ypoint[1] < heigth - margin


def get_squares(file_path) -> List[np.ndarray]:
    image = cv2.imread(file_path)
    squares = []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold with 150 as minumim value -> one of the most critical point
    _, thresh = cv2.threshold(gray, MIN_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Opening is just another name of erosion followed by dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Median blur filter
    blur = cv2.medianBlur(close, 5)

    # Second parameter is the hierarchy, which is deleted
    contours, _ = cv2.findContours(blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # findcontour detects whole image as shape
        if i == 0:
            continue

        # cv2.approxPloyDP() function to approximate the shape
        # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            first, second, third, _ = box

            if is_square(first, third) and side_length(first, second) > 5 and is_located_middle_upper_y(image, first):
                squares.append(image[first[1]:second[1], first[0]:third[0]])

    return squares
