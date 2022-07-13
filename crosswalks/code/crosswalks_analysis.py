import glob
import cv2
import numpy as np
import argparse
import sys
from itertools import combinations, filterfalse
import evaluation as ev
import os
from matplotlib import pyplot as plt

FOLDER = '05'
FILENAME = '05_320.png'


# White Balancing of an image (gray world assumption)
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


# Function for computing Bird's Eye View
def bird_eye(img, numFolder=FOLDER):
    # clockwise order starting from top-left point
    if int(numFolder) in (5, 26):
        src = np.array([[890, 471], [1030, 471], [1919, 1079], [0, 1079]], dtype='float32')
    elif int(numFolder) == 6:
        src = np.array([[911, 562], [1009, 562], [1919, 1079], [0, 1079]], dtype='float32')
    elif int(numFolder) == 35:
        src = np.array([[853, 600], [964, 600], [1919, 1079], [0, 1079]], dtype='float32')
    else:
        sys.exit("Wrong number in bird eye computation")
    dst = np.array([[0, 0], [1919, 0], [1919, 6999], [0, 6999]], dtype='float32')
    homography_matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, homography_matrix, (1920, 7000), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    return result[3500:, :], homography_matrix


# Function for finding intersection points from hough lines
def hough_intersection(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.lstsq(A, b, rcond=None)[0]


# Function for compute distance car-crosswalk
def distance_car_crosswalk(stripe_width_pixel, y, numFolder=FOLDER):
    if int(numFolder) == 26:
        hood = 3135
    elif int(numFolder) == 5:
        hood = 3151
    elif int(numFolder) == 6:
        hood = 3311
    elif int(numFolder) == 35:
        hood = 3347
    else:
        hood = 0
    distance_pixel = hood - y
    if distance_pixel <= 0: return 0
    stripe_width_cm = 50
    return int((stripe_width_cm / stripe_width_pixel) * distance_pixel)


# Preprocessing of image: White Balancing -> Grayscale -> Bilateral Filter -> Canny -> Bird's Eye -> Erosion -> Thinning
def preprocessing(img, numFolder=FOLDER):
    # White balancing
    img_wb = white_balance(img)
    # From BGR to Gray
    gray = cv2.cvtColor(img_wb, cv2.COLOR_BGR2GRAY)
    # Bilateral filtering to smooth the image and maintain edges, parameters empirically chosen
    bil_gray = cv2.bilateralFilter(gray, 5, 80, 80)
    # Edge detection with Canny, parameters empirically chosen
    edges = cv2.Canny(bil_gray, 50, 125, L2gradient=True)
    # Bird's eye view of edges image
    bird_edges, _ = bird_eye(edges, numFolder)
    # Morphological operator: 2 iterations of Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erosion_bird_edges = cv2.erode(bird_edges, kernel, iterations=1)
    # Thinning
    thin_bird_edges = cv2.ximgproc.thinning(erosion_bird_edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Visualization
    images = [img, img_wb, bil_gray, edges, bird_edges, thin_bird_edges]
    titles = ['Original', 'White balanced', 'Bilateral filtering', 'Canny edge detection',
              'Bird\'s eye view', 'Erosion + Thinning']
    plt.subplot(331), plt.imshow(images[0][:, :, ::-1])
    plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(332), plt.imshow(images[1][:, :, ::-1])
    plt.title(titles[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(333), plt.imshow(images[2], 'gray')
    plt.title(titles[2]), plt.xticks([]), plt.yticks([])
    plt.subplot(334), plt.imshow(images[3], 'gray')
    plt.title(titles[3]), plt.xticks([]), plt.yticks([])
    plt.subplot(335), plt.imshow(images[4], 'gray')
    plt.title(titles[4]), plt.xticks([]), plt.yticks([])
    plt.subplot(336), plt.imshow(images[5], 'gray')
    plt.title(titles[5]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Return original image, gray one and bird's eye view of the edges image (thinned)
    return img, bil_gray, thin_bird_edges


# Detection of crosswalk: hough transform + check all possible quadrilaters + distance computation. Returns 1 if
# crosswalk, 0 otherwise
def detection(img, gray, edges, numFolder=FOLDER):
    img_bird, homography = bird_eye(img, numFolder)
    # copy_img_bird = np.copy(img_bird)
    gray_bird, _ = bird_eye(gray, numFolder)
    rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    copy_edges = np.copy(edges)

    # Hough Transform for finding lines and selection of horizontal and vertical lines
    lines = cv2.HoughLines(image=copy_edges, rho=1, theta=np.pi / 180, threshold=50, min_theta=0, max_theta=np.pi)
    if lines is None: return 0
    if lines.shape[0] < 6: return 0
    lines = lines.squeeze()
    horizontal_lines = []
    vertical_lines = []
    for i in range(len(lines)):
        angle = lines[i][1] * 180 / np.pi
        if 80 <= np.abs(angle) <= 100:
            horizontal_lines.append(tuple(lines[i]))
        elif 0 <= np.abs(angle) <= 10 or 170 <= np.abs(angle) <= 180:
            vertical_lines.append(tuple(lines[i]))
    if (len(horizontal_lines) <= 1) or (len(vertical_lines) < 4): return 0
    horizontal_lines = sorted(horizontal_lines, key=lambda y: np.abs(y[0]))
    vertical_lines = sorted(vertical_lines, key=lambda y: np.abs(y[0]))

    # For visualization in debugging
    before_cluster_rgb_edges = np.copy(rgb_edges)
    for line in horizontal_lines + vertical_lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
        pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
        if tuple(line) in vertical_lines:
            cv2.line(before_cluster_rgb_edges, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        elif tuple(line) in horizontal_lines:
            cv2.line(before_cluster_rgb_edges, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)

    # Clusterization of vertical and horizontal lines
    vertical_clusters = {}
    anchor = 0
    count = 1
    for i in range(len(vertical_lines)):
        if i == len(vertical_lines) - 1: break
        if i == 0:
            vertical_clusters[vertical_lines[i]] = count
            anchor = i
        if np.abs(np.abs(vertical_lines[anchor][0]) - np.abs(vertical_lines[i + 1][0])) > 100:
            count += 1
            anchor = i + 1
            vertical_clusters[vertical_lines[i + 1]] = count
        else:
            vertical_clusters[vertical_lines[i]] = count
            vertical_clusters[vertical_lines[i + 1]] = count
    horizontal_clusters = {}
    anchor = 0
    count = 1
    for i in range(len(horizontal_lines)):
        if i == len(horizontal_lines) - 1: break
        if i == 0:
            horizontal_clusters[horizontal_lines[i]] = count
            anchor = i
        if np.abs(np.abs(horizontal_lines[anchor][0]) - np.abs(horizontal_lines[i + 1][0])) > 450:
            count += 1
            anchor = i + 1
            horizontal_clusters[horizontal_lines[i + 1]] = count
        else:
            horizontal_clusters[horizontal_lines[i]] = count
            horizontal_clusters[horizontal_lines[i + 1]] = count

    # For each cluster we keep only the middle line
    single_horizontal_lines = []
    for i in range(1, max(horizontal_clusters.values()) + 1):
        keys_list = [k for k, v in horizontal_clusters.items() if v == i]
        single_horizontal_lines.append(keys_list[int(len(keys_list) / 2)])
    single_vertical_lines = []
    for i in range(1, max(vertical_clusters.values()) + 1):
        keys_list = [k for k, v in vertical_clusters.items() if v == i]
        single_vertical_lines.append(keys_list[int(len(keys_list) / 2)])

    if (len(single_horizontal_lines) <= 1) or (len(single_vertical_lines) < 4): return 0

    # Lines computation
    for line in single_vertical_lines + single_horizontal_lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
        pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
        if tuple(line) in single_horizontal_lines:
            cv2.line(rgb_edges, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
        elif tuple(line) in single_vertical_lines:
            cv2.line(rgb_edges, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    # Filter out too much isolated horizontal lines (for reducing intersection points -> performance reasons)
    h_lines = single_horizontal_lines.copy()
    for i in range(0, len(h_lines)):
        if i == len(h_lines) - 1: continue
        if h_lines[i + 1][0] - h_lines[i][0] >= 1400:
            single_horizontal_lines.pop(i)

    # Intersection points search
    intersection_points = []
    for horizontal in single_horizontal_lines:
        for vertical in single_vertical_lines:
            point = hough_intersection(horizontal[0], horizontal[1], vertical[0], vertical[1])
            if point is not None:
                intersection_points.append(tuple(np.around(point).astype(int)))
    if len(intersection_points) == 0: return 0
    intersection_points = sorted(intersection_points)
    for point in intersection_points:
        points_rgb_edges = cv2.circle(rgb_edges, point, radius=20, color=(0, 255, 255), thickness=-1)

    # Visualization
    images = [before_cluster_rgb_edges, points_rgb_edges]
    titles = ['Hough Transform', 'Lines reduction']
    plt.subplot(121), plt.imshow(images[0][:, :, ::-1])
    plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(images[1][:, :, ::-1])
    plt.title(titles[1]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # 4-combination for evaluate possible single stripes (that are quadrilaterals)
    combinations_list = list(combinations(intersection_points, 4))

    # Filter out quartets of points in which at least three are collinear
    def noCollinear(e):
        triplets = list(combinations(e, 3))
        for triplet in triplets:
            area = triplet[0][0] * (triplet[1][1] - triplet[2][1]) + triplet[1][0] * (triplet[2][1] - triplet[0][1]) + \
                   triplet[2][0] * (triplet[0][1] - triplet[1][1])
            if area == 0: return True

    quartets = list(filterfalse(noCollinear, combinations_list))

    # For each quartet check if it is a stripe: appropriate aspect ratio, not too large area and not too small, avg
    # stripe intensity higher (grayscale) of the avg outer intensity and enough difference between the two
    def quartets_filter(e):
        _, _, width, height = cv2.boundingRect(np.asarray(e))
        if height <= width: return True
        aspect_ratio = width / height
        if (aspect_ratio >= 0.4) or (aspect_ratio < 0.2): return True
        if height * width > 400000 or height * width < 50000: return True
        return False

    filtered_quartets = list(filterfalse(quartets_filter, quartets))
    combinations_list.clear()
    quartets.clear()
    stripes = np.zeros((4, 2), dtype=int)
    for i in range(len(filtered_quartets)):
        inner_mask = np.zeros_like(edges)
        l_x, t_y, width, height = cv2.boundingRect(np.asarray(filtered_quartets[i]))
        r_x = l_x + width
        b_y = t_y + height
        cv2.rectangle(inner_mask, (l_x, t_y), (r_x, b_y), color=255, thickness=-1)
        inner_locs = np.where(inner_mask == 255)
        inner_pixels = gray_bird[inner_locs]
        if len(inner_pixels) == 0: return 0
        inner_masked = cv2.bitwise_and(gray_bird, inner_mask)  # for visualization in debugging
        avg_inside_intensity = np.mean(inner_pixels, dtype='float64')
        if avg_inside_intensity < 128: continue
        outer_mask = np.zeros_like(edges)
        cv2.rectangle(outer_mask, (l_x - 100, t_y - 100), (r_x + 100, b_y + 100), color=255, thickness=-1)
        outer_mask -= inner_mask
        outer_locs = np.where(outer_mask == 255)
        outer_pixels = gray_bird[outer_locs]
        outer_masked = cv2.bitwise_and(gray_bird, outer_mask)  # for visualization in debugging
        avg_outside_intensity = np.mean(outer_pixels, dtype='float64')
        if avg_outside_intensity >= 128: continue
        if avg_inside_intensity > avg_outside_intensity:
            if np.abs(avg_inside_intensity - avg_outside_intensity) > 20:
                stripe = ((l_x, t_y), (r_x, t_y), (r_x, b_y), (l_x, b_y))
                stripes = np.dstack((stripes, np.asarray(stripe, dtype=int)))
    if len(stripes.shape) != 3: return 0
    stripes = np.delete(stripes, 0, axis=2)
    stripes = np.unique(stripes, axis=2)
    # At least 2 stripes detected to say we are in presence of a crosswalk
    if stripes.shape[2] < 2: return 0
    # stripe_width_pixel = stripes[1, 0, 0] - stripes[0, 0, 0]
    # Avg r_x - l_x
    stripe_width_pixel = int(np.mean(stripes[1, 0, :] - stripes[0, 0, :]))
    bb_tl = np.amin(stripes[0, :, :], axis=1)
    bb_br = np.amax(stripes[2, :, :], axis=1)
    y = bb_br[1]
    cv2.rectangle(img_bird, bb_tl, bb_br, (255, 0, 0), 30)
    # Inverse homography matrix computation to find the bounding box points for original image
    inv_homography = np.linalg.inv(homography)
    bb_tl = np.append(bb_tl, [1], axis=0)
    bb_tl[1] += 3500  # due to cropping of bird_eye function
    bb_br = np.append(bb_br, [1], axis=0)
    bb_br[1] += 3500  # due to cropping of bird_eye function
    tmp = inv_homography.dot(bb_tl)
    bb_tl = np.int0(np.around(tmp / tmp[2]))[0:2]
    tmp = inv_homography.dot(bb_br)
    bb_br = np.int0(np.around(tmp / tmp[2]))[0:2]
    cv2.rectangle(img, bb_tl, bb_br, (255, 0, 0), 2)
    cv2.putText(img, 'Crosswalk', (bb_tl[0], bb_tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Distance computation
    distance = distance_car_crosswalk(stripe_width_pixel, y, numFolder=numFolder)
    cv2.putText(img, '~ {}cm'.format(distance), (bb_tl[0], bb_br[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                2)
    os.makedirs('..\\dataset\\dreyeve\\{}\\found'.format(numFolder), exist_ok=True)
    ret = cv2.imwrite(filename='..\\dataset\\dreyeve\\{}\\found\\{}'.format(numFolder, FILENAME), img=img)
    if ret is False:
        print("Error in saving {}".format(FILENAME))
        return 0
    return 1


def crosswalks_analysis_dataset(numFolder):
    # imgs = [cv2.imread(file) for file in glob.glob("..\dataset\dreyeve\{}\*.png".format(numFolder))]
    for file in glob.glob("*.png", root_dir="..\dataset\dreyeve\{}\\".format(numFolder)):
        global FILENAME
        FILENAME = file
        print('Analyzing \'{}\''.format(FILENAME))
        im = cv2.imread("..\\dataset\\dreyeve\\{}\\".format(numFolder) + file)
        # Original image, gray one and bird's eye view of the edges image (thinned)
        img, gray, edges = preprocessing(im, numFolder)
        stripe = detection(img, gray, edges, numFolder)
        with open("..\\dataset\\dreyeve\\{}\\{}_pred.txt".format(numFolder, numFolder), "at") as f:
            print('Writing \'{},{}\''.format(FILENAME, str(stripe)))
            f.write(file + ',' + str(stripe) + '\n')


def crosswalks_analysis(im):
    img, gray, edges = preprocessing(im)
    stripe = detection(img, gray, edges)
    print("Detection: " + str(stripe))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--numFolder", help="number of folder: 05, 06, 26, or 35", required=False)
    args = a.parse_args()
    if args.numFolder is None:
        print('Analyzing \'{}\''.format(FILENAME))
        img = cv2.imread("..\\dataset\\dreyeve\\{}\\{}".format(FOLDER, FILENAME))
        crosswalks_analysis(img)
    elif int(args.numFolder) not in (5, 6, 26, 35):
        print(args)
        sys.exit("Wrong number")
    else:
        crosswalks_analysis_dataset(args.numFolder)
        ev.evaluation(args.numFolder)


if __name__ == '__main__':
    main()
