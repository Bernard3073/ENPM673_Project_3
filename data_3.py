#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:44:30 2021

@author: bernard
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

K1 = [5806.559, 0, 1429.219, 0, 5806.559, 993.403, 0, 0, 1 ]
K2 = [5806.559, 0, 1543.51, 0, 5806.559, 993.403, 0, 0, 1 ]
K1 = np.reshape(K1, (3, 3))
K2 = np.reshape(K2, (3, 3))
B = 174.019
f = 5806.559
BLOCK_SIZE = 31
SEARCH_BLOCK_SIZE = 10 # 56
vmin = 38
vmax = 222
def fundamental_matrix(feature_1, feature_2):
    # Compute the centroid of all corresponding points in a single image
    feature_1_mean_x = np.mean(feature_1[:, 0])
    feature_1_mean_y = np.mean(feature_1[:, 1])
    feature_2_mean_x = np.mean(feature_2[:, 0])
    feature_2_mean_y = np.mean(feature_2[:, 1])

    # Recenter the coordinates by subtracting themean
    feature_1[:, 0] = feature_1[:, 0] - feature_1_mean_x
    feature_1[:, 1] = feature_1[:, 1] - feature_1_mean_y
    feature_2[:, 0] = feature_2[:, 0] - feature_2_mean_x
    feature_2[:, 1] = feature_2[:, 1] - feature_2_mean_y

    # Define the scale terms s_1 and s_2 to be the average distances of the centered points from the origin
    s_1 = np.sqrt(2.) / np.mean(np.sum(feature_1 ** 2, axis=1) ** 0.5)
    s_2 = np.sqrt(2.) / np.mean(np.sum(feature_2 ** 2, axis=1) ** 0.5)

    # Construct the transformation matrices T_a and T_b
    T_a_1 = np.array([[s_1, 0, 0], [0, s_1, 0], [0, 0, 1]])
    T_a_2 = np.array([[1, 0, -feature_1_mean_x], [0, 1, -feature_1_mean_y], [0, 0, 1]])
    T_a = T_a_1 @ T_a_2

    T_b_1 = np.array([[s_2, 0, 0], [0, s_2, 0], [0, 0, 1]])
    T_b_2 = np.array([[1, 0, -feature_2_mean_x], [0, 1, -feature_2_mean_y], [0, 0, 1]])
    T_b = T_b_1 @ T_b_2

    # Compute the normalized correspondence points
    x1 = (feature_1[:, 0].reshape((-1, 1))) * s_1
    y1 = (feature_1[:, 1].reshape((-1, 1))) * s_1
    x2 = (feature_2[:, 0].reshape((-1, 1))) * s_2
    y2 = (feature_2[:, 1].reshape((-1, 1))) * s_2

    # Solve for the fundamental matrix by applying the 8 point algorithm
    # A is (8x9) matrix
    A = np.hstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1), 1))))

    # Solve for A using SVD
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    V = V.T
    # last col = solution
    sol = V[:, -1]
    F = sol.reshape((3, 3))
    U_F, S_F, V_F = np.linalg.svd(F)

    # Rank 2 constraint
    S_F[2] = 0
    S_new = np.diag(S_F)

    # Recompute normalized F
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm / F_norm[-1, -1]
    return F_norm


def estimate_fundamental_matrix(feature_1, feature_2):
    threshold = 0.5
    max_num_inliers = 0
    F_matrix_best = []
    p = 0.99
    N = np.inf
    count = 0
    while count < N:
        inlier_count = 0
        feature_1_rand = []
        feature_2_rand = []
        feature_1_temp = []
        feature_2_temp = []
        random = np.random.randint(len(feature_1), size=8)
        for i in random:
            feature_1_rand.append(feature_1[i])
            feature_2_rand.append(feature_2[i])
        F = fundamental_matrix(np.array(feature_1_rand), np.array(feature_2_rand))
        ones = np.ones((len(feature_1), 1))
        x1 = np.hstack((feature_1, ones))
        x2 = np.hstack((feature_2, ones))
        e1, e2 = x1 @ F.T, x2 @ F
        error = np.sum(e2 * x1, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((e1[:, :-1], e2[:, :-1])) ** 2, axis=1, keepdims=True)
        inliers = error <= threshold
        inlier_count = np.sum(inliers)

        for i in range(len(inliers)):
            if inliers[i] == True:
                feature_1_temp.append(feature_1[i])
                feature_2_temp.append(feature_2[i])

        # for i in range(len(feature_1)):
        #     a = np.array([feature_1[i][0], feature_2[i][1], 1]).reshape(3, 1)
        #     b = np.array([feature_1[i][0], feature_2[i][1], 1]).reshape(1, 3)
        #     if abs(b @ F @ a) <= threshold:
        #         inlier_count += 1
        #         # test += 1
        #         feature_1_temp.append(feature_1[i])
        #         feature_2_temp.append(feature_2[i])

        if inlier_count > max_num_inliers:
            max_num_inliers = inlier_count
            F_matrix_best = F
            inlier_1 = feature_1_temp
            inlier_2 = feature_2_temp

        inlier_ratio = inlier_count / len(feature_1)

        if np.log(1 - (inlier_ratio ** 8)) == 0:
            continue

        N = np.log(1 - p) / np.log(1 - (inlier_ratio ** 8))
        count += 1
    return F_matrix_best, inlier_1, inlier_2


def essential_matrix(F, K):
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    # Due to the noise in K, the singular values of E are not necessarily (1, 1, 0)
    # This can be corrected by reconstructing it with (1, 1, 0) singular values
    S[0] = 1
    S[1] = 1
    S[2] = 0
    S_new = np.diag(S)
    E_new = U @ S_new @ V

    return E_new


def extract_camera_pose(E, K):
    U, D, V = np.linalg.svd(E)
    V = V.T
    W = np.reshape([0, -1, 0, 1, 0, 0, 0, 0, 1], (3, 3))
    C_1 = U[:, 2]
    R_1 = U @ W @ V.T
    C_2 = -U[:, 2]
    R_2 = U @ W @ V.T
    C_3 = U[:, 2]
    R_3 = U @ W.T @ V.T
    C_4 = -U[:, 2]
    R_4 = U @ W.T @ V.T

    if np.linalg.det(R_1) < 0:
        R_1 = -R_1
        C_1 = -C_1
    if np.linalg.det(R_2) < 0:
        R_2 = -R_2
        C_2 = -C_2
    if np.linalg.det(R_3) < 0:
        R_3 = -R_3
        C_3 = -C_3
    if np.linalg.det(R_4) < 0:
        R_4 = -R_4
        C_4 = -C_4

    C_1 = C_1.reshape((3, 1))
    C_2 = C_2.reshape((3, 1))
    C_3 = C_3.reshape((3, 1))
    C_4 = C_4.reshape((3, 1))

    return [R_1, R_2, R_3, R_4], [C_1, C_2, C_3, C_4]


def display_image(img1, img2):
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_feature(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()  # opencv-contrib-python required
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    return kp1, kp2, good


def drawlines(img1, img2, lines, pts1, pts2):
    """
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
    """
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_line = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1_circle = cv2.circle(img1_line, tuple(pt1), 5, color, -1)
        img2_circle = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1_circle, img2_circle


def sum_squared_distance(img1_pixel, img2_pixel):
    if img1_pixel.shape != img2_pixel.shape:
        return -1

    return np.sum((img1_pixel - img2_pixel) ** 2)


def compare_blocks(y, x, block_left, right_array, block_size):
    # Get search range for the right image
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    min_ssd = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y + block_size,
                      x: x + block_size]

        ssd = sum_squared_distance(block_left, block_right)

        if min_ssd:
            if ssd < min_ssd:
                min_ssd = ssd
                min_index = (y, x)
        else:
            min_ssd = ssd
            min_index = (y, x)

    return min_index


def get_disparity_map(img1, img2):
    left_array = np.asarray(img1)
    right_array = np.asarray(img2)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        raise Exception("Left-Right image shape mismatch!")
    h, w, _ = left_array.shape
    disparity_map = np.zeros((h, w))
    for y in range(BLOCK_SIZE, h - BLOCK_SIZE):
        for x in range(BLOCK_SIZE, w - BLOCK_SIZE):
            block_left = left_array[y: y + BLOCK_SIZE,
                         x: x + BLOCK_SIZE]
            min_index = compare_blocks(y, x, block_left,
                                       right_array,
                                       block_size=BLOCK_SIZE)
            disparity_map[y, x] = abs(min_index[1] - x)

    return disparity_map


def get_depth_map(disparity_map_gray):
    h, w = disparity_map_gray.shape
    depth_map = np.zeros_like(disparity_map_gray)
    for y in range(h):
        for x in range(w):
            if disparity_map_gray[y, x] == 0:
                depth_map[y, x] = 0
            else:
                depth_map[y, x] = int(B * f / (disparity_map_gray[y, x]))

    return depth_map


def rectification(img1, img2):
    img1_copy = img1
    img2_copy = img2
    kp1_warp, kp2_warp, good_warp = detect_feature(img1_copy, img2_copy)
    feature_1_warp = []
    feature_2_warp = []

    for i, match in enumerate(good_warp):
        feature_1_warp.append(kp1_warp[match.queryIdx].pt)
        feature_2_warp.append(kp2_warp[match.trainIdx].pt)

    feature_1_warp = np.int32(feature_1_warp)
    feature_2_warp = np.int32(feature_2_warp)

    F_warp, mask = cv2.findFundamentalMat(feature_1_warp, feature_2_warp, cv2.FM_LMEDS)
    # We select only inlier points
    feature_1_warp = feature_1_warp[mask.ravel() == 1]
    feature_2_warp = feature_2_warp[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(feature_2_warp.reshape(-1, 1, 2), 2, F_warp)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_copy, img2_copy, lines1, feature_1_warp, feature_2_warp)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(feature_1_warp.reshape(-1, 1, 2), 1, F_warp)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_copy, img1_copy, lines2, feature_2_warp, feature_1_warp)

    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.suptitle("Epilines in both images")
    # plt.show()

    return img5, img3

def main():
    img1 = cv2.imread('./Dataset 3/im0.png')
    img2 = cv2.imread('./Dataset 3/im1.png')
    
    # scale_percent = 10 # percent of original size
    # width = int(img1.shape[1] * scale_percent / 100)
    # height = int(img1.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    # img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    
    h1, w1, ch1 = img1.shape
    h2, w2, ch2 = img2.shape

    kp1, kp2, good = detect_feature(img1, img2)
    
    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)
    
    # Only for testing the result
    pts1 = np.int32(feature_1)
    pts2 = np.int32(feature_2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    
    
    Best_F_matrix, new_feature_1, new_feature_2 = estimate_fundamental_matrix(feature_1, feature_2)
    # print(Best_F_matrix)
    
    new_feature_1 = np.int32(new_feature_1)
    new_feature_2 = np.int32(new_feature_2)
    
    E_matrix = essential_matrix(Best_F_matrix, K1)
    R, T = extract_camera_pose(E_matrix, K1)
    # print(R)
    # print(T)
    H = []
    I = np.array([0, 0, 0, 1])
    for i, j in zip(R, T):
        h = np.hstack((i, j))
        h = np.vstack((h, I))
        H.append(h)
    # print('H:\n', H)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(new_feature_1), np.float32(new_feature_2), Best_F_matrix, imgSize=(w1, h1))
    # print("H1:\n", H1)
    # print("H2:\n",H2)
    
    # Store the output data in the text file
    file = open('./dataset3_output_data.txt', 'w')
    file.write('*'*50 + '\n' + '(Acquired by the inbuilt function) Foundatmental matrix' + '\n')
    file.write(str(F) + '\n')
    file.write('*'*50 + '\n' + 'Estimated foundatmental matrix' + '\n')
    file.write(str(Best_F_matrix) + '\n')
    file.write('*'*50 + '\n' + 'Essential matrix' + '\n')
    file.write(str(E_matrix) + '\n')
    file.write('*'*50 + '\n' + 'Rotation vector' + '\n')
    file.write(str(R) + '\n')
    file.write('*'*50 + '\n' + 'Translation vector' + '\n')
    file.write(str(T) + '\n')
    file.write('*'*50 + '\n' + 'Homography matrix (H1)' + '\n')
    file.write(str(H) + '\n')
    file.write('*'*50 + '\n' + 'Homography matrix img1' + '\n')
    file.write(str(H1) + '\n')
    file.write('*'*50 + '\n' + 'Homography matrix img2' + '\n')
    file.write(str(H2) + '\n')
    file.write('*'*50 + '\n')
    file.close()
    
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    # display_image(img1, img2)
    
    
    img1_res, img2_res = rectification(img1_rectified, img2_rectified)
    
    res = np.concatenate((img1_res, img2_res), axis = 1)
    cv2.imshow("Rectification", res)

    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2)) 
    
    disparity_map = get_disparity_map(img1_rectified, img2_rectified)
    
    disparity_map_gray = None
    disparity_map_gray = cv2.normalize(disparity_map, disparity_map_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disparity_gray', disparity_map_gray)
    
    depth_map = get_depth_map(disparity_map_gray)
    
    disparity_map_heat = None
    disparity_map_heat = cv2.normalize(disparity_map, disparity_map_heat, alpha=vmin, beta=vmax, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_map_heat = cv2.applyColorMap(disparity_map_heat, cv2.COLORMAP_JET)
    cv2.imshow("disparity_heat", disparity_map_heat)
    
    depth_map_gray = None
    depth_map_gray = cv2.normalize(depth_map, depth_map_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('depth_gray', depth_map_gray)
    
    
    depth_map_heat = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_JET)
    cv2.imshow("depth_heat", depth_map_heat)
    
    cv2.imwrite('./output/data_3_Rectification.jpg', res)
    cv2.imwrite('./output/data_3_disparity_gray.jpg', disparity_map_gray)
    cv2.imwrite('./output/data_3_disparity_heat.jpg', disparity_map_heat)
    cv2.imwrite('./output/data_3_depth_gray.jpg', depth_map)
    cv2.imwrite('./output/data_3_depth_heat.jpg', depth_map_heat)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()