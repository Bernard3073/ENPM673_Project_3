import cv2
import numpy as np
import matplotlib.pyplot as plt

K1 = [5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1 ]
K2 = [5299.313, 0, 1438.004, 0, 5299.313, 977.763, 0, 0, 1 ]
K1 = np.reshape(K1, (3, 3))
K2 = np.reshape(K2, (3, 3))

def fundamental_matrix(feat_1,feat_2):
    
    #compute the centroids
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])
    
    #Recenter the coordinates by subtracting mean
    feat_1[:,0] = feat_1[:,0] - feat_1_mean_x
    feat_1[:,1] = feat_1[:,1] - feat_1_mean_y
    feat_2[:,0] = feat_2[:,0] - feat_2_mean_x
    feat_2[:,1] = feat_2[:,1] - feat_2_mean_y
    
        
    
    #Compute the scaling terms which are the average distances from origin
    s_1 = np.sqrt(2.)/np.mean(np.sum((feat_1)**2,axis=1)**(1/2))
    s_2 = np.sqrt(2.)/np.mean(np.sum((feat_2)**2,axis=1)**(1/2))
    
     
    #Calculate the transformation matrices
    T_a_1 = np.array([[s_1,0,0],[0,s_1,0],[0,0,1]])
    T_a_2 = np.array([[1,0,-feat_1_mean_x],[0,1,-feat_1_mean_y],[0,0,1]])
    T_a = T_a_1 @ T_a_2
    
    
    T_b_1 = np.array([[s_2,0,0],[0,s_2,0],[0,0,1]])
    T_b_2 = np.array([[1,0,-feat_2_mean_x],[0,1,-feat_2_mean_y],[0,0,1]])
    T_b = T_b_1 @ T_b_2
    

    #Compute the normalized point correspondences
    x1 = ( feat_1[:, 0].reshape((-1,1)))*s_1
    y1 = ( feat_1[:, 1].reshape((-1,1)))*s_1
    x2 = (feat_2[:, 0].reshape((-1,1)))*s_2
    y2 = (feat_2[:, 1].reshape((-1,1)))*s_2
    
    #-point Hartley
    #A is (8x9) matrix
    A = np.hstack((x2*x1, x2*y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1),1))))
        
        
    #Solve for A using SVD
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    V = V.T
    
    #last col - soln
    sol = V[:,-1]
    F = sol.reshape((3,3))
    U_F, S_F, V_F = np.linalg.svd(F)
    
    #Rank-2 constraint
    S_F[2] = 0
    S_new = np.diag(S_F)
    
    #Recompute normalized F
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm/F_norm[-1,-1]
    return F_norm


def estimate_fundamental_matrix(feature_1, feature_2):
    threshold = 0.5
    max_num_inliers = 0
    F_matrix_best = []
    p = 0.99
    N = np.inf
    count = 0
    while count < N:
        inliers_count = 0
        feature_1_rand = []
        feature_2_rand = []
        random = np.random.randint(len(feature_1), size=8)
        for i in random:
            feature_1_rand.append(feature_1[i])
            feature_2_rand.append(feature_2[i])
        F = fundamental_matrix(np.array(feature_1_rand), np.array(feature_2_rand))
        ones = np.ones((len(feature_1),1))
        x1 = np.hstack((feature_1,ones))
        x2 = np.hstack((feature_2,ones))
        e1, e2 = x1 @ F.T, x2 @ F
        error = np.sum(e2* x1, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
        inliers = error <= threshold
        inliers_count = np.sum(inliers)
        
        if inliers_count > max_num_inliers:
            max_num_inliers = inliers_count
            F_matrix_best = F 
        #Iterations to run the RANSAC for every frame
        inlier_ratio = inliers_count/len(feature_1)
        
        if np.log(1-(inlier_ratio**8)) == 0: 
            continue
        
        N = np.log(1-p)/np.log(1-(inlier_ratio**8))
        count += 1
    return F_matrix_best


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


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def display_image(img1, img2):
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Given Projection matrix and point correspondences, estimate 3-D point
def point_3d(pt,pt_,R2,C2,K):
    #Find the projection matrices for respective frames
    C1 = [[0],[0],[0]]
    R1 = np.identity(3)
    R1C1 = -R1@C1
    R2C2 = -R2@C2
    #Current frame has no Rotation and Translation
    P1 = K @ np.hstack((R1, R1C1))
    
    #Estimate the projection matrix for second frame using returned R and T values
    P2 = K @ np.hstack((R2, R2C2))
    #P1_T = P1.T
    #P2_T = P2.T	
    X = []
    
    #Solve linear system of equations using cross-product technique, estimate X using least squares technique
    for i in range(len(pt)):
        x1 = pt[i]
        x2 = pt_[i]
        A1 = x1[0]*P1[2,:]-P1[0,:]
        A2 = x1[1]*P1[2,:]-P1[1,:]
        A3 = x2[0]*P2[2,:]-P2[0,:]
        A4 = x2[1]*P2[2,:]-P2[1,:]		
        A = [A1, A2, A3, A4]
        
        # A1 = x1[1]*P1[2,:] - P1[0,:]
        # A2 = P1[0:,] - x1[0]*P1[2,:]
        # A3 = x2[1]*P2[2:,] - P2[0:,]
        # A4 = P2[0:,] - x1[0]*P2[2:,]
        # A = np.vstack([A1,A2,A3,A4])
        U,S,V = np.linalg.svd(A)
        V = V[3]
        V = V/V[-1]
        X.append(V)
    return X

#cheirality condition
def linear_triangulation(pt,pt_, R,C,K):
    #Check if the reconstructed points are in front of the cameras using cheilarity equations
    X1 = point_3d(pt,pt_,R,C,K)
    X1 = np.array(X1)	
    count = 0
    #r3(X-C)>0
    for i in range(X1.shape[0]):
        x = X1[i,:].reshape(-1,1)
        if R[2]@np.subtract(x[0:3],C) > 0 and x[2] > 0: 
            count += 1
    return count


def main():
    img1 = cv2.imread('./Dataset 1/im0.png')
    img2 = cv2.imread('./Dataset 1/im1.png')
                            
    h1, w1, ch1 = img1.shape
    h2, w2, ch2 = img2.shape
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()  # opencv-contrib-python required
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    best_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            best_matches.append([m])
            
    # # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()

    # feature_1 = [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in good]
    # feature_2 = [[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1] ] for m in good]
    
    # F_matrix = fundamental_matrix(feature_1, feature_2)   
    # print(F_matrix)
    
    # matchesMask = [[0, 0] for i in range(len(matches))]

    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)
        
    Best_F_matrix = estimate_fundamental_matrix(feature_1, feature_2)
    print(Best_F_matrix)
    
    E_matrix = essential_matrix(Best_F_matrix, K1)
    R, T = extract_camera_pose(E_matrix, K1)
    print(R)
    print(T)
    H = []
    I = np.array([0, 0, 0, 1])
    for i, j in zip(R, T):
        h = np.hstack((i, j))
        h = np.vstack((h, I))
        H.append(h)
    print('H:\n', H)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(feature_1), np.float32(feature_2), Best_F_matrix, imgSize=(w1, h1))
    print("H1:\n", H1)
    print("H2:\n",H2)
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    # cv2.imwrite("rectified_1.png", img1_rectified)
    # cv2.imwrite("rectified_2.png", img2_rectified)
    distances = {}
    for j in range(len(feature_1)):
        new_list = np.array([feature_1[j][0], feature_2[j][1], 1])
        new_list = np.reshape(new_list, (3, 1))
        new_list_2 = np.array([feature_1[j][0], feature_2[j][1], 1])
        new_list_2 = np.reshape(new_list_2, (1, 3))
        distances[j] = abs(new_list_2 @ Best_F_matrix @ new_list)

    distances_sorted = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    
    distances_list = []
    v_list = []
    for k,v in distances_sorted.items():
        #print(v[0][0])
        if v[0][0]<0.05: #threshold distance
            distances_list.append(v[0][0])
            v_list.append(k)

    
    len_distance_list = len(distances_list)

    len_distance_list = min(len_distance_list,30) #30 might have to be changed
    
    inliers_1 = []
    inliers_2 = []
    
    for x in range(len_distance_list):
        inliers_1.append(feature_1[v_list[x]])
        inliers_2.append(feature_2[v_list[x]])
    
    
    # for pts in inliers_1:
    #     int_x = int(pts[0])
    #     int_y = int(pts[1])
    #     pts2 = (int_x,int_y)
    #     cv2.circle(img1_rectified,pts2,5,[0,0,255],1)
    
    # for pts in inliers_2:
    #     int_x = int(pts[0])
    #     int_y = int(pts[1])
    #     pts2 = (int_x, int_y)
    #     cv2.circle(img2_rectified,pts2,5,[0,0,255],1)
        

    # Draw the rectified images
    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # axes[0].imshow(img1_rectified, cmap="gray")
    # axes[1].imshow(img2_rectified, cmap="gray")
    # axes[0].axhline(250)
    # axes[1].axhline(250)
    # axes[0].axhline(450)
    # axes[1].axhline(450)   
    # plt.suptitle("Rectified images")
    # plt.savefig("rectified_images.png")
    
    img3 = cv2.drawMatchesKnn(img1_rectified, kp1, img2_rectified, kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    cv2.imshow('t', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # feature_1 = np.int32(feature_1)
    # feature_2 = np.int32(feature_2)
    # F, mask = cv2.findFundamentalMat(feature_1, feature_2, cv2.FM_LMEDS)
    # print(mask)
    # feature_1 = feature_1[mask.ravel() == 1]
    # feature_2 = feature_2[mask.ravel() == 1]


if __name__ == '__main__':
    main()

# def fundamental_matrix(feature_1, feature_2):

#     # Compute the centroid of all corresponding points in a single image:
#     feature_1_mean_x, feature_1_mean_y  = np.mean(feature_1, axis=0)
#     feature_2_mean_x, feature_2_mean_y = np.mean(feature_2, axis=0)

#     # Recenter by subtracting the mean
#     for i, j in zip(feature_1, feature_2):
#         i[0] -= feature_1_mean_x
#         i[1] -= feature_1_mean_y
#         j[0] -= feature_2_mean_x
#         j[1] -= feature_2_mean_y


#     # Define the scale terms s_1 and s_2 to be the average distances of the centered points from the origin
#     s_1 = np.sqrt(2)/np.mean(np.sum(np.square(feature_1), axis=1) ** 0.5)
#     s_2 = np.sqrt(2)/np.mean(np.sum(np.square(feature_2), axis=1) ** 0.5)


#     # Construct the transformation matrices T_a and T_b
#     T_a_1 = np.array([[s_1,0,0],[0,s_1,0],[0,0,1]])
#     T_a_2 = np.array([[1,0,-feature_1_mean_x],[0,1,-feature_1_mean_y],[0,0,1]])
#     T_a = T_a_1 @ T_a_2

#     T_b_1 = np.array([[s_2,0,0],[0,s_2,0],[0,0,1]])
#     T_b_2 = np.array([[1,0,-feature_2_mean_x],[0,1,-feature_2_mean_y],[0,0,1]])
#     T_b = T_b_1 @ T_b_2


#     # Compute the normalized correspondence points

#     # x1, y1, x2, y2 = [], [], [], []
#     x1, x2 = [], []
#     for i, j in zip(feature_1, feature_2):
#         x1.append(T_a @ np.array([i[0], i[1], 1]).T)
#         x2.append(T_b @ np.array([j[0], j[1], 1]).T)
#         # x1.append(i[0] * s_1)
#         # y1.append(i[1] * s_1)
#         # x2.append(j[0] * s_2)
#         # y2.append(j[1] * s_2)

#     # Solve for the fundamental matrix by applying the 8 point algorithm
#     # A is (8x9) matrix
#     # A = np.hstack((x2*x1, x2*y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1),1))))
#     A = np.ones((len(x1), 9))
#     for i in range(len(x1)):
#         A[i] = [x2[i][0] * x1[i][0], x2[i][0] * x1[i][1], x2[i][0],
#                 x2[i][1] * x1[i][0], x2[i][1] * x1[i][1], x2[i][1],
#                 x1[i][0], x1[i][1], 1]

#     # Solve for A using SVD
#     A = np.array(A)
#     U,S,V = np.linalg.svd(A)
#     V = V.T

#     # last col - soln
#     sol = V[:, -1]
#     F = sol.reshape((3, 3))
#     U_F, S_F, V_F = np.linalg.svd(F)

#     # Rank-2 constraint
#     S_F[2] = 0
#     S_new = np.diag(S_F)

#     # Recompute normalized F
#     F_new = U_F @ S_new @ V_F
#     F_norm = T_b.T @ F_new @ T_a
#     F_norm = F_norm/F_norm[-1, -1]
#     return F_norm