import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image


def hole_filling(Il, labels, outlier):
    """
    There are two types of hole:
            boundary point: replaced by the pixel with lowest disparity is selected
            outlier:  replaced by the pixel with similar color is selected
    """
    h, w, ch = Il.shape
    all_dir = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]])
    num_dir = 4
    result = np.empty_like(labels)
    
    for y in range(h):
        for x in range(w):
            result[y, x] = labels[y, x]
            if outlier[y, x] != 0:
                min_distance = np.inf
                min_disp = -1
                for d in range(num_dir):
                    dir_y, dir_x = all_dir[d, 0], all_dir[d, 1]
                    verti, hori = y, x
                    while 0 <= verti < h and 0 <= hori < w and outlier[verti, hori] != 0:
                        verti += dir_y
                        hori += dir_x
                    if 0 <= verti < h and 0 <= hori < w:
                        assert(outlier[verti, hori] == 0)
                        if outlier[y, x] == 1:
                            curr_dist = max(abs(Il[y, x] - Il[verti, hori]))
                        else:
                            curr_dist = labels[verti, hori]

                        if curr_dist < min_distance:
                            min_distance = curr_dist
                            min_disp = labels[verti, hori]
                    result[y, x] = min_disp
    return result.astype(np.uint8)

def detect_outlier(labels, labels_r, max_disp):
    h, w = labels.shape
    outlier = np.empty_like(labels)
    # 0: matched point, 1: outlier, 2: boundary point
    threshold = 1.1
    for y in range(h):
        for x in range(w):
            if x - labels[y, x] < 0:
                outlier[y, x] = 2
            elif abs(labels[y, x] - labels_r[y, x - labels[y, x]]) < threshold:
                 outlier[y, x] = 0
            else:
                for disp in range(max_disp):
                    if x - disp > 0 and abs(disp - labels_r[y, x - disp]) < threshold:
                        outlier[y, x] = 1
                        break
                    else:
                        outlier[y, x] = 2
    return outlier

def compute_cost(Il, Ir, max_disp, cost_type='sd', left_fixed=True):
    h, w, ch = Il.shape
    assert(Il.shape == Ir.shape)
    cost = np.zeros((h, w, max_disp + 1), dtype=np.float32) # census cost
    
    if cost_type == 'sd':
        for disp in range(max_disp + 1):
            if left_fixed:
                padded_Ir = cv2.copyMakeBorder(Ir,  0,  0, disp,  0, cv2.BORDER_REFLECT)[0:h, 0:w]
                cost[:,:,disp] = np.sum((Il - padded_Ir)**2) / 3.
            else:
                padded_Il = cv2.copyMakeBorder(Il,  0,  0, 0,  disp, cv2.BORDER_REFLECT)[0:h, disp:disp+w]
                cost[:,:,disp] = np.sum((Ir - padded_Il)**2) / 3.
    
    elif cost_type == 'census': 
        w_s = 9 # window size
        l = w_s // 2 
        for disp in range(max_disp + 1):
            if left_fixed:
                padded_Il = cv2.copyMakeBorder(Il, l, l, l, l, cv2.BORDER_REFLECT)
                padded_Ir = cv2.copyMakeBorder(Ir, l, l, disp+l, l, cv2.BORDER_REFLECT)
                for y in range(l, l+h):
                    for x in range(l, l+w):
                            # left image window
                            c_census_l = padded_Il[y-l : y+l+1, x-l : x+l+1] > padded_Il[y, x]
                            # right image window
                            c_census_r = padded_Ir[y-l : y+l+1, x-l : x+l+1] > padded_Ir[y, x]
                            # Hamming distance calculation
                            assert(c_census_l.shape == c_census_r.shape)
                            c_census = float(np.sum(c_census_l ^ c_census_r)) / 3.
                            cost[y-l, x-l, disp] = c_census
            else:
                padded_Il = cv2.copyMakeBorder(Il, l, l, l, disp+l, cv2.BORDER_REFLECT)
                padded_Ir = cv2.copyMakeBorder(Ir, l, l, l, l, cv2.BORDER_REFLECT)
                for y in range(l, l+h):
                    for x in range(l, l+w):
                            # left image window
                            c_census_l = padded_Il[y-l : y+l+1, x+disp-l : x+disp+l+1] > padded_Il[y, x+disp]
                            # right image window
                            c_census_r = padded_Ir[y-l : y+l+1, x-l : x+l+1] > padded_Ir[y, x]
                            # Hamming distance calculation
                            assert(c_census_l.shape == c_census_r.shape)
                            c_census = float(np.sum(c_census_l ^ c_census_r)) / 3.
                            cost[y-l, x-l, disp] = c_census

    # >>> Cost aggregation
    for i in range(max_disp + 1):
        cost[:, :, i] = cv2.bilateralFilter(cost[:, :, i], 9, 75, 75)
    return cost


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    tic = time.time()
    # >>> Cost computation
    # >>> Cost aggregation
    # >>> Disparity optimization
    cost_l = compute_cost(Il, Ir, max_disp, 'census', True)
    labels = np.array([[np.argmin(cost_l[y, x]) for x in range(w)] for y in range(h)])
    
    # >>> Disparity refinement
    # ex: Left-right consistency check + hole filling + weighted median filtering
    cost_r = compute_cost(Il, Ir, max_disp, 'census', False)
    labels_r = np.array([[np.argmin(cost_r[y, x]) for x in range(w)] for y in range(h)])
    # Left-right consistency check 
    outlier = detect_outlier(labels, labels_r, max_disp)
    # hole filling
    labels = hole_filling(Il, labels, outlier)
    # weighted median filtering
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 17, 5,  cv2.ximgproc.WMF_JAC)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 11, 5,   cv2.ximgproc.WMF_JAC)
    labels = cv2.medianBlur(labels, 3)

    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    return labels.astype(np.uint8)