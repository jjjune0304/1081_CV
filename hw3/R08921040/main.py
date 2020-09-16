import numpy as np
import time
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    #A = np.zeros((2*N, 8))
	# if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    # sol 2:
    for i in range(N):
        r = 2*i
        A[r,:3] = [u[i,0], u[i,1], 1]
        A[r,-3:] = [-u[i,0] * v[i,0], -u[i,1] * v[i,0], -v[i,0]]
        A[r+1,3:6] = [u[i,0], u[i,1], 1]
        A[r+1,-3:] = [-u[i,0] * v[i,1], -u[i,1] * v[i,1], -v[i,1]]

    _, _, vh = np.linalg.svd(A)
    h = vh.T[:,-1]
    H[0,:] = h[0:3]
    H[1,:] = h[3:6]
    H[2,:] = h[6:9]
    return H

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic
    img_corner = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography_matrix = solve_homography(img_corner, corners)

    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = int(new_pos[0, 0] / new_pos[2, 0]), int(new_pos[1, 0] / new_pos[2, 0])
            canvas[new_y, new_x] = img[y, x]
    return canvas

# corners are N-by-2 arrays, representing the N image corner (x, y) pairs
def backward_warping(img, output, corners,output_corner=None):
    h, w, _ = output.shape
    h_img, w_img,  _ = img.shape
    if output_corner is None:
        output_corner = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography_matrix = solve_homography(output_corner, corners)

    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = int(new_pos[0, 0] / new_pos[2, 0]), int(new_pos[1, 0] / new_pos[2, 0])
            if h_img > new_y >= 0 and w_img > new_x >= 0 :
                output[y, x] = img[new_y, new_x]
    return output

def main():
    # Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])

    # TODO: some magic
    canvas = transform(img1, canvas, canvas_corners1)
    canvas = transform(img2, canvas, canvas_corners2)
    canvas = transform(img3, canvas, canvas_corners3)
    canvas = transform(img4, canvas, canvas_corners4)
    canvas = transform(img5, canvas, canvas_corners5)
    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    
    # Part 2
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')
    # TODO: some magic
    output2 = np.zeros((150,150,3))
    corners = np.array([[1980,1236],[2040,1216],[2024,1392],[2084,1360]])
    output2 = backward_warping(img, output2, corners)
    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    output3 = np.zeros((400,500,3))
    h, w, ch = output3.shape
    output3_corner = np.array([[0, 0], [w-1, 0], [int(w/4), h-1], [int(3*w/4), h-1],[0,int(4*h/5)],[w-1,int(4*h/5)]])
    corners = np.array([[111,154],[607,154],[111,355],[607,355],[0,271],[724,271]])
    output3 = backward_warping(img_front, output3, corners, output3_corner)
    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    
    

if __name__ == '__main__':
    main()
