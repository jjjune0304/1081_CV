import numpy as np
import cv2
import sys
import time


def main(ref_image,template,video):
    
    ts = time.time()
    ref_image = cv2.imread(ref_image)  ## 3100*3100
    template = cv2.imread(template)  ## 410*410
    ref_image = cv2.resize(ref_image, (1000 , 1000 ), interpolation=cv2.INTER_CUBIC)
    factor = ref_image.shape[0] / template.shape[0] # size fraction between reference image and template
    
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
    i = 0

    # feature detector
    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    #matcher = cv2.BFMatcher()

    # reference image  features
    template_kps, template_des = sift.detectAndCompute(template, None)

    MIN_MATCH_COUNT = 10

    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            ## TODO: homography transform, feature detection, ransanc, etc.
            
            # frame features
            frame_kps, frame_des = sift.detectAndCompute(frame, None)
            # find matching descriptor
            matches = matcher.knnMatch(template_des, frame_des, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([ template_kps[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ frame_kps[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                # multiply the size factor to match the coordinate
                homography_matrix, mask = cv2.findHomography(src_pts * factor, dst_pts, cv2.RANSAC,5.0)
                h, w, _ = ref_image.shape
                # project the template to the video frame
                for y in range(h):
                    for x in range(w):
                        new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
                        new_x, new_y = int(new_pos[0, 0] / new_pos[2, 0]), int(new_pos[1, 0] / new_pos[2, 0])

                        # prevent from negative index or out of boundary
                        if (film_w > new_x >= 0) and (film_h > new_y >= 0):
                            frame[new_y, new_x] = ref_image[y, x]

            else:
                print("Not enough matches are found - {}/{}".format (len(good),MIN_MATCH_COUNT))

            videowriter.write(frame)
            i += 1

        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)
