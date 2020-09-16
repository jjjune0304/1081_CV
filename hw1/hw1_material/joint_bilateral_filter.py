import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def joint_bilateral_filter(self, input, guidance):
        ## TODO
        return output


