###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
import torch

if __name__ == "__main__":
    # TODO
