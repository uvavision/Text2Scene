#!/usr/bin/env python

import os, sys
import cv2, json
import math
from time import time
import numpy as np
import os.path as osp
from glob import glob
from sklearn import (manifold, datasets)
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean


def tsne_embedding(X, metric='euclidean'):
    print("Start t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, metric=metric)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    print("t-SNE embedding completes (time %.2fs)" % (time() - t0))
    return X_tsne


def tsne_images(images, X_tsne, output_size, patch_size):
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    coords = (X_tsne - x_min) / (x_max - x_min)

    output_size = np.array(output_size)
    patch_size  = np.array(patch_size)

    coords = np.multiply(coords, (output_size - patch_size - 1).reshape((1,2))).astype(np.int)

    output_img = np.ones((output_size[0], output_size[1], 3)) * 255

    for i in range(coords.shape[0]):
        p = coords[i]
        output_img[p[0]:(p[0]+patch_size[0]), p[1]:(p[1]+patch_size[1]), :] = \
            cv2.resize(images[i], (patch_size[1], patch_size[0]))

    return output_img.astype(np.uint8)


def tsne_colors(colors, X_tsne, output_size):
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    coords = (X_tsne - x_min) / (x_max - x_min)

    output_size = np.array(output_size)
    output_img = np.ones((output_size[0], output_size[1], 3)) * 255

    coords = np.multiply(coords, output_size.reshape((1,2))).astype(np.int)
    for i in range(coords.shape[0]):
        p = coords[i]
        cv2.circle(output_img, (p[0], p[1]), 10, (colors[i][0], colors[i][1], colors[i][2]), -1)
    
    return output_img