#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.cluster import MiniBatchKMeans
import argparse


def rgb2gray(image):
    return image.mean(axis=-1).astype("int64")


def patch_division(image, size=16):
    image = np.array(np.split(image, image.shape[1] // size, axis=1))
    image = np.array(np.split(image, image.shape[1] // size, axis=1))
    return image.reshape(-1, size, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU implementation for Local Binary Pattern"
    )
    parser.add_argument("image", metavar="I", type=str)
    args = parser.parse_args()

    image = plt.imread(args.image)
    image = rgb2gray(image)

    nrows, ncols = image.shape

    print(nrows, ncols)
    tile_size = 16
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128])

    for i in range(nrows):
        print("Gone through ", i * 100 // nrows, "%", end="\r")
        for j in range(ncols):
            top_line = padded_image[i, j : j + 3]
            bottom_line = padded_image[i + 2, j : j + 3]
            left = padded_image[i + 1, j]
            right = padded_image[i + 1, j + 2]
            test = np.array([*top_line, right, *bottom_line, left])
            test = np.where(test < image[i, j], 0, 1)
            image[i, j] = np.sum(test * weights)

    t_nrows = nrows // tile_size
    t_ncols = ncols // tile_size
    feature_vector = np.zeros(
        (t_nrows * t_ncols, tile_size * tile_size),
        dtype="float64",
    )
    for i in range(t_nrows):
        for j in range(t_ncols):
            hist = np.bincount(
                image[
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ].flatten(),
                minlength=256,
            ).astype("float64")
            hist /= np.linalg.norm(hist)
            feature_vector[i + j * t_nrows] = hist

    n_clusters = 16
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(feature_vector)
    LUT = np.random.rand(n_clusters, 3)
    clusters = kmeans.predict(feature_vector)
    colored_patches = LUT[clusters].reshape(t_nrows, t_ncols, 3)
    colored_patches = np.repeat(colored_patches, tile_size, axis=0)
    colored_patches = np.repeat(colored_patches, tile_size, axis=1)
    print(colored_patches.shape)

    plt.imsave("test.jpg", colored_patches)
