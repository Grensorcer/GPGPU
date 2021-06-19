#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.cluster import MiniBatchKMeans
import argparse


def rgb2gray(image):
    return image.mean(axis=-1).astype("int64")


def compute_neighbors(image, tol=1):
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    val = np.where(image - tol >= 0, image - tol, 0)

    top_left = np.where(padded_image[:-2, :-2] < val, 0, 1)
    top = np.where(padded_image[:-2, 1:-1] < val, 0, 1) * 2
    top_right = np.where(padded_image[:-2, 2:] < val, 0, 1) * 4
    right = np.where(padded_image[1:-1, 2:] < val, 0, 1) * 8
    bottom_right = np.where(padded_image[2:, 2:] < val, 0, 1) * 16
    bottom = np.where(padded_image[2:, 1:-1] < val, 0, 1) * 32
    bottom_left = np.where(padded_image[2:, :-2] < val, 0, 1) * 64
    left = np.where(padded_image[1:-1, :-2] < val, 0, 1) * 128

    image = (
        top_left + top + top_right + left + right + bottom_left + bottom + bottom_right
    )
    return image


def compute_neighbors_per_tile(image, tile_size=16):
    t_nrows = image.shape[0] // tile_size
    t_ncols = image.shape[1] // tile_size

    for k in range(t_nrows):
        for l in range(t_ncols):
            print(
                "Gone through ",
                (l + k * t_ncols) * 100 // (t_nrows * t_ncols),
                "%",
                end="\r",
            )
            patch = image[
                k * tile_size : (k + 1) * tile_size,
                l * tile_size : (l + 1) * tile_size,
            ]
            padded_patch = np.pad(
                patch, ((1, 1), (1, 1)), mode="constant", constant_values=0
            )

            top_left = np.where(padded_patch[:-2, :-2] < patch, 0, 1)
            top = np.where(padded_patch[:-2, 1:-1] < patch, 0, 1) * 2
            top_right = np.where(padded_patch[:-2, 2:] < patch, 0, 1) * 4
            right = np.where(padded_patch[1:-1, 2:] < patch, 0, 1) * 8
            bottom_right = np.where(padded_patch[2:, 2:] < patch, 0, 1) * 16
            bottom = np.where(padded_patch[2:, 1:-1] < patch, 0, 1) * 32
            bottom_left = np.where(padded_patch[2:, :-2] < patch, 0, 1) * 64
            left = np.where(padded_patch[1:-1, :-2] < patch, 0, 1) * 128

            patch = (
                top_left
                + top
                + top_right
                + left
                + right
                + bottom_left
                + bottom
                + bottom_right
            )

    return image


def compute_feature_vectors(image, tile_size=16):
    t_nrows = image.shape[0] // tile_size
    t_ncols = image.shape[1] // tile_size
    feature_vector = np.zeros(
        (t_nrows * t_ncols, 256),
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
            feature_vector[j + i * t_ncols] = hist

    return feature_vector


def patch_cluster_classify(feature_vector, nrows, ncols, tile_size, n_clusters=16):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(feature_vector)
    clusters = kmeans.predict(feature_vector)
    colored_patches = clusters.reshape(nrows // tile_size, ncols // tile_size)
    # colored_patches = clusters.reshape(nrows // tile_size, ncols // tile_size)
    colored_patches = np.repeat(colored_patches, tile_size, axis=0)
    colored_patches = np.repeat(colored_patches, tile_size, axis=1)

    return colored_patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU implementation for Local Binary Pattern"
    )
    parser.add_argument("image", metavar="I", type=str)
    args = parser.parse_args()

    image = plt.imread(args.image)
    image = rgb2gray(image)
    plt.imsave("gray.jpg", image, cmap="gray")

    nrows, ncols = image.shape
    tile_size = 16
    n_clusters = 16

    image0 = compute_neighbors(image, tol=0)
    feature_vector = compute_feature_vectors(image0, tile_size=tile_size)
    colored_patches = patch_cluster_classify(
        feature_vector, nrows, ncols, tile_size, n_clusters=n_clusters
    )

    plt.imsave("test_notol.jpg", colored_patches, cmap="tab20")

    image1 = compute_neighbors(image, tol=1)
    feature_vector = compute_feature_vectors(image1, tile_size=tile_size)
    colored_patches = patch_cluster_classify(
        feature_vector, nrows, ncols, tile_size, n_clusters=n_clusters
    )

    plt.imsave("test_1.jpg", colored_patches, cmap="tab20")

    image2 = compute_neighbors(image, tol=2)
    feature_vector = compute_feature_vectors(image2, tile_size=tile_size)
    colored_patches = patch_cluster_classify(
        feature_vector, nrows, ncols, tile_size, n_clusters=n_clusters
    )

    plt.imsave("test_2.jpg", colored_patches, cmap="tab20")

    image3 = compute_neighbors(image, tol=3)
    feature_vector = compute_feature_vectors(image3, tile_size=tile_size)
    colored_patches = patch_cluster_classify(
        feature_vector, nrows, ncols, tile_size, n_clusters=n_clusters
    )

    plt.imsave("test_3.jpg", colored_patches, cmap="tab20")

    image4 = compute_neighbors(image, tol=4)
    feature_vector = compute_feature_vectors(image4, tile_size=tile_size)
    colored_patches = patch_cluster_classify(
        feature_vector, nrows, ncols, tile_size, n_clusters=n_clusters
    )

    plt.imsave("test_4.jpg", colored_patches, cmap="tab20")
