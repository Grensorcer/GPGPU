#!/usr/bin/env python

import argparse
import timeit
import matplotlib.pyplot as plt
import cv2
import cProfile
import pstats
import math
from sklearn.cluster import MiniBatchKMeans
from io import StringIO
from pathlib import Path
from lbp import lbp1, lbp2


def bench_image(image_path, save_path, nb_samples):
    save_path = save_path / (image_path.stem + "_bench")
    save_path.mkdir(exist_ok=True)

    image = plt.imread(image_path)
    h = image.shape[0]
    w = image.shape[1]
    bench_images = [
        cv2.resize(
            image, (math.ceil(h * s / nb_samples), math.ceil(w * s / nb_samples))
        )
        for s in range(1, nb_samples + 1)
    ]
    bench_times = []
    for s, bench_image in enumerate(bench_images):
        profiler = cProfile.Profile()
        filename = f"{math.ceil(h * (s + 1) / nb_samples)}x{math.ceil(w * (s + 1) / nb_samples)}.prof"
        with open(save_path / filename, "a") as bench_file:
            stream = StringIO()

            profiler.enable()
            feature_vector = lbp1(bench_image, 16, 0)
            profiler.disable()
            kmeans = MiniBatchKMeans(n_clusters=16)
            kmeans.fit(feature_vector)
            profiler.enable()
            colored_patches = lbp2(bench_image, kmeans, feature_vector, 16)
            profiler.disable()

            bench_prof_stats = pstats.Stats(profiler, stream=stream)
            bench_prof_stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
            bench_prof_stats.print_stats(20)

            bench_file.write(stream.getvalue())
            bench_time_1 = min(
                timeit.repeat(
                    lambda: lbp1(bench_image, 16, 0),
                    repeat=5,
                    number=1,
                )
            )
            bench_time_2 = min(
                timeit.repeat(
                    lambda: lbp2(bench_image, kmeans, feature_vector, 16),
                    repeat=5,
                    number=1,
                )
            )

            bench_times.append(bench_time_1 + bench_time_2)
            bench_file.write(
                f"Execution time in seconds for step 1: {bench_time_1:.4f}\n"
            )
            bench_file.write(
                f"Execution time in seconds for step 2: {bench_time_2:.4f}\n"
            )
            bench_file.write(
                f"Total execution time in seconds: {bench_time_1 + bench_time_2:.4f}\n"
            )

    plt.figure()
    plt.xlabel("Nb pixels")
    plt.ylabel("Computation time (s)")
    plt.title("LBP computation time")
    plt.plot(
        [
            math.ceil(h * s / nb_samples) * math.ceil(w * s / nb_samples)
            for s in range(1, nb_samples + 1)
        ],
        bench_times,
    )
    plt.savefig(save_path / "time_graph.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--name", type=str, default="bench_results")
    args = parser.parse_args()
    nb_samples = 10

    data_path = Path(args.path)
    if not data_path.is_dir():
        bench_image(data_path, Path("."), nb_samples)
    else:
        save_path = Path(args.name)
        save_path.mkdir(exist_ok=True)
        for img_file in data_path.iterdir():
            bench_image(img_file, save_path, nb_samples)
