import pandas as pd
import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans

data = pd.read_csv("out.csv")
x = []
for i in range(int(len(data.val) / 256)):
    x.append(data.val[256 * i: 256 * (i+1)])
x = np.array(x)

kmeans = MiniBatchKMeans(n_clusters=16)
kmeans.fit(x)

with open('cluster.csv', mode='w') as cluster_file:
    cluster_writer = csv.writer(cluster_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for cluster in kmeans.cluster_centers_ :
        cluster_writer.writerow(cluster)