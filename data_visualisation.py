import pandas
import numpy as np
from sklearn.cluster import KMeans, estimate_bandwidth

negatives = pandas.read_csv('./n.csv', dtype=np.float)
negatives_location = negatives.iloc[:, :2]
print(negatives_location.head(10))

km = KMeans(n_clusters=16)
print("******train.csv")
km.fit(negatives_location)

print("****** fit")
labels = km.labels_
print("******** labels")
print(labels)

cluster_sizes = {i: len(np.where(km.labels_ == i)[0]) for i in range(km.n_clusters)}
print("******** cluster sizes")
print(sorted(cluster_sizes.items()))

cluster_centers = km.cluster_centers_
print("******** cluster centers")
print(cluster_centers)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters ", n_clusters_, km.n_clusters)
