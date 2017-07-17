import pandas
import numpy as np

pickup_locations = pandas.read_csv('./train.csv', dtype=np.float)

from sklearn.cluster import KMeans

clustering_algo = KMeans(n_clusters=8)

labels = clustering_algo.fit_predict(pickup_locations)

cluster_sizes = {i: len(np.where(clustering_algo.labels_ == i)[0]) for i in range(clustering_algo.n_clusters)}

cluster_centers = clustering_algo.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

test_location = [-6.2040803, 106.8217632]
predicted_label = clustering_algo.predict(np.array(test_location).reshape(1, -1))

predicted_cluster_center = cluster_centers[predicted_label][0]

pickup_locations['class'] = labels
colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'yellow', 4: 'm', 5: 'orange', 6: 'black', 7: 'cyan'}

predicted_cluster_points = pickup_locations[pickup_locations['class'] == predicted_label[0]]
from haversine import haversine

center = predicted_cluster_points.iloc[0]
min_dist = haversine((test_location[0], test_location[1]), (center.lat, center.lng))
for _, point in predicted_cluster_points.iterrows():
    dist = haversine((test_location[0], test_location[1]), (point.lat, point.lng))
    if dist < min_dist:
        min_dist = dist
        center = point

optimal_pickup_location = center
print("optimal_pickup_location")
print(optimal_pickup_location)
print("minimum distance", min_dist)
