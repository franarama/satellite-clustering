import matplotlib.pyplot as plt
import imageio
from sklearn.cluster import KMeans
import numpy as np
import sys
import math

"""
Displays provided satellite image
"""
def show_image(filepath):
	image = imageio.imread(filepath)
	plt.imshow(image)
	plt.show()

"""
Runs the k-means algorithm on a provided image
"""
def run_kmeans(filepath, k=2):
	image = imageio.imread(filepath)
	x, y, z = image.shape
	image_2d = get_image_2d(filepath)

	kmeans_cluster = KMeans(random_state=0, n_clusters=k)
	kmeans_cluster.fit(image_2d)
	cluster_centers = kmeans_cluster.cluster_centers_
	cluster_labels = kmeans_cluster.labels_

	plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z).astype(np.uint8))
	plt.show()

"""
Reshapes provided image to 2D
"""
def get_image_2d(filepath):
	image = imageio.imread(filepath)
	x, y, z = image.shape
	image_2d = image.reshape(x*y, z)
	return image_2d

"""
Plots number of clusters vs. within cluster sum of squares
(which we aim to minimize)
"""
def elbow_method(image_2d, max_k):
	wcss = []
	for i in range(2, max_k):
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
	    	random_state = 42)
	    kmeans.fit(image_2d)
	    wcss.append(math.log(kmeans.inertia_))

	plt.plot(range(2, max_k), wcss)
	plt.title('The Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('log(WCSS)')
	plt.show()

if __name__ == "__main__":

    # parse command line arg
    try:
    	image = sys.argv[1]

    # catch no file given 
    except IndexError:
    	print("Must provide a filename")
    	sys.exit(0)

    # show_image(image)
    run_kmeans(image)

    # max number of clusters to plot with elbow method
    MAX_K = 20

    image_2d = get_image_2d(image)
    elbow_method(image_2d, (MAX_K + 1))



