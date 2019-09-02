import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio
import cv2
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import numpy as np
from osgeo import gdal, gdal_array

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


"""
Run hierarchical clustering
"""

def run_hierarchical_clustering(filepath, k):
    img = get_image_2d(filepath)
    img = img.astype(np.int32)
    img = img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]
    X = np.reshape(img, (-1, 1))

    # Define the structure A of the data. Pixels connected to their neighbors.
    connectivity = grid_to_graph(*img.shape)

    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = k
    ward = AgglomerativeClustering(n_clusters=n_clusters,
            linkage='ward', connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, img.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    ###############################################################################
    # Plot the results on an image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l, contours=1,
                    colors=[plt.cm.get_cmap("Spectral")(l / float(n_clusters)), ])
    plt.xticks(())
    plt.yticks(())
    plt.show()


""""
Reshapes provided image to 2D
"""


def get_image_2d(filepath):
    image = imageio.imread(filepath)
    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    return image_2d


if __name__ == "__main__":

    # parse command line arg
    try:
        image_jpg = sys.argv[1]
        image_bin = sys.argv[2]

    # catch no file given
    except IndexError:
        print("Must provide a jpg file and a bin file")
        sys.exit(0)


    # number of clusters
    K = 10

    # -- perform hierarchical clustering -- #
    run_hierarchical_clustering(image_jpg, K)

