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
Displays provided satellite image
"""


def show_image(filepath):
    image = imageio.imread(filepath)
    plt.imshow(image)
    plt.show()


"""
Runs the k-means algorithm on a provided image
"""


def run_kmeans(filepath, k):
    image_2d = get_image_2d(filepath)

    kmeans_cluster = KMeans(random_state=0, n_clusters=k)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z).astype(np.uint8))
    # plt.show()

    return cluster_labels, cluster_centers

"""
Run hierarchical clustering
"""

def run_hierarchical_clustering(image, k):
    # make the image similar to the Lena image from example
    gray_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    resize_img = cv2.resize(gray_img, (512, 512))
    img = resize_img.astype(np.int32)
    
    # print("resize img shape:", img.shape)
    # print("max:", img.max())
    # print("dtype:", img.dtype)

    # Downsample the image by a factor of 4
    image = img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]
    X = np.reshape(img, (-1, 1))
    print("resize img shape:", image.shape)

    # Define the structure A of the data. Pixels connected to their neighbors.
    # -- THIS IS WHERE THE ERROR HAPPENS -- #
    connectivity = grid_to_graph(*image.shape)

    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = k  # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters,
            linkage='ward', connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, image.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    ###############################################################################
    # Plot the results on an image
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l, contours=1,
                    colors=[plt.cm.get_cmap('Multispectral')(l / float(n_clusters)), ])
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


"""
Plots number of clusters vs. within cluster sum of squares
(which we aim to minimize)
"""


def elbow_method(image_2d, max_k):
    wcss = []
    for i in range(2, max_k):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        random_state=42)
        kmeans.fit(image_2d)
        wcss.append(kmeans.inertia_)

    x = [i for i in range(2, max_k)]
    plt.plot(x, wcss, '--bo')
    plt.xticks(x, x)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

"""
Uses kmeans to create a colormap of the image with colorbar
"""

def get_colormap(image_bin, k):
    # Read in raster image
    img_ds = gdal.Open(image_bin, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :13].reshape(new_shape)

    k_means = KMeans(n_clusters=k)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    plt.figure(figsize=(20, 20))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0),
            (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(
        "my map", colors, N=k)
    plt.imshow(X_cluster, cmap=cm)
    plt.colorbar(ticks=range(K))
    title = "Colormap with K=" + str(k)
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    # parse command line arg
    try:
        image_jpg = sys.argv[1]
        image_bin = sys.argv[2]

    # catch no file given
    except IndexError:
        print("Must provide a jpg file and a bin file")
        sys.exit(0)

    # show_image(image_jpg)

    # number of clusters
    K = 10

    # max number of clusters to plot with elbow method
    # MAX_K = 20

    # image_2d = get_image_2d(image_bin)
    # elbow_method(image_2d, (MAX_K + 1))

    # get_colormap(image_bin, K)

    # -- perform hierarchical clustering -- #
    run_hierarchical_clustering(image_jpg, K)

