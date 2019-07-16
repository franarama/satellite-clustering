# satellite-clustering

`read_multispectral.py`

Makes a satellite image into a jpg.
* Must be run using Python 2. 
* To run: `python read_multispectral.py /path/to/bin/file` where the path to the bin file is the path to the multispectral image (BSQ format), that also has a `.hdr` file.

`cluster.py`

Runs k-means and elbow method on given satellite image.
* Provides functions to run k-means clustering and plot elbow method. 
* Created for Python 3.
* First run `read_multispectral.py` to get the satellite image as a jpg
* To run: `python3 cluster.py /path/to/jpg/satellite/image`

