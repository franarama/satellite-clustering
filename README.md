# satellite-clustering

## `read_multispectral.py`

displays a multispectral image. Expects image to be in the following "open" format: ENVI "binary", data type: IEEE 32-bit floating point (ENVI type 4), byte-order: zero, interleave: band-sequential (BSQ)
* Designed for python 2.7, could be run with python 3. with some minor modification
* To run: `python read_multispectral.py /path/to/bin/file` where the path to the ".bin" file is the path to the multispectral image (in format above); note: a `.hdr` file must also be present.

    e.g. to view Sentinel-2 image: `python read_multispectral.py mS2.bin`
    
    e.g. to view Landsat-8 image: `python read_multispectral.py mL8.bin`

## `cluster.py`

Runs hierarchical agglomerative clustering on given satellite image.
* Runs agglomerative clustering method
* Compatible with python 2 and 3
* First run `read_multispectral.py` to inspect the satellite image
* To run: `python cluster.py /path/to/bin/file


    e.g. to cluster Sentinel-2 image: `python cluster.py mS2.bin`
    
    e.g. to cluster Landsat-8 image: `python cluster.py mL8.bin`

    e.g. to cluster fused Sentinel-2 and Landsat-8 images: `python cluster.py mS2_L8.bin`


## small data test chips:

### Sentinel-2 scene (12 bands)

mS2.bin  	

mS2.hdr 	

### Landsat-8 scene (11 bands)

mL8.bin

mL8.hdr 	

### Fused Sentinel-2 and Landsat-8 scene (23 bands)

mS2_L8.bin

mS2_L8.hdr
