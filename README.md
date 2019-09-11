# satellite-clustering

`read_multispectral.py`

displays a multispectral image. Expects image to be in the following "open" format: ENVI "binary", data type: IEEE 32-bit floating point (ENVI type 4), byte-order: zero, interleave: band-sequential (BSQ)
* Designed for python 2.7, could be run with python 3. with some minor modification
* To run: `python read_multispectral.py /path/to/bin/file` where the path to the ".bin" file is the path to the multispectral image (in format above); note: a `.hdr` file must also be present.

`cluster.py`

Runs hierarchical agglomerative clustering on given satellite image.
* Runs agglomerative clustering method
* Compatible with python 2 and 3
* First run `read_multispectral.py` to inspect the satellite image
* To run: `python cluster.py /path/to/bin/file
