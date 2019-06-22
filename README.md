# Image array and histogram

This python package helps you to get an array and histogram of a greyscale image. You can also convert an array to greyscale image.

## Installation
```sh
$ pip install image-array-and-histogram-rrsc
```

## Functions
* `getImageArray(image_object)`: Takes an image object and returns image array. The dimensions of the array is the size of the image.
* `getHist(image_array)`: Takes an image array (of pixels) and returns an array of size 256 as the histogram of the image.
* `getImageFromArray(image_array, width, height)`: Takes an array, width and height and returns an image formed from the passed array. The dimensions of the image are `width*height`