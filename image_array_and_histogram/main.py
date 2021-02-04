import numpy
import Image

def getImageArray(image_object):
    '''Returns an array of the image with elements being pixel values'''
    # get list of pixels
    pixel_values = list(image_object.getdata())
    # get image array
    image_arr = numpy.array(pixel_values)
    # reshape the array in the dimensions of image
    image_arr = image_arr.reshape(image_object.size[0], image_object.size[1])
    return image_arr

def getHist(image_arr):
    '''Returns histogram as an 1D array'''
    row = image_arr.shape[0]    # get row of the image row
    col = image_arr.shape[1]    # get column of the image row
    hist = [0 for i in range(256)]
    for r in range(row):
        for c in range(col):
            hist[image_arr[r,c]] += 1
    return hist

def getImageFromArray(arr, width, height):
    '''Returns image from array'''
    arr = numpy.array(arr)
    # reshaping array
    arr = arr.reshape(width, height)
    arr = Image.fromarray(numpy.uint8(arr))
    return arr
