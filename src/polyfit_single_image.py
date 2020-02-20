import cv2
import os
import runlength
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math
import zlib


# add test image path
path = os.path.abspath("../data/top100/test_tiff87.tiff")
z_lib = 0
improved_zlib = 0

z_lib_win_count = 0
improved_zlib_win_count = 0

## Read image.
img_path = path
print img_path
img = cv2.imread(img_path,0)
#img = cv2.resize(img, (16,25))
#img = np.zeros((16,25))
img_shape = img.shape


## Compute Zlib compression ratio

## Compress image.
x = np.asarray(img.ravel()).astype(np.uint8)

original_image = np.getbuffer(x)
y = zlib.compress(original_image)
#print "Compressed img size:"+str(len(y))
compress_ratio = (float(len(original_image)) - float(len(y))) / float(len(original_image))
compress_ratio_percent_zlib = 100.0 * compress_ratio

compress_ratio2 = float(len(original_image))/ (float(len(y)))
print 'Compressed zlib: %f%%' % (100.0 * compress_ratio)
#print 'Compressed zlib2: %f' % (compress_ratio2)

values = img.ravel().astype(np.uint8)

# ## Image to runlength.
# starts, lengths, values = runlength.rlencode(img.ravel())
# rle_decoded_arr = runlength.rldecode(starts, lengths, values)
# rle_decoded_img = np.asarray(rle_decoded_arr).astype(np.uint8).reshape(img_shape)
# #print starts, lengths, values
# lengths = lengths.astype(np.uint8)
# values = values.astype(np.uint8)



# ## Compress RLE.
# x = np.asarray(img.ravel()).astype(np.uint8)
# #print len(x)
# original_image = np.getbuffer(x)

# x = list(lengths)
# #print len(x)
# x = np.asarray(x)
# buffer1 = np.getbuffer(x)
# y = zlib.compress(buffer1)

# x = list(values)
# #print len(x)
# x = np.asarray(values)
# buffer2 = np.getbuffer(x)
# y2 = zlib.compress(buffer2)

# x = list(lengths)+list(values)
# buffer3 = np.getbuffer(np.asarray(x).astype(np.int8))

# compress_ratio = (float(len(original_image)) - float(len(y)+len(y2))) / float(len(original_image))
# compress_ratio_rle = (float(len(original_image)) - float((len(buffer3)))) / float(len(original_image))
# print 'Compressed zlib_rle: %f%%' % (100.0 * compress_ratio)
# print 'Compressed rle: %f%%' % (100.0 * compress_ratio_rle)


# ## Exclude rle.
# values = img.ravel()


def poly_compress(arr, k, deg):
    differences = []
    polynomial_coefficients = []
    n_poly = len(arr)/k
    for i in range(0, len(arr), n_poly):
        inc = n_poly
        if i+n_poly > len(arr):
            inc = len(arr)-i
        x = np.arange(inc)
        y = arr[i:i+inc]
        if(deg > inc):
            deg = 0
        z = np.polyfit(x, y, deg=deg)
        p = np.poly1d(z)

        polynomial_coefficients.append(z)
        diff = (p(x).astype(np.int8)-y)
        differences += list(diff)
        # xp = np.linspace(0, len(x), len(x)*100)
        # plt.figure(figsize=(6.5,4))
        # plt.plot(x,y,'o',label='data')
        # plt.plot(xp, p(xp),label='polyfit')
        # plt.show()
    return (np.asarray(differences), polynomial_coefficients)


def poly_decompress(differences, coefficients, k):
    # Revert differences.
    n_poly = len(differences)/k
    decoded_values = []
    for i in range(0, len(differences), n_poly):
        inc = n_poly
        if i+n_poly > len(differences):
            inc = len(differences)-i
        z = coefficients[i/n_poly]
        p = np.poly1d(z)
        diff = differences[i:i+inc]
        x = np.arange(inc)
        decoded_values += list(p(x).astype(np.int8)-diff)
    return np.asarray(decoded_values)

# def split(array, nrows, ncols):
#     """Split a matrix into sub-matrices."""

#     r, h = array.shape
#     return (array.reshape(h//nrows, nrows, -1, ncols)
#                  .swapaxes(1, 2)
#                  .reshape(-1, nrows, ncols))

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    #win_height = img.shape[0]/nrows
    #win_width = img.shape[1]/ncols
    win_height = nrows
    win_width = ncols
    vecs = []
    #--
    #print "splitting.."
    #--
    for r in range(0,img.shape[0], win_height):
        for c in range(0,img.shape[1], win_width):
            window = img[r:r+win_height,c:c+win_width]
            #print window.shape
            vecs.append(window)
    #print vecs

    #--
    #print "done."
    #--
    return np.asarray(vecs)


def merge(array, nrows, ncols):
    #print nrows,ncols
    result = []

    # #print array.shape
    # print len(array)
    # r, c = array.shape
    # result = np.zeros(array.shape)
    # array = array.ravel()
    # i_counter = 0
    # for i in range(0,r,nrows):
    #     counter = nrows*ncols
    #     for j in range(0, c, ncols):
    #         result[i:i+nrows, j:j+ncols] = array[i_counter:i_counter+nrows*ncols].reshape((nrows, ncols))
    #         counter += nrows*ncols
    #         i_counter += nrows*ncols
    height, width = img_shape
    #print array
    v = []
    #print v.shape
    for i in range(0, height/(nrows)):
        h = np.empty((nrows,ncols),dtype=np.int8)
        h2 = []
        for j in range(0, width/(ncols)):
            window = array[i*(width/ncols)+j].reshape(nrows,ncols)
            #print window.shape
            #print h.shape
            #print array[j].shape
            
            h = np.hstack((h, window))
            h2.append(window)
        h = np.hstack(np.asarray(h2))
        v.append(h)
        #v = np.vstack((v,h))
    #print v
    v = np.vstack(np.asarray(v))
    return v
# def merge(array, nrows, ncols):
#     result = split(array.reshape(img_shape),nrows,ncols).reshape(img_shape)
#     return result

def poly_compress_grid(arr, n, m, deg):
    #print arr
    splitted = split(arr, n, m)
    differences = []
    coeffs = []
    #print splitted[0]
    counter = 0
    for i in range(len(splitted)):
        grid_arr = splitted[i]
        #print grid_arr
        #print grid_arr
        diff, coeff = poly_compress(grid_arr.ravel(), 1, deg)
        differences.append(list(diff))
        #coeffs.append(list(np.asarray(coeff).ravel()))
        coeffs.append(np.asarray(coeff).ravel())
        counter += 1
        #print counter/float(len(splitted))
    return (np.asarray(differences), np.asarray(coeffs))

def poly_decompress_grid(differences, coefficients, n, m):
    #differences = np.asarray(differences).reshape(img_shape)
    #differences = split(differences, n, m)
    # coefficients = coefficients.ravel()
    # decoded_values = poly_decompress(differences, coefficients,1)
    #print coefficients.shape
    #differences = split(merge(differences, n, m), n, m)
    #coefficients = split(merge(coefficients, n, m), n,m)
    #differences = split(merge(differences, n, m), n, m)
    decoded_values = []
    # #coefficients = coefficients.reshape(coefficients.shape[0],coefficients.shape[2])
    # print coefficients.shape
    for i in range(len(differences)):
        diff = np.asarray(differences[i]).ravel()
        coeff = coefficients[i]
        z = coeff
        p = np.poly1d(z)
        x = np.arange(len(diff))
        decoded_values += list(p(x).astype(np.int8)-diff)

    # decoded_values = np.reshape(decoded_values, img_shape)
    # decoded_values = split(decoded_values, n, m)
    # decoded_values = decoded_values.ravel().reshape(img_shape)
    # decoded_values = merge(decoded_values, n, m)

    decoded_values = np.reshape(decoded_values, img_shape)
    decoded_values = split(decoded_values, n, m)
    decoded_values = merge(decoded_values,n ,m)

    #decoded_values = merge(np.asarray(decoded_values), n, m)
    #decoded_values = merge(np.asarray(decoded_values), n, m)
    #decoded_values = merge(split(decoded_values,n,m), n, m)
    #decoded_values = split(decoded_values, n,m)
    return np.asarray(decoded_values)


#k = int(img_shape[1]*3*float(img_shape[1])/img_shape[0])
k = 10000
deg = 4
# n1 = 5
# n2 = 4
# n = img.shape[0]/n1
# m = img.shape[1]/n2

n1 = 5
n2 = 5

r1 = img_shape[0] % n1
c1 = img_shape[1] % n2

# if r1 == 0:
#     r1 = img_shape[0]
# if c1 == 0:
#     c1 = img_shape[1]

#--
#print r1,c1
#print img_shape[1]
#--
if r1 != 0:
    zeros = np.zeros((n1-r1,img_shape[1]))
    img = np.vstack((img, zeros))
    if c1 != 0:
        zeros = np.zeros((img_shape[0]+n1-r1,n2-c1))
        img = np.hstack((img, zeros))
if r1 == 0:
    if c1 != 0:
        zeros = np.zeros((img_shape[0],n2-c1))
        #--
        #print img.shape[1]
        #print zeros.shape
        #--
        img = np.hstack((img, zeros))
#--
#print img.shape[1]
#--
img_shape = img.shape

n = img.shape[0]/n1
m = img.shape[1]/n2

#--
#print img_shape, n, m
#--

# remaining_cols = img[:,-c1:]
# left_part = img[:,:-c1]
# remaining_rows = left_part[-r1:,:]
# remaining_img = left_part[:-r1,:]

# print "hello"
# print len(remaining_rows)
# if (remaining_rows.size == 0):
#     remaining_rows = np.asarray([1,2,3])
# if (remaining_cols.size == 0):
#     remaining_cols = np.asarray([1,2,3])
# print "rem rows"+str(remaining_rows)

# img = remaining_img
# img_shape = remaining_img.shape

#--
#print img
#--

#n, m = 2,2
#--
#print "compressing.."
#--

#differences, polynomial_coefficients = poly_compress(values, k, deg)
differences, polynomial_coefficients = poly_compress_grid(img, n, m, deg)
# d_remaining1, d_coeff1 = poly_compress(remaining_rows.ravel(), 1, deg)
# d_remaining2, d_coeff2 = poly_compress(remaining_cols.ravel(), 1, deg)
#print differences
differences = differences.astype(np.int8)
# d_remaining1 = d_remaining1.astype(np.int8)
# d_remaining2 = d_remaining2.astype(np.int8)
#length_differences, length_polynomial_coefficients = poly_compress(lengths,k,deg)

#--
#print "done."
#--

## Compress.
x = np.asarray(img.ravel()).astype(np.uint8)
#--
#print "Image size:"+str(len(np.getbuffer(x)))
#--
original_image = np.getbuffer(x)


# compress differences
x = np.asarray(differences).astype(np.int8)
buffer = np.getbuffer(x)
y = zlib.compress(buffer)

#compress polynomial coefficients
x = np.asarray(polynomial_coefficients)
buffer = np.getbuffer(x)
y2 = zlib.compress(buffer)

# x = np.asarray(d_remaining1)
# buffer = np.getbuffer(x)
# y3 = zlib.compress(buffer)

# x = np.asarray(d_remaining2)
# buffer = np.getbuffer(x)
# y4 = zlib.compress(buffer)

# x = np.asarray(d_coeff1)
# buffer = np.getbuffer(x)
# y5 = zlib.compress(buffer)

# x = np.asarray(d_coeff2)
# buffer = np.getbuffer(x)
# y6 = zlib.compress(buffer)

# x = np.asarray(length_differences).astype(np.int8)
# buffer = np.getbuffer(x)
# y3 = zlib.compress(buffer)

# x = np.asarray(length_polynomial_coefficients)
# buffer = np.getbuffer(x)
# y4 = zlib.compress(buffer)

# x = np.asarray(lengths)
# buffer = np.getbuffer(x)
# y5= zlib.compress(buffer)

#print "Compressed:"+str(len(y)+len(y2))

#--
#print "Encoded image size: "+str(float(len(y)+len(y2)))
#--

## get zfit compression ratio
compress_ratio = (float(len(original_image)) - float(len(y)+len(y2))) / float(len(original_image))
compress_ratio_imp_zlib = 100.0 * compress_ratio





print 'Compressed zlib_polyfit: %f%%' % (100.0 * compress_ratio)


compress_ratio3 = float(len(original_image))/ (float(len(y)))
#print 'Compressed zlib_polyfit3: %f' % (compress_ratio3)

#--
#print np.histogram(differences, bins=10)
#

## RLE on differences.
# starts, lengths, values = runlength.rlencode(differences)

# x = np.asarray(lengths).astype(np.int8)
# buffer1 = np.getbuffer(x)

# x = np.asarray(values).astype(np.int8)
# buffer2 = np.getbuffer(x)

# compress_ratio = (float(len(original_image)) - float(len(buffer1)+len(buffer2))) / float(len(original_image))
# print 'Compressed rle_differences: %f%%' % (100.0 * compress_ratio)


## Decompress
#decoded_values = poly_decompress(differences.astype(np.int8), polynomial_coefficients, k)
decoded_values = poly_decompress_grid(differences, polynomial_coefficients, n1, n2)
decoded_values = np.asarray(decoded_values).astype(np.uint8)
#print img.shape, decoded_values.shape
#print decoded_values.shape

# print np.asarray(decoded_values).astype(np.uint8).ravel()[:1000]
# print img.ravel()[:1000]

#print decoded_values[:100]
#print img[:100]
# print np.asarray(decoded_values).astype(np.uint8).ravel()[-1000:]
# print img.ravel()[-1000:]

#--
#print "Decode error: "+str(np.array_equal(decoded_values.ravel(), img.ravel()))
#--

#decoded_img = np.asarray(decoded_values).astype(np.uint8).reshape(img_shape)
#print "Decoded image size: "+str(len(np.getbuffer(decoded_values)))
# cv2.imshow("decoded_img",decoded_img)
# cv2.waitKey(0)
#decoded_lengths = poly_decompress(length_differences, length_polynomial_coefficients,k)
#decoded_starts = runlength.lengths_to_starts(decoded_lengths)
#decoded_rle = runlength.rldecode(decoded_starts,decoded_lengths,decoded_values)
#decoded_img = decoded_rle.astype(np.uint8).reshape(img_shape)
