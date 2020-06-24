import cv2
import os
import runlength
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math
import zlib
import glob
import pickle, gzip
import sys


def poly_compress(arr, k, deg):    
    differences = []
    polynomial_coefficients = []
    n_poly = len(arr)/k
    n_poly = int(n_poly)

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
        if (str(img_dtype) == "uint8"):
            diff = (p(x).astype(np.uint8)-y)
        else:
            diff = (p(x).astype(np.uint16)-y)
        #print(diff)   
        
        differences += list(diff)
        xp = np.linspace(0, len(x), len(x)*100)
        #plt.figure(figsize=(6.5,4))
        #plt.hist(x=differences, bins='auto')
        #plt.plot(x,y,'o',label='data')
        #plt.plot(xp, p(xp),label='polyfit')
        #plt.xlabel("Difference")
        #plt.ylabel("Count")
        
        #plt.show()
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
        if (str(img_dtype) == "uint8"):
            decoded_values += list(p(x).astype(np.uint8)-diff)
        else:
            decoded_values += list(p(x).astype(np.uint16)-diff)
    return np.asarray(decoded_values)

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    #win_height = img.shape[0]/nrows
    #win_width = img.shape[1]/ncols
    win_height = nrows
    win_height = int(win_height)
    win_width = ncols
    win_width = int(win_width)
    vecs = []
    #--
    #print "splitting.."
    #--
    print(img.shape[0],win_height)
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

    height, width = img_shape
    #print array
    v = []
    #print v.shape
    for i in range(0, int(height/(nrows))):
        #bu int32 için if else gelecek
        
        if (str(img_dtype) == "uint8"):
            h = np.empty((nrows,ncols),dtype=np.int8)
        else:
            h = np.empty((nrows,ncols),dtype=np.int16)
        h2 = []
        for j in range(0, int(width/(ncols))):
            window = array[i*(int(width/(ncols)))+j].reshape(nrows,ncols)
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
        if (str(img_dtype) == "uint8"):
            decoded_values += list(p(x).astype(np.uint8)-diff)
        else:
            decoded_values += list(p(x).astype(np.uint16)-diff)

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


## Choose grid length
n1 = 30
n2 = 30

file_paths = glob.glob("../data/*.tiff")

z_lib = 0
improved_zlib = 0

z_lib_win_count = 0
improved_zlib_win_count = 0
total_count = 0
two_channel_images_count = 0
for path in file_paths:
#for i in range(1, 31):
    ## Read image.
    img_path = path
    print(img_path)
    img = cv2.imread(img_path,-1)
    print(img.dtype)
    print(len(img.shape))
    print(img.shape)

    if (len(img.shape) == 3):
        #b = img[:,:,0]
        #g = img[:,:,1]
        #r = img[:,:,2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.dtype)
    else:
        print("WARNING:this image has " + str(len(img.shape)) + " channels")
        two_channel_images_count += 1
    img_shape = img.shape
    img_dtype = img.dtype
    
    #img = cv2.resize(img, (16,25))
    #img = np.zeros((16,25))
    
    print(img_shape)
    ## Compress image.
    #print("ravel" + str(img.ravel().dtype))
    print(len(img.ravel()))
    if (str(img_dtype) == "uint8"):
        x = np.asarray(img.ravel()).astype(np.uint8)
    else:
        x = np.asarray(img.ravel()).astype(np.uint16)
    #print("ravel" + str(img.ravel().dtype))
    #print robust.mad(x)

    original_image = pickle.dumps(x)
    #original_image = np.getbuffer(x)
    y = zlib.compress(original_image)
    #print "Compressed img size:"+str(len(y))
    compress_ratio = (float(len(original_image)) - float(len(y))) / float(len(original_image))
    compress_ratio_percent_zlib = 100.0 * compress_ratio
    z_lib += compress_ratio_percent_zlib
    compress_ratio2 = float(len(original_image))/ (float(len(y)))
    print ('Compressed zlib: %f%%' % (100.0 * compress_ratio))
    #print 'Compressed zlib2: %f' % (compress_ratio2)

    if (str(img_dtype) == "uint8"):
        values = img.ravel().astype(np.uint8)
    else:
        values = img.ravel().astype(np.uint16)

    #k = int(img_shape[1]*3*float(img_shape[1])/img_shape[0])
    k = 10000
    deg = 4
   

    

    r1 = img_shape[0] % n1
    c1 = img_shape[1] % n2

    
    if r1 != 0:
        zeros = np.zeros((n1-r1,img_shape[1]))
        img = np.vstack((img, zeros))
        if c1 != 0:
            zeros = np.zeros((img_shape[0]+n1-r1,n2-c1))
            img = np.hstack((img, zeros))
    if r1 == 0:
        if c1 != 0:
            zeros = np.zeros((img_shape[0],n2-c1))
           
            img = np.hstack((img, zeros))
   
    img_shape = img.shape

    n = img.shape[0]/n1
    m = img.shape[1]/n2

   
    differences, polynomial_coefficients = poly_compress_grid(img, n, m, deg)
   
    if (str(img_dtype) == "uint8"):   
        differences = differences.astype(np.uint8)
    else:
        differences = differences.astype(np.uint16)
    
    ## Compress.
    if (str(img_dtype) == "uint8"):   
        x = np.asarray(img.ravel()).astype(np.uint8)
    else:
        x = np.asarray(img.ravel()).astype(np.uint16)
   
    original_image = pickle.dumps(x)

    if (str(img_dtype) == "uint8"):
        x = np.asarray(differences).astype(np.uint8)
    else:
        x = np.asarray(differences).astype(np.uint16)
    #buffer = np.getbuffer(x)
    buffer = pickle.dumps(x)
    y = zlib.compress(buffer)

    x = np.asarray(polynomial_coefficients)
    #buffer = np.getbuffer(x)
    buffer = pickle.dumps(x)
    y2 = zlib.compress(buffer)


    #--
    #print "Encoded image size: "+str(float(len(y)+len(y2)))
    #--

    compress_ratio = (float(len(original_image)) - float(len(y)+len(y2))) / float(len(original_image))
    compress_ratio_imp_zlib = 100.0 * compress_ratio
    print('Compressed imp_zlib: %f%%' % (compress_ratio_imp_zlib))

    improved_zlib += compress_ratio_imp_zlib
    
    compress_ratio3 = float(len(original_image))/ (float(len(y)))

    if compress_ratio_percent_zlib > compress_ratio_imp_zlib:
        z_lib_win_count += 1
        print ('WARNING: zlib won!' + 'this is the image: ' + img_path)
    elif compress_ratio_percent_zlib < compress_ratio_imp_zlib:
        improved_zlib_win_count += 1

    total_count += 1


    
    
    #--
    #print (np.histogram(differences, bins=10))
    #



    print(len(differences))
    ## Decompress
    #ADDED NEW
    
    differences = pickle.loads(zlib.decompress(y))
    print(len(differences))
    print(len(polynomial_coefficients))
    polynomial_coefficients = pickle.loads(zlib.decompress(y2))
    print(len(polynomial_coefficients))
    #ADDED NEW
   
    decoded_values = poly_decompress_grid(differences, polynomial_coefficients, n1, n2)
    if (str(img_dtype) == "uint8"):
        decoded_values = np.asarray(decoded_values).astype(np.uint8)
    else:
        decoded_values = np.asarray(decoded_values).astype(np.uint16)
    #print img.shape, decoded_values.shape
    #print decoded_values.shape

    # print np.asarray(decoded_values).astype(np.uint32).ravel()[:1000]
    # print img.ravel()[:1000]

    #print (decoded_values[:100])
    #print (img[:100])
    # print np.asarray(decoded_values).astype(np.uint32).ravel()[-1000:]
    # print img.ravel()[-1000:]
    
    
    print ("Decode error: "+str(np.array_equal(decoded_values.ravel(), img.ravel().astype(np.uint8))))
    print (decoded_values.shape)
    cv2.imwrite("../data/out.jpg", decoded_values)

    #decoded_img = np.asarray(decoded_values).astype(np.uint32).reshape(img_shape)
    #print "Decoded image size: "+str(len(np.getbuffer(decoded_values)))
    # cv2.imshow("decoded_img",decoded_img)
    # cv2.waitKey(0)
    #decoded_lengths = poly_decompress(length_differences, length_polynomial_coefficients,k)
    #decoded_starts = runlength.lengths_to_starts(decoded_lengths)
    #decoded_rle = runlength.rldecode(decoded_starts,decoded_lengths,decoded_values)
    #decoded_img = decoded_rle.astype(np.uint32).reshape(img_shape)

print ("average zlib percent: "+ str(z_lib/float(total_count)))
print ("average improved zlib percent: "+ str(improved_zlib/float(total_count)))
print ('zlib wins:' + str(z_lib_win_count))
print ('improved zlib wins:' + str(improved_zlib_win_count))
#print ' hebele' + str(z_lib/float(5)), improved_zlib