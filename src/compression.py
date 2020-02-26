import cv2
import os
import runlength
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Lars, LogisticRegression, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
import glob
import math
import zlib
import pickle
import sys

def poly_compress(arr, k, deg, regressor):
    differences = []
    polynomial_coefficients = []
    n_poly = len(arr)//k
    for i in range(0, len(arr), n_poly):
        inc = n_poly
        if i+n_poly > len(arr):
            inc = len(arr)-i
        x = np.arange(inc)
        y = arr[i:i+inc]
        if(deg > inc):
            deg = 0

        x_reshaped = x.reshape(-1, 1)
        regressor = regressor.fit(x_reshaped, y)
        z = pickle.dumps(regressor)
        polynomial_coefficients.append(z)
        

        diff = regressor.predict(x_reshaped).astype(np.int8) - y
        differences += list(diff)
    return (np.asarray(differences), polynomial_coefficients)


def poly_decompress(differences, coefficients, k):
    # Revert differences.
    n_poly = len(differences)//k
    decoded_values = []
    for i in range(0, len(differences), n_poly):
        inc = n_poly
        if i+n_poly > len(differences):
            inc = len(differences)-i
        z = coefficients[i//n_poly]
        regressor = pickle.loads(z)
        diff = differences[i:i+inc]
        x = np.arange(inc)
        x_reshaped = x.reshape(-1, 1)
        decoded_values += list(regressor.predict(x_reshaped).astype(np.int8)-diff)
    return np.asarray(decoded_values)


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    win_height = nrows
    win_width = ncols
    vecs = []
    for r in range(0,img.shape[0], win_height):
        for c in range(0,img.shape[1], win_width):
            window = img[r:r+win_height,c:c+win_width]
            vecs.append(window)
    return np.asarray(vecs)


def merge(array, nrows, ncols):
    result = []

    height, width = img_shape
    v = []
    for i in range(0, height//(nrows)):
        h = np.empty((nrows,ncols),dtype=np.int8)
        h2 = []
        for j in range(0, width//(ncols)):
            window = array[i*(width//ncols)+j].reshape(nrows,ncols)
            
            h = np.hstack((h, window))
            h2.append(window)
        h = np.hstack(np.asarray(h2))
        v.append(h)
    v = np.vstack(np.asarray(v))
    return v

def poly_compress_grid(arr, n, m, deg, regressor):
    splitted = split(arr, n, m)
    differences = []
    coeffs = []
    counter = 0
    for i in range(len(splitted)):
        grid_arr = splitted[i]
        diff, coeff = poly_compress(grid_arr.ravel(), 1, deg, regressor)
        differences.append(list(diff))
        coeffs.append(np.asarray(coeff).ravel())
        counter += 1
        #print(counter, len(splitted))
    return (np.asarray(differences), np.asarray(coeffs))

def poly_decompress_grid(differences, coefficients, n, m):
    decoded_values = []
    for i in range(len(differences)):
        diff = np.asarray(differences[i]).ravel()
        coeff = coefficients[i]
        z = coeff
        regressor = pickle.loads(z)
        x = np.arange(len(diff))
        x_reshaped = x.reshape(-1, 1)
        decoded_values += list(regressor.predict(x_reshaped).astype(np.int8)-diff)

    decoded_values = np.reshape(decoded_values, img_shape)
    decoded_values = split(decoded_values, n, m)
    decoded_values = merge(decoded_values,n ,m)

    return np.asarray(decoded_values)



## Define the regressor.
regressor = LinearRegression()

## Choose grid length
n1 = 30
n2 = 30
print("n1=%d, n2=%d, model=%s" % (n1,n2,regressor))

file_paths = glob.glob("../data/*.tif")

z_lib = 0
improved_zlib = 0

z_lib_win_count = 0
improved_zlib_win_count = 0
total_count = 0

for path in file_paths:
    # add test image path
    #path = os.path.abspath("../data/heic1509a.tif")

    ## Read image.
    img_path = path
    print(img_path)
    img = cv2.imread(img_path,0)
    img_shape = img.shape

    ## Compress with Zlib.
    x = np.asarray(img.ravel()).astype(np.uint8)
    original_image = pickle.dumps(x)
    y = zlib.compress(original_image)
    compress_ratio_zlib = (float(len(original_image)) - float(len(y))) / float(len(original_image))
    compress_ratio_percent_zlib = 100.0 * compress_ratio_zlib
    print('Compressed zlib: %f%%' % (100.0 * compress_ratio_zlib))

    ## Compress with fitting.
    values = img.ravel().astype(np.uint8)
    k = 10000
    deg = 4

    ## Compute grid shape.
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
    n = img.shape[0]//n1
    m = img.shape[1]//n2

    ## Compute differences and coefficients..
    differences, polynomial_coefficients = poly_compress_grid(img, n, m, deg, regressor)
    differences = differences.astype(np.int8)
    x = np.asarray(img.ravel()).astype(np.uint8)
    original_image = pickle.dumps(x)

    # Compress differences
    x = np.asarray(differences).astype(np.int8)
    buffer = pickle.dumps(x)
    y = zlib.compress(buffer)

    # Compress coefficients
    x = np.asarray(polynomial_coefficients)
    buffer = pickle.dumps(x)
    y2 = zlib.compress(buffer)
    compress_ratio_reg = (float(len(original_image)) - float(len(y)+len(y2))) / float(len(original_image))
    compress_ratio_percent_reg = 100.0 * compress_ratio_reg
    print('Compressed zlib_regressoion: %f%%' % (100.0 * compress_ratio_reg))

    ## Check losssless.
    decoded_values = poly_decompress_grid(differences, polynomial_coefficients, n1, n2)
    decoded_values = np.asarray(decoded_values).astype(np.uint8)

    ## Compare with zlib.
    z_lib += compress_ratio_zlib
    improved_zlib += compress_ratio_reg
    if compress_ratio_zlib > compress_ratio_percent_reg:
        z_lib_win_count += 1
        print('WARNING: zlib won!' + 'this is the image: ' + path)
    elif compress_ratio_zlib < compress_ratio_percent_reg:
        improved_zlib_win_count += 1

    total_count += 1


print("average zlib percent: "+ str(z_lib/float(total_count)))
print("average improved zlib percent: "+ str(improved_zlib/float(total_count)))
print('zlib wins:' + str(z_lib_win_count))
print('improved zlib wins:' + str(improved_zlib_win_count))