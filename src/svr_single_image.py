import cv2
import os
import runlength
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Lars, LogisticRegression, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
import math
import zlib
import pickle
import sys


# add test image path
path = os.path.abspath("../data/heic1509a.tif")
z_lib = 0
improved_zlib = 0

z_lib_win_count = 0
improved_zlib_win_count = 0

## Read image.
img_path = path
print(img_path)
img = cv2.imread(img_path,0)
#img = cv2.resize(img, (3000,3000))
#img = img[:img.shape[0]//10, :img.shape[1]//10]
img_shape = img.shape
print(img_shape)

## Compute Zlib compression ratio

## Compress image.
x = np.asarray(img.ravel()).astype(np.uint8)

#original_image = np.getbuffer(x)
original_image = pickle.dumps(x)
y = zlib.compress(original_image)
compress_ratio = (float(len(original_image)) - float(len(y))) / float(len(original_image))
compress_ratio_percent_zlib = 100.0 * compress_ratio

compress_ratio2 = float(len(original_image))/ (float(len(y)))
print('Compressed zlib: %f%%' % (100.0 * compress_ratio))

values = img.ravel().astype(np.uint8)



def poly_compress(arr, k, deg):
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

        #regressor = SVR(kernel="rbf", C=1)
        #regressor = SGDRegressor()
        #regressor = GaussianProcessRegressor()
        #regressor = BayesianRidge()
        regressor = LinearRegression()
        #regressor = TheilSenRegressor()
        x_reshaped = x.reshape(-1, 1)
        #print(x_reshaped)
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

def poly_compress_grid(arr, n, m, deg):
    splitted = split(arr, n, m)
    differences = []
    coeffs = []
    counter = 0
    for i in range(len(splitted)):
        grid_arr = splitted[i]
        diff, coeff = poly_compress(grid_arr.ravel(), 1, deg)
        differences.append(list(diff))
        coeffs.append(np.asarray(coeff).ravel())
        counter += 1
        print(counter, len(splitted))
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


k = 10000
deg = 4

n1 = 30
n2 = 30

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


differences, polynomial_coefficients = poly_compress_grid(img, n, m, deg)
differences = differences.astype(np.int8)

## Compress.
x = np.asarray(img.ravel()).astype(np.uint8)
#original_image = np.getbuffer(x)
original_image = pickle.dumps(x)

# compress differences
x = np.asarray(differences).astype(np.int8)
#buffer = np.getbuffer(x)
buffer = pickle.dumps(x)
y = zlib.compress(buffer)

#compress polynomial coefficients
x = np.asarray(polynomial_coefficients)
#buffer = np.getbuffer(x)
buffer = pickle.dumps(x)
y2 = zlib.compress(buffer)

compress_ratio = (float(len(original_image)) - float(len(y)+len(y2))) / float(len(original_image))
compress_ratio_imp_zlib = 100.0 * compress_ratio





print('Compressed zlib_polyfit: %f%%' % (100.0 * compress_ratio))


compress_ratio3 = float(len(original_image))/ (float(len(y)))


decoded_values = poly_decompress_grid(differences, polynomial_coefficients, n1, n2)
decoded_values = np.asarray(decoded_values).astype(np.uint8)
