import cv2
import os
import runlength
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math
import zlib
from sklearn.metrics import mean_squared_error
path = os.path.abspath("data/top100/test_tiff63.tiff")
img_path = path
img = cv2.imread(img_path,0)
img_shape = img.shape

x = np.asarray(img.ravel()).astype(np.uint8)
original_image = np.getbuffer(x)
y = zlib.compress(original_image)
print y.shape[:2]




y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
#print mean_squared_error(original_image, zlib_image)