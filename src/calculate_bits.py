import cv2
import os
import numpy as np
path = os.path.abspath("data/")
for i in range(32, 34):
    img_path = path+"/test_tiff"+str(i)+".tif"
    print img_path
    img = cv2.imread(img_path, -1)
    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    if img is not None:
        x = np.asarray(img)
        buffer = np.getbuffer(x)
        #print len(buffer)*8
        bits = (len(buffer)*8)/(img.shape[1]*img.shape[0])
        print bits
