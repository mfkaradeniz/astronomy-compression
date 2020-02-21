# import matplotlib.pyplot as plt
# import numpy as np
# import scipy

# ## Data
# x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
# y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

# ## Polyfit
# z = np.polyfit(x, y, deg=3)
# p = np.poly1d(z)

# ## Plot
# xp = np.linspace(-2, 6, 100)
# plt.figure(figsize=(6.5,4))
# plt.plot(x,y,'o',label='data')
# plt.plot(xp, p(xp),label='polyfit')
# plt.show()

#X = [[0], [1], [2], [3]]
#y = [0, 0, 1, 1]
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import scipy
import os, cv2

## Regressor
#x = np.array([[0.0], [1.0], [2.0], [3.0],  [4.0],  [5.0]])
#y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
path = os.path.abspath("../data/heic1509a.jpg")
img = cv2.imread(path,0)
#img = cv2.resize(img, (100,100))

x = np.arange(len(img.ravel())).reshape(-1,1)
y = img.ravel()

#regressor = SVR(kernel="rbf", C=100)
regressor = KNeighborsRegressor()
regressor.fit(x, y)
print(regressor.predict([[3.5]]))


## Polynomial Curve Fitting
x = np.array(x)
z = np.polyfit(x.ravel(), y, deg=3)
p = np.poly1d(z)

## Plot
# xp = np.linspace(0, len(x), 100)
# xp_reshaped = xp.reshape(-1, 1)
# plt.figure(figsize=(6.5,4))
# plt.plot(x,y,'o',label='data')
# plt.plot(xp, regressor.predict(xp_reshaped), label='regressor')
# plt.plot(xp, p(xp),label='polyfit')
# plt.legend(loc="upper left")
# plt.show()