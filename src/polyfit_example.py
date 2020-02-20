import matplotlib.pyplot as plt
import numpy as np
import scipy

## Data
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

## Polyfit
z = np.polyfit(x, y, deg=3)
p = np.poly1d(z)

## Plot
xp = np.linspace(-2, 6, 100)
plt.figure(figsize=(6.5,4))
plt.plot(x,y,'o',label='data')
plt.plot(xp, p(xp),label='polyfit')
plt.show()