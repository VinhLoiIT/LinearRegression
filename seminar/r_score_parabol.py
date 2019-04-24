import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return -3*x**2+4*x-1
    # return 2*x

def least_square_estimation(x:np.ndarray,y:np.ndarray):
    inv = np.linalg.inv(x.transpose().dot(x))
    return inv.dot(x.transpose()).dot(y)
    # return x.transpose().dot(x).inv().dot(x.transpose).dot(y)

seed = 42
np.random.seed(seed) # make repeate random values
# rand_error = np.random.randn(10)
# print(rand_error)

x = np.linspace(-10,10,50)
y = f(x)
z_0 = np.ones((50))

Z = np.vstack((z_0,x)).transpose()
Y = y.transpose()
b_hat = least_square_estimation(Z,Y)
print(b_hat.shape)
print(b_hat)

plt.scatter(x,y)
y_plot = np.vstack((np.ones((50)), x)).transpose().dot(b_hat)
print(y_plot)
plt.plot(x,y_plot, color='r')
# plt.scatter(x,y_plot, color='g')


r_squares = np.var(y_plot)/np.var(y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt.text(-10,-2,"r-squares = {:.02f}".format(r_squares), bbox=props)
print(r_squares)
# plt.legend()

plt.show()

Z_2 = np.vstack((z_0,x,x**2)).transpose()
Y = y.transpose()
b_hat = least_square_estimation(Z_2, Y)
# print(b_hat.reshape(1,-1))
print(b_hat)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=x, ys=x**2, zs= f(x), s=40, alpha=1, edgecolors='w', color='r')

xx1, xx2 = np.meshgrid(
    np.linspace(x.min(), x.max(), num=50),
    np.linspace((x**2).min(), (x**2).max(), num=50)
)
XX = np.column_stack([np.ones(50**2), xx1.ravel(), xx2.ravel()])

print("xx", XX.shape)
Z = XX.transpose()
print(Z.shape)
yy = b_hat.dot(Z)
ax.plot_surface(xx1, xx2, yy.reshape(xx1.shape), color='b')

r_squares = np.var(yy)/np.var(Y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(-10,-2,5,s="r-squares = {:.02f}".format(r_squares), bbox=props)
# ax.set_xlabel(x.columns[0])
# ax.set_ylabel(x.columns[1])
# ax.set_zlabel(y.name)

plt.show()