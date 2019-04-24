import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return -3*x**2+4*x-1

def least_square_estimation(x:np.ndarray,y:np.ndarray):
    inv = np.linalg.inv(x.transpose().dot(x))
    return inv.dot(x.transpose()).dot(y)
    # return x.transpose().dot(x).inv().dot(x.transpose).dot(y)

seed = 42
np.random.seed(seed) # make repeate random values
# rand_error = np.random.randn(10)
# print(rand_error)

n = 20
z_1 = np.linspace(-10,10,n).transpose().reshape(-1,1)
epsilon = np.random.randn(n,1) # n rows, 1 column
y = -3*z_1**2+4*z_1-1 #+ epsilon

# plot data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Linear Regression")
ax.set_xlabel("z")
ax.set_ylabel("y")
ax.scatter(z_1,y, label="real data")
ax.legend()
fig.savefig("real_data_2.png")

z_0 = np.ones((n,1))
Z = np.hstack((z_0,z_1))
Y = y
b_hat = least_square_estimation(Z,Y)
print("Beta =", b_hat)

y_plot = Z.dot(b_hat)
b_hat_ravel = b_hat.ravel()
ax.plot(z_1,y_plot, color='r', label="y = {:.02f} + {:.02f}z".format(b_hat_ravel[0], b_hat_ravel[1]))
ax.legend()
fig.savefig("least_square_estimated_2.png")

r_squares = np.var(y_plot)/np.var(y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(-10,-2,"r-squares={:.02f}".format(r_squares), bbox=props)
fig.savefig("r_square_2.png")

z_2 = z_1**2
Z = np.hstack((Z, z_2))
b_hat = least_square_estimation(Z, Y)
# print(b_hat.reshape(1,-1))
print("Beta_2 =",b_hat)

plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=z_1, ys=z_2, zs=y, s=40, alpha=1, edgecolors='w', color='r')
fig.savefig("real_data_2_after.png")

xx1, xx2 = np.meshgrid(
    np.linspace(z_1.min(), z_1.max(), num=n),
    np.linspace(z_2.min(), z_2.max(), num=n)
)
XX = np.column_stack([np.ones((n,n)).ravel(), xx1.ravel(), xx2.ravel()])

Z = XX
yy = Z.dot(b_hat)
ax.plot_surface(xx1, xx2, yy.reshape(xx1.shape))
fig.savefig("least_square_estimated_2_after.png")

r_squares = np.var(yy)/np.var(Y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(-10,-2,5,s="r-squares = {:.02f}".format(r_squares), bbox=props)
ax.set_xlabel("z")
ax.set_ylabel("z^2")
ax.set_zlabel("y")
fig.savefig("r_square_2_after.png")

plt.show()