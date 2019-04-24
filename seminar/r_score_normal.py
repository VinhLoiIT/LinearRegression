import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

def least_square_estimation(z:np.ndarray,y:np.ndarray):
    inv = np.linalg.inv(z.transpose().dot(z))
    return inv.dot(z.transpose()).dot(y)

seed = 42
np.random.seed(seed) # make repeate random values

n = 20
z = np.linspace(-10,10,n).transpose().reshape(-1,1)
epsilon = np.random.randn(n,1) # n rows, 1 column
y = 2*z + epsilon

# plot data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Linear Regression")
ax.set_xlabel("z")
ax.set_ylabel("y")
ax.scatter(z,y, label="real data")
ax.legend()
fig.savefig("real_data.png")


z_0 = np.ones((n,1))
Z = np.hstack((z_0,z))
Y = y
b_hat = least_square_estimation(Z,Y)
print("Beta =", b_hat)

# plot line
y_plot = Z.dot(b_hat)
b_hat_ravel = b_hat.ravel()
ax.plot(z,y_plot, color='r', label="y = {:.02f} + {:.02f}z".format(b_hat_ravel[0], b_hat_ravel[1]))
ax.legend()
fig.savefig("least_square_estimated.png")

# show r_square score
r_squares = np.var(y_plot)/np.var(y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(-10,-2,"r-squares={:.02f}".format(r_squares), bbox=props)
fig.savefig("r_square_1.png")

plt.show()