import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

# create data
n = 7
r = 2
z_0 = np.ones((n,1))
z_1 = np.array([123.5,146.1,133.9,128.5,151.5,136.2,92.0]).reshape(-1,1)
z_2 = np.array([2.108,9.213,1.905,0.815,1.061,8.603,1.125]).reshape(-1,1)
y = np.array([141.5,168.9,154.8,146.5,172.8,160.1,108.5]).reshape(-1,1)


# plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Linear Regression")
ax.set_xlabel(r"$z_1$")
ax.set_ylabel(r"$z_2$")
ax.set_zlabel("y")

ax.scatter(xs=z_1, ys=z_2, zs=y, label="real_data")
ax.legend()
fig.savefig("example_data.png")

# least squares estimation
Z = np.hstack((z_0, z_1, z_2))
Y = y
b_hat = np.linalg.inv(Z.transpose().dot(Z)).dot(Z.transpose()).dot(Y)
print("Beta hat =", b_hat)

# plot surface
xx1, xx2 = np.meshgrid(
    np.linspace(z_1.min(), z_1.max(), num=n),
    np.linspace(z_2.min(), z_2.max(), num=n)
)
XX = np.column_stack([np.ones((n,n)).ravel(), xx1.ravel(), xx2.ravel()])

yy = XX.dot(b_hat)
ax.plot_surface(xx1, xx2, yy.reshape(xx1.shape), color="red", alpha=.1)
fig.savefig("least_square_estimated_example.png")

# calculate R-square score
y_hat = Z.dot(b_hat)
r_squares = np.var(y_hat)/np.var(Y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(z_1.max(),z_2.max(),y.max(),s="r-squares = {:.02f}".format(r_squares), bbox=props)
fig.savefig("r_square_example.png")

# Predict
epsilon_hat = Y - y_hat
s_square = epsilon_hat.transpose().dot(epsilon_hat)/(n-r-1)
print("s_squares =", s_square)

z_new = np.array([1,130,7.5]).reshape(-1,1)
# Estimate mean
y_mean = z_new.transpose().dot(b_hat)
print("y_mean",y_mean)
ax.scatter(xs=z_new[1][0], ys=z_new[2][0], zs=y_mean, label="y_mean", color='g')
alpha = 0.05
t_prop = stats.t.ppf(alpha/2, df=n-r-1)
print("student", t_prop)
std = np.sqrt(
    z_new.transpose().dot(
        np.linalg.inv(Z.transpose().dot(Z))
    ).dot(z_new)*s_square
)
print("s...", std)
std = t_prop * std
print("std", std)

print("confident interval 100(1-{})% of y_mean is: {:.02f} +- {:.02f}\n or ({:.02f}, {:.02f})".format(
    alpha, y_mean[0][0], np.abs(std)[0][0],
    (y_mean+std)[0][0], (y_mean-std)[0][0])
)

# Plot confidence
ax.plot(
    [z_new[1][0],z_new[1][0]],
    [z_new[2][0],z_new[2][0]],
    [(y_mean + std)[0][0], (y_mean - std)[0][0]]
)

# cau b
std = np.sqrt(s_square*(1+z_new.transpose().dot(
        np.linalg.inv(Z.transpose().dot(Z))
    ).dot(z_new))
)
print("s...", std)
std = t_prop * std
print("std", std)
print("confident interval 100(1-{})% of y is: {:.02f} +- {:.02f}\n or ({:.02f}, {:.02f})".format(
    alpha, y_mean[0][0], np.abs(std)[0][0],
    (y_mean+std)[0][0], (y_mean-std)[0][0])
)

plt.show()

